"""Score description arms by mean normalised rank with partial pooling.

No model is fit. Within each trial a product's normalised rank is its index in
the returned ranking divided by 7, so 0 is best and 1 is worst. Cell means
shrink toward the intent-by-arm pooled mean.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config import DATA_DIR, MAIN_RUN_ARMS, SHRINKAGE_PRIOR
from src.jsonl import read_jsonl

app = typer.Typer(add_completion=False)
console = Console()

Z_95 = 1.96


def normalised_rank(index: int, n_options: int) -> float:
    if n_options < 2:
        raise ValueError("a ranking needs at least two options")
    return index / (n_options - 1)


def shrink(cell_mean: float, n: int, pooled_mean: float, prior: float = SHRINKAGE_PRIOR) -> float:
    weight = n / (n + prior)
    return weight * cell_mean + (1.0 - weight) * pooled_mean


def mean_ci(values: list[float], z: float = Z_95) -> tuple[float, float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    margin = z * math.sqrt(variance / n)
    return mean, mean - margin, mean + margin


def score_rankings(
    trials: list[dict[str, Any]],
    prior: float = SHRINKAGE_PRIOR,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]], int]:
    """Return scores, a per-intent leaderboard, and how many trials were used.

    Scores are keyed by parent_asin, then intent, then arms ordered best-first
    (ascending shrunk score). The leaderboard is the unpooled mean normalised
    rank, which is also what the console report prints.
    """
    cell_sum: dict[tuple[str, str, str], float] = defaultdict(float)
    cell_n: dict[tuple[str, str, str], int] = defaultdict(int)
    arm_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    used = 0

    for trial in trials:
        ranking = trial.get("ranking")
        options = trial.get("options")
        intent = trial.get("intent")
        if not isinstance(ranking, list) or not isinstance(options, list) or not intent:
            continue
        if len(ranking) < 2:
            continue
        rank_of = {option_id: index for index, option_id in enumerate(ranking)}
        if len(rank_of) != len(ranking):
            continue
        n_options = len(ranking)
        complete = True
        contributions: list[tuple[str, str, float]] = []
        for option in options:
            option_id = option.get("option_id")
            asin = option.get("parent_asin")
            arm = option.get("arm")
            if option_id not in rank_of or not asin or not arm:
                complete = False
                break
            contributions.append((asin, arm, normalised_rank(rank_of[option_id], n_options)))
        if not complete or len(contributions) != n_options:
            continue
        used += 1
        for asin, arm, value in contributions:
            cell_sum[(asin, intent, arm)] += value
            cell_n[(asin, intent, arm)] += 1
            arm_values[(intent, arm)].append(value)

    pooled: dict[tuple[str, str], float] = {}
    for key, values in arm_values.items():
        pooled[key] = sum(values) / len(values)

    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (asin, intent, arm), total in cell_sum.items():
        n = cell_n[(asin, intent, arm)]
        cell_mean = total / n
        pooled_mean = pooled[(intent, arm)]
        by_cell[(asin, intent)].append(
            {
                "arm": arm,
                "score": shrink(cell_mean, n, pooled_mean, prior),
                "n": n,
                "cell_mean": cell_mean,
                "pooled_mean": pooled_mean,
            }
        )

    scores: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (asin, intent), rows in by_cell.items():
        rows.sort(key=lambda row: (row["score"], row["arm"]))
        scores.setdefault(asin, {})[intent] = rows

    leaderboard: dict[str, list[dict[str, Any]]] = {}
    intents = sorted({intent for intent, _arm in arm_values})
    for intent in intents:
        rows = []
        for (key_intent, arm), values in arm_values.items():
            if key_intent != intent:
                continue
            mean, lo, hi = mean_ci(values)
            rows.append({"arm": arm, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)})
        rows.sort(key=lambda row: (row["mean"], row["arm"]))
        for position, row in enumerate(rows, start=1):
            row["position"] = position
        leaderboard[intent] = rows

    return scores, leaderboard, used


def winning_generated_arm(
    scores: dict[str, Any],
    intent: str,
    exclude: tuple[str, ...] = ("original", "probe"),
) -> Optional[str]:
    """Arm with the lowest observation-weighted cell mean for this intent.

    `original` and `probe` are controls, so the prompt-only baseline copies the
    best rhetorical form rather than the baseline listing.
    """
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for product in scores.values():
        if not isinstance(product, dict):
            continue
        for row in product.get(intent) or []:
            arm = row.get("arm")
            if arm in exclude or arm not in MAIN_RUN_ARMS:
                continue
            n = float(row.get("n") or 0)
            totals[arm][0] += float(row["cell_mean"]) * n
            totals[arm][1] += n
    ranked = [(total / n, arm) for arm, (total, n) in totals.items() if n > 0]
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][1]


def _print_leaderboard(leaderboard: dict[str, list[dict[str, Any]]], trials_used: int) -> None:
    console.print(f"Scored {trials_used} trials. Lower mean normalised rank is better.")
    for intent, rows in leaderboard.items():
        table = Table(title=f"{intent} — mean normalised rank")
        table.add_column("rank", justify="right")
        table.add_column("arm")
        table.add_column("mean", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("n", justify="right")
        for row in rows:
            table.add_row(
                str(row["position"]),
                row["arm"],
                f"{row['mean']:.3f}",
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
                str(row["n"]),
            )
        console.print(table)
        original = next((row for row in rows if row["arm"] == "original"), None)
        if original:
            console.print(
                f"  original is rank {original['position']} of {len(rows)} on {intent}"
            )
        probe = next((row for row in rows if row["arm"] == "probe"), None)
        if probe:
            console.print(f"  probe is rank {probe['position']} of {len(rows)} on {intent}")


@app.command()
def main(
    rankings_path: Path = typer.Option(DATA_DIR / "rankings.jsonl", "--rankings"),
    out: Path = typer.Option(DATA_DIR / "variant_scores.json"),
    prior: float = typer.Option(SHRINKAGE_PRIOR),
) -> None:
    """Write shrunk arm scores and print the per-intent leaderboard."""
    trials = read_jsonl(rankings_path)
    if not trials:
        raise typer.BadParameter(f"No rankings in {rankings_path}.")
    scores, leaderboard, trials_used = score_rankings(trials, prior=prior)
    counts = [
        row["n"]
        for product in scores.values()
        for arms in product.values()
        for row in arms
    ]
    if counts:
        console.print(f"Median observations per cell: {statistics.median(counts):.0f}")
    _print_leaderboard(leaderboard, trials_used)
    payload = {asin: scores[asin] for asin in sorted(scores)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]{len(payload)} products[/green] → {out}")


if __name__ == "__main__":
    app()
