"""Rank shortlists of description variants with a shopping-agent model.

Each trial is deterministic from its id: the intent and panel cycle, then a
seeded draw picks one arm per product and shuffles display order. The run
appends one JSON line per completed trial and can be stopped on a clock or
resumed by skipping trial ids already on disk.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from src.config import (
    CONCURRENCY,
    DATA_DIR,
    INTENTS,
    MAIN_RUN_ARMS,
    RANK_MODEL,
)
from src.jsonl import read_jsonl
from src.llm import Caller, run_bounded

app = typer.Typer(add_completion=False)
console = Console()

RANK_ATTEMPTS = 3

RANK_SYSTEM = (
    "You are a shopping assistant. Rank every option from best to worst for the "
    "shopper. Weigh the description together with the title, brand, and price. "
    "Return every option id exactly once, best first. "
    "Return JSON only. The object has one key, ranking, whose value lists every "
    "option id from best to worst."
)


def trial_grid(
    n_trials: int,
    intents: list[str],
    panel_ids: list[str],
) -> list[tuple[str, str, int]]:
    """Cycle intents, then panels, so a short run stays balanced.

    Returns (intent, panel_id, rep) with a unique triple per trial.
    """
    if n_trials < 0:
        raise ValueError("n_trials must be >= 0")
    if not intents or not panel_ids:
        return []
    n_intents = len(intents)
    n_panels = len(panel_ids)
    grid: list[tuple[str, str, int]] = []
    for index in range(n_trials):
        intent = intents[index % n_intents]
        panel_id = panel_ids[(index // n_intents) % n_panels]
        rep = index // (n_intents * n_panels)
        grid.append((intent, panel_id, rep))
    return grid


def trial_id(intent: str, panel_id: str, rep: int) -> str:
    return f"{intent}-{panel_id}-{rep:05d}"


def draw_assignment(
    seed: str,
    asins: list[str],
    arms: list[str],
) -> tuple[dict[str, str], list[str]]:
    """One uniform arm per product, then a shuffled display order."""
    rng = random.Random(seed)
    chosen = {asin: rng.choice(list(arms)) for asin in asins}
    order = list(asins)
    rng.shuffle(order)
    return chosen, order


def canonical_option_id(value: Any) -> str:
    """Map the id formats models actually return back to O1..O8."""
    text = str(value).strip().upper()
    if text.startswith("OPTION"):
        text = text[len("OPTION") :].strip()
    if text.isdigit():
        text = f"O{int(text)}"
    return text


def parse_ranking(payload: dict[str, Any], option_ids: list[str]) -> Optional[list[str]]:
    """A ranking is valid only when it is a permutation of every option id."""
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list):
        return None
    cleaned = [canonical_option_id(item) for item in ranking]
    if len(cleaned) != len(option_ids) or len(set(cleaned)) != len(option_ids):
        return None
    if set(cleaned) != set(option_ids):
        return None
    return cleaned


def rank_user(shopper: str, cards: list[dict[str, Any]]) -> str:
    blocks = [f"Shopper: {shopper}", "", "Options:"]
    for card in cards:
        blocks.append(
            f"Option {card['option_id']}\n"
            f"Title: {card['title']}\n"
            f"Brand: {card['brand']}\n"
            f"Price: ${card['price']:.2f}\n"
            f"Description: {card['text']}\n"
        )
    ids = ", ".join(card["option_id"] for card in cards)
    blocks.append(f"Rank these ids exactly once, best first: {ids}. Return JSON only.")
    return "\n".join(blocks)


def option_cards(
    order: list[str],
    assignment: dict[str, str],
    products: dict[str, dict[str, Any]],
    texts: dict[tuple[str, str], str],
) -> Optional[list[dict[str, Any]]]:
    """Build display cards. `texts` is keyed by (parent_asin, arm)."""
    cards: list[dict[str, Any]] = []
    for position, asin in enumerate(order):
        arm = assignment[asin]
        text = texts.get((asin, arm))
        if not text or asin not in products:
            return None
        product = products[asin]
        price = product.get("price") or 0
        cards.append(
            {
                "option_id": f"O{position + 1}",
                "parent_asin": asin,
                "arm": arm,
                "position": position,
                "title": product.get("title") or "",
                "brand": product.get("store") or "Unknown",
                "price": float(price),
                "rating": product.get("average_rating"),
                "reviews": product.get("rating_number"),
                "text": text,
            }
        )
    if len(cards) != len(order):
        return None
    return cards


def stored_options(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "option_id": card["option_id"],
            "parent_asin": card["parent_asin"],
            "arm": card["arm"],
            "position": card["position"],
        }
        for card in cards
    ]


def load_variant_texts(path: Path) -> dict[tuple[str, str, str], str]:
    texts: dict[tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        asin = row.get("parent_asin")
        intent = row.get("intent")
        arm = row.get("arm")
        text = row.get("text")
        if asin and intent and arm and text:
            texts[(asin, intent, arm)] = text
    return texts


def load_panels(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_products(path: Path) -> dict[str, dict[str, Any]]:
    return {row["parent_asin"]: row for row in read_jsonl(path)}


@app.command()
def main(
    n_trials: int = typer.Option(..., help="How many trials to attempt"),
    max_minutes: Optional[float] = typer.Option(None, help="Stop cleanly after this many minutes"),
    concurrency: int = typer.Option(CONCURRENCY),
    model: str = typer.Option(RANK_MODEL),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    limit_panels: int = typer.Option(0, help="Use only the first N panels (smoke test)"),
    include_probe: bool = typer.Option(False, "--include-probe", help="Draw the probe arm too"),
    temperature: float = typer.Option(0.0),
    products_path: Path = typer.Option(DATA_DIR / "study_products.jsonl", "--products"),
    panels_path: Path = typer.Option(DATA_DIR / "panels.json", "--panels"),
    variants_path: Path = typer.Option(DATA_DIR / "variants.jsonl", "--variants"),
    out: Path = typer.Option(DATA_DIR / "rankings.jsonl"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENROUTER_API_KEY"),
) -> None:
    """Run shopping-agent ranking trials."""
    panels = load_panels(panels_path)
    if limit_panels:
        panels = panels[:limit_panels]
    products = load_products(products_path)
    all_texts = load_variant_texts(variants_path)
    if not all_texts:
        raise typer.BadParameter(f"No variants in {variants_path}. Run generate_variants first.")

    arms = list(MAIN_RUN_ARMS)
    if include_probe:
        arms.append("probe")
    elif any(arm == "probe" for _, _, arm in all_texts):
        console.print("[dim]Probe variants are on disk. Pass --include-probe to draw them.[/dim]")

    intents = list(INTENTS)
    panel_ids = [panel["panel_id"] for panel in panels]
    by_id = {panel["panel_id"]: panel for panel in panels}
    grid = trial_grid(n_trials, intents, panel_ids)

    if not resume and out.exists():
        out.unlink()
    done = {row["trial_id"] for row in read_jsonl(out) if row.get("trial_id")}
    pending = [
        (intent, panel_id, rep)
        for intent, panel_id, rep in grid
        if trial_id(intent, panel_id, rep) not in done
    ]
    console.print(
        f"[cyan]{len(pending)} trials to run[/cyan] "
        f"({len(done)} already on disk) using {model}"
    )
    if not pending:
        console.print(f"[green]{len(done)} rankings on disk[/green] → {out}")
        return

    runnable: list[tuple[str, str, int]] = []
    missing = 0
    for intent_key, panel_id, rep in pending:
        tid = trial_id(intent_key, panel_id, rep)
        asins = list(by_id[panel_id]["parent_asins"])
        assignment, _order = draw_assignment(tid, asins, arms)
        if all(all_texts.get((asin, intent_key, assignment[asin])) for asin in asins):
            runnable.append((intent_key, panel_id, rep))
        else:
            missing += 1
    if missing:
        console.print(
            f"[yellow]{missing} trials are missing a variant and were not started.[/yellow] "
            "They will run once those descriptions exist."
        )
    if not runnable:
        raise typer.BadParameter(
            "Every trial is missing variant text. "
            "Generate variants for the products in these panels first."
        )
    pending = runnable

    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("a", encoding="utf-8")
    drops: list[tuple[str, str]] = []

    def write(row: dict[str, Any]) -> None:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()

    caller = Caller(
        model=model,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=300,
        api_key=api_key,
    )
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60

    async def work(spec: tuple[str, str, int]) -> Optional[dict[str, Any]]:
        intent_key, panel_id, rep = spec
        tid = trial_id(intent_key, panel_id, rep)
        asins = list(by_id[panel_id]["parent_asins"])
        assignment, order = draw_assignment(tid, asins, arms)
        texts = {
            (asin, arm): all_texts.get((asin, intent_key, arm), "")
            for asin, arm in assignment.items()
        }
        cards = option_cards(order, assignment, products, texts)
        if cards is None:
            drops.append((tid, "missing variant"))
            return None
        user = rank_user(INTENTS[intent_key]["shopper"], cards)
        option_ids = [card["option_id"] for card in cards]
        prompt = user
        last = "no attempt"
        for _ in range(RANK_ATTEMPTS):
            try:
                payload = await caller.json(RANK_SYSTEM, prompt)
            except Exception as exc:  # noqa: BLE001
                last = str(exc)
                continue
            ranking = parse_ranking(payload, option_ids)
            if ranking is None:
                last = "ranking is not a permutation of the option ids"
                # Temperature 0 would otherwise repeat the same bad ranking.
                prompt = (
                    f"{user}\n"
                    "The previous reply was rejected. Return JSON whose ranking "
                    f"contains each of these ids exactly once, best first: {', '.join(option_ids)}."
                )
                continue
            return {
                "trial_id": tid,
                "intent": intent_key,
                "panel_id": panel_id,
                "model": model,
                "options": stored_options(cards),
                "ranking": ranking,
            }
        drops.append((tid, last))
        return None

    _, left = asyncio.run(
        run_bounded(
            pending,
            work,
            concurrency=concurrency,
            desc="rankings",
            deadline=deadline,
            on_result=write,
        )
    )
    sink.close()
    written = len(read_jsonl(out))
    console.print(f"[green]{written} rankings on disk[/green] → {out}")
    if left:
        console.print(f"[yellow]Stopped on the clock with {left} trials not started.[/yellow]")
    if drops:
        counts: dict[str, int] = {}
        for _, reason in drops:
            key = "missing variant" if reason == "missing variant" else reason.split(":")[0][:120]
            counts[key] = counts.get(key, 0) + 1
        console.print(f"[yellow]Dropped {len(drops)} trials[/yellow]")
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            console.print(f"  {count} × {reason}")


if __name__ == "__main__":
    app()
