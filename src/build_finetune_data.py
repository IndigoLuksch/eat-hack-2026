"""Build SFT and DPO sets from the shrunk arm scores.

Hold out products by parent_asin before either set is built, so a product
cannot leak through a second intent. Each remaining product × intent cell
contributes its best arm, and its second-best arm only when that arm beats the
untouched Amazon description. DPO pairs are the top-versus-bottom and the
second-versus-fifth arms, and only when the score gap clears the threshold.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from src.config import (
    DATA_DIR,
    DPO_MIN_GAP,
    HOLDOUT_PRODUCTS,
    HOLDOUT_SEED,
    INTENTS,
    MAIN_RUN_ARMS,
    SFT_SYSTEM,
)
from src.jsonl import read_jsonl

app = typer.Typer(add_completion=False)
console = Console()

MAIN_ARMS = set(MAIN_RUN_ARMS)


def holdout_count(n_products: int, requested: int) -> int:
    """10% of the catalogue, capped at the requested count.

    Fewer than ten products is a smoke test: hold nothing out so the training
    file is not emptied.
    """
    if requested <= 0 or n_products < 10:
        return 0
    return min(requested, n_products // 10)


def select_holdout(asins: list[str], n: int, seed: int = HOLDOUT_SEED) -> list[str]:
    if n <= 0:
        return []
    if n >= len(asins):
        raise ValueError(f"holdout {n} leaves no training products out of {len(asins)}")
    chosen = random.Random(seed).sample(sorted(asins), n)
    return sorted(chosen)


def sft_user(product: dict[str, Any], intent_key: str) -> str:
    """Facts the model is allowed to see. The original description is omitted."""
    intent = INTENTS[intent_key]
    features = " | ".join(product.get("features") or []) or "(none listed)"
    brand = product.get("store") or "Unknown"
    price = product.get("price")
    price_s = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    return (
        f"Intent: {intent['label']}\n"
        f"Brief: {intent['brief']}\n"
        f"Title: {product.get('title')}\n"
        f"Brand: {brand}\n"
        f"Price: {price_s}\n"
        f"Rating: {product.get('average_rating')} from {product.get('rating_number')} reviews\n"
        f"Bullet features: {features}\n"
        "Write the product description."
    )


def sft_messages(product: dict[str, Any], intent_key: str, description: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": sft_user(product, intent_key)},
        {"role": "assistant", "content": description},
    ]


def prompt_messages(product: dict[str, Any], intent_key: str) -> list[dict[str, str]]:
    return sft_messages(product, intent_key, "")[:-1]


def ordered_arms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The six main arms, best score first. Empty when any arm is missing."""
    by_arm = {row["arm"]: row for row in rows if row.get("arm") in MAIN_ARMS}
    if set(by_arm) != MAIN_ARMS:
        return []
    return sorted(by_arm.values(), key=lambda row: (float(row["score"]), row["arm"]))


def split_by_asin(
    rows: list[dict[str, Any]],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows on parent_asin so one product stays entirely on one side."""
    asins = sorted({row["parent_asin"] for row in rows})
    if len(asins) < 2 or eval_fraction <= 0:
        return rows, []
    n_eval = max(1, int(round(len(asins) * eval_fraction)))
    n_eval = min(n_eval, len(asins) - 1)
    eval_asins = set(random.Random(seed).sample(asins, n_eval))
    train = [row for row in rows if row["parent_asin"] not in eval_asins]
    held = [row for row in rows if row["parent_asin"] in eval_asins]
    return train, held


def build_records(
    scores: dict[str, Any],
    products: dict[str, dict[str, Any]],
    variants: dict[tuple[str, str, str], str],
    *,
    holdout_n: int = HOLDOUT_PRODUCTS,
    holdout_seed: int = HOLDOUT_SEED,
    min_gap: float = DPO_MIN_GAP,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], dict[str, int]]:
    asins = sorted(asin for asin in scores if asin in products)
    n_holdout = holdout_count(len(asins), holdout_n)
    holdout = select_holdout(asins, n_holdout, holdout_seed)
    held = set(holdout)
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    stats = {
        "cells": 0,
        "cells_skipped": 0,
        "sft_rank2_dropped": 0,
        "dpo_pairs": 0,
    }

    for asin in asins:
        if asin in held:
            continue
        product = products[asin]
        for intent_key, rows in scores[asin].items():
            if intent_key not in INTENTS:
                continue
            ordered = ordered_arms(rows)
            if not ordered:
                stats["cells_skipped"] += 1
                continue
            texts = []
            missing = False
            for row in ordered:
                text = variants.get((asin, intent_key, row["arm"]))
                if not text:
                    missing = True
                    break
                texts.append(text)
            if missing:
                stats["cells_skipped"] += 1
                continue
            stats["cells"] += 1
            prompt = prompt_messages(product, intent_key)
            original = next(row for row in ordered if row["arm"] == "original")

            def emit_sft(rank_index: int) -> None:
                sft_rows.append(
                    {
                        "parent_asin": asin,
                        "intent": intent_key,
                        "arm": ordered[rank_index]["arm"],
                        "rank": rank_index + 1,
                        "messages": sft_messages(product, intent_key, texts[rank_index]),
                    }
                )

            emit_sft(0)
            if float(ordered[1]["score"]) < float(original["score"]):
                emit_sft(1)
            else:
                stats["sft_rank2_dropped"] += 1

            for chosen_i, rejected_i in ((0, 5), (1, 4)):
                gap = float(ordered[rejected_i]["score"]) - float(ordered[chosen_i]["score"])
                if gap <= min_gap:
                    continue
                dpo_rows.append(
                    {
                        "parent_asin": asin,
                        "intent": intent_key,
                        "chosen_arm": ordered[chosen_i]["arm"],
                        "rejected_arm": ordered[rejected_i]["arm"],
                        "gap": gap,
                        "prompt": prompt,
                        "chosen": [{"role": "assistant", "content": texts[chosen_i]}],
                        "rejected": [{"role": "assistant", "content": texts[rejected_i]}],
                    }
                )
                stats["dpo_pairs"] += 1

    stats["sft_examples"] = len(sft_rows)
    stats["holdout"] = len(holdout)
    return sft_rows, dpo_rows, holdout, stats


def variant_index(path: Path) -> dict[tuple[str, str, str], str]:
    index: dict[tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        asin = row.get("parent_asin")
        intent = row.get("intent")
        arm = row.get("arm")
        text = row.get("text")
        if asin and intent and arm and text:
            index[(asin, intent, arm)] = text
    return index


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.command()
def main(
    scores_path: Path = typer.Option(DATA_DIR / "variant_scores.json", "--scores"),
    products_path: Path = typer.Option(DATA_DIR / "study_products.jsonl", "--products"),
    variants_path: Path = typer.Option(DATA_DIR / "variants.jsonl", "--variants"),
    sft_out: Path = typer.Option(DATA_DIR / "sft.jsonl"),
    dpo_out: Path = typer.Option(DATA_DIR / "dpo.jsonl"),
    holdout_out: Path = typer.Option(DATA_DIR / "holdout_asins.json"),
    holdout: int = typer.Option(HOLDOUT_PRODUCTS),
    seed: int = typer.Option(HOLDOUT_SEED),
    min_gap: float = typer.Option(DPO_MIN_GAP, help="Minimum shrunk-score gap for a DPO pair"),
) -> None:
    """Write the SFT set, the DPO set, and the held-out product ids."""
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    products = {row["parent_asin"]: row for row in read_jsonl(products_path)}
    sft_rows, dpo_rows, held, stats = build_records(
        scores,
        products,
        variant_index(variants_path),
        holdout_n=holdout,
        holdout_seed=seed,
        min_gap=min_gap,
    )
    write_jsonl(sft_out, sft_rows)
    write_jsonl(dpo_out, dpo_rows)
    holdout_out.write_text(json.dumps(held, indent=2), encoding="utf-8")
    console.print(
        f"[green]{stats['sft_examples']} SFT examples[/green] "
        f"from {stats['cells']} cells "
        f"({stats['sft_rank2_dropped']} rank-2 targets dropped, "
        f"{stats['cells_skipped']} cells skipped)"
    )
    console.print(f"[green]{stats['dpo_pairs']} DPO pairs[/green] (min gap {min_gap})")
    console.print(f"[green]{len(held)} products held out[/green] → {holdout_out}")
    console.print(f"→ {sft_out}\n→ {dpo_out}")


if __name__ == "__main__":
    app()
