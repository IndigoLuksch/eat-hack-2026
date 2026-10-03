"""Generate intent-conditioned description variants for each study product.

For every product and intent there are six arms. `original` is the untouched
Amazon description, copied once per intent with no API call. The other five
hold the intent fixed and vary rhetorical form. Every generated arm is pinned
to the same word count so length does not confound the comparison.

Resumable: already-written (product, intent, arm) triples are skipped.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from src.config import (
    ARMS,
    DATA_DIR,
    INTENTS,
    MAIN_RUN_ARMS,
    TARGET_WORDS,
    VARIANT_MODEL,
    VARIANT_REASONING_EFFORT,
    WORD_TOLERANCE,
)
from src.jsonl import read_jsonl
from src.llm import Caller, gather_with_progress

app = typer.Typer(add_completion=False)
console = Console()

SYSTEM = (
    "You write Amazon product descriptions for non-alcoholic drinks.\n"
    "Hard rules:\n"
    "- Use ONLY facts present in the supplied listing. Never invent ingredients, "
    "certifications, health claims, awards or nutritional numbers.\n"
    "- If the listing does not support a claim, leave it out silently.\n"
    "- Never mention alcohol, star ratings, review counts, or customer reviews.\n"
    "- Write flowing prose. No headings, no markdown, no bullet characters.\n"
    '- Return JSON only: {"description": "..."}'
)

_MARKDOWN = re.compile(r"(^|\n)\s*#{1,6}\s|(^|\n)\s*[-*•]\s|\*\*|__|```")
_ALCOHOL = re.compile(
    r"alcohol|\bbeer\b|\bwine\b|\blager\b|\bstout\b|\bcider\b|\bchampagne\b|"
    r"\bprosecco\b|\bvodka\b|\bwhiskey\b|\bwhisky\b|\bbourbon\b|\brum\b|\bgin\b|"
    r"\btequila\b|\bliqueur\b|\bbrandy\b|\bcognac\b|\bmezcal\b|\bhard seltzer\b|\babv\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def word_bounds(target: int, tolerance: float = WORD_TOLERANCE) -> tuple[int, int]:
    """Inclusive word band. 70 words ±10% is 63..77."""
    slack = int(math.floor(target * tolerance + 0.5))
    return target - slack, target + slack


def product_brief(product: dict[str, Any]) -> str:
    details = product.get("details") or {}
    detail_s = "; ".join(f"{k}: {v}" for k, v in list(details.items())[:8])
    features = " | ".join(product.get("features") or [])
    price = product.get("price")
    price_s = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
    return (
        f"Title: {product.get('title')}\n"
        f"Brand: {product.get('store')}\n"
        f"Price: {price_s}\n"
        f"Category: {product.get('category')}\n"
        f"Bullet features: {features}\n"
        f"Existing description: {product.get('description_text')}\n"
        f"Details: {detail_s}"
    )


def validate_variant(
    text: str,
    source: str,
    arm: str,
    target_words: int = TARGET_WORDS,
    tolerance: float = WORD_TOLERANCE,
) -> Optional[str]:
    """Return a rejection reason, or None when the draft is acceptable.

    Length, markdown, alcohol mentions and numbers that do not appear in the
    listing are checked here. Ingredient and certification invention beyond
    bare numbers cannot be verified mechanically and stays in the prompt.
    """
    if not text or not text.strip():
        return "empty description"
    if _MARKDOWN.search(text):
        return "contains markdown or a heading"
    if _ALCOHOL.search(text):
        return "mentions alcohol"
    lo, hi = word_bounds(target_words, tolerance)
    n_words = len(text.split())
    if not lo <= n_words <= hi:
        return f"{n_words} words, outside {lo}..{hi}"
    if arm == "probe":
        if _NUMBER.search(text):
            return "probe arm contains a number"
        return None
    allowed = set(_NUMBER.findall(source))
    invented = [n for n in _NUMBER.findall(text) if n not in allowed]
    if invented:
        return f"numbers not in the listing: {', '.join(invented[:6])}"
    return None


def build_prompt(
    product: dict[str, Any],
    intent_key: str,
    arm: str,
    target_words: int,
    feedback: Optional[str] = None,
) -> str:
    intent = INTENTS[intent_key]
    lo, hi = word_bounds(target_words)
    retry = f"\nPrevious attempt rejected: {feedback}. Rewrite it." if feedback else ""
    return (
        f"{product_brief(product)}\n\n"
        f"Shopper intent: {intent['label']}. {intent['brief']}\n"
        f"Rhetorical form: {ARMS[arm]}\n"
        f"Write between {lo} and {hi} words (target {target_words}).\n"
        "Return JSON only."
        f"{retry}"
    )


def load_products(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        out[row["parent_asin"]] = row
    return out


def load_done(path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        asin = row.get("parent_asin")
        intent = row.get("intent")
        arm = row.get("arm")
        if asin and intent and arm:
            done.add((asin, intent, arm))
    return done


@app.command()
def main(
    products_path: Path = typer.Option(DATA_DIR / "study_products.jsonl", "--products"),
    out: Path = typer.Option(DATA_DIR / "variants.jsonl"),
    intents: str = typer.Option(",".join(INTENTS), help="Comma-separated intent keys"),
    arms: str = typer.Option(",".join(MAIN_RUN_ARMS), help="Comma-separated arms"),
    include_probe: bool = typer.Option(False, "--include-probe", help="Also generate the pilot probe arm"),
    model: str = typer.Option(VARIANT_MODEL),
    target_words: int = typer.Option(TARGET_WORDS),
    concurrency: int = typer.Option(16),
    temperature: float = typer.Option(0.8),
    limit: int = typer.Option(0, help="Only the first N products (smoke test)"),
    attempts: int = typer.Option(4, help="Regenerations when a draft fails the checks"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENROUTER_API_KEY"),
) -> None:
    """Generate one description per (product, intent, arm)."""
    intent_list = [s.strip() for s in intents.split(",") if s.strip()]
    arm_list = [s.strip() for s in arms.split(",") if s.strip()]
    if include_probe and "probe" not in arm_list:
        arm_list.append("probe")
    for key in intent_list:
        if key not in INTENTS:
            raise typer.BadParameter(f"Unknown intent {key}; choose from {list(INTENTS)}")
    for arm in arm_list:
        if arm not in ARMS:
            raise typer.BadParameter(f"Unknown arm {arm}; choose from {list(ARMS)}")

    products = load_products(products_path)
    asins = list(products)[: limit or None]
    done = load_done(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("a", encoding="utf-8")

    def write(row: dict[str, Any]) -> None:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()

    jobs: list[tuple[str, str, str]] = []
    for asin in asins:
        for intent_key in intent_list:
            for arm in arm_list:
                if (asin, intent_key, arm) in done:
                    continue
                if arm == "original":
                    text = products[asin]["description_text"]
                    write(
                        {
                            "parent_asin": asin,
                            "intent": intent_key,
                            "arm": "original",
                            "text": text,
                            "words": len(text.split()),
                            "model": "none",
                        }
                    )
                    continue
                jobs.append((asin, intent_key, arm))

    console.print(
        f"[cyan]{len(jobs)} variants to generate[/cyan] "
        f"({len(done)} triples already present) using {model}"
    )
    if not jobs:
        sink.close()
        console.print(f"[green]{len(load_done(out))} variants on disk[/green] → {out}")
        return

    caller = Caller(
        model=model,
        concurrency=concurrency,
        temperature=temperature,
        max_tokens=500,
        api_key=api_key,
        reasoning_effort=VARIANT_REASONING_EFFORT,
    )

    async def work(job: tuple[str, str, str]) -> Optional[dict[str, Any]]:
        asin, intent_key, arm = job
        product = products[asin]
        source = product_brief(product)
        feedback: Optional[str] = None
        last = "no attempt"
        for _ in range(attempts):
            data = await caller.json(
                SYSTEM,
                build_prompt(product, intent_key, arm, target_words, feedback),
            )
            text = str(data.get("description") or data.get("text") or "").strip()
            reason = validate_variant(text, source, arm, target_words)
            if reason is None:
                return {
                    "parent_asin": asin,
                    "intent": intent_key,
                    "arm": arm,
                    "text": text,
                    "words": len(text.split()),
                    "model": model,
                }
            last = reason
            feedback = reason
        raise ValueError(f"{asin}/{intent_key}/{arm}: {last}")

    asyncio.run(gather_with_progress(jobs, work, desc="variants", on_result=write))
    sink.close()
    total = len(load_done(out))
    console.print(f"[green]{total} variants on disk[/green] → {out}")


if __name__ == "__main__":
    app()
