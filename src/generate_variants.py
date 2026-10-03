"""Generate controlled description variants for each study product.

One variant per style per product. Every engineered variant is pinned to the
same word count so that length does not confound the style comparison; the
`original` arm is left untouched because it is the real-world baseline we want
to beat, not a style.

Resumable: already-generated (product, style) pairs are skipped, so an
interrupted run can simply be restarted.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from src.config import (
    DATA_DIR,
    MAIN_RUN_STYLES,
    TARGET_WORDS,
    VARIANT_MODEL,
    VARIANT_STYLES,
)
from src.llm import Caller, gather_with_progress

app = typer.Typer(add_completion=False)
console = Console()

SYSTEM = (
    "You rewrite Amazon product descriptions for non-alcoholic drinks.\n"
    "Hard rules:\n"
    "- Use ONLY facts present in the supplied listing. Never invent ingredients, "
    "certifications, health claims, awards or nutritional numbers.\n"
    "- If the listing does not support a claim, leave it out silently.\n"
    "- Never mention alcohol.\n"
    "- Write flowing marketing copy. No headings, no markdown, no bullet characters.\n"
    "- Return JSON only: {\"description\": \"...\"}"
)


def product_brief(product: dict[str, Any]) -> str:
    details = product.get("details") or {}
    detail_s = "; ".join(f"{k}: {v}" for k, v in list(details.items())[:8])
    features = " | ".join(product.get("features") or [])
    return (
        f"Title: {product['title']}\n"
        f"Brand: {product.get('store')}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Rating: {product['average_rating']} from {product['rating_number']} reviews\n"
        f"Category: {product.get('category')}\n"
        f"Bullet features: {features}\n"
        f"Existing description: {product['description_text']}\n"
        f"Details: {detail_s}"
    )


def build_prompt(product: dict[str, Any], style: str, target_words: int) -> str:
    return (
        f"{product_brief(product)}\n\n"
        f"Style instruction: {VARIANT_STYLES[style]}\n"
        f"Write exactly about {target_words} words (within 10%).\n"
        "Return JSON only."
    )


def load_products(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["parent_asin"]] = row
    return out


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row["parent_asin"], row["style"]))
    return done


@app.command()
def main(
    products_path: Path = typer.Option(DATA_DIR / "study_products.jsonl", "--products"),
    out: Path = typer.Option(DATA_DIR / "variants.jsonl"),
    styles: str = typer.Option(",".join(MAIN_RUN_STYLES), help="Comma-separated styles"),
    model: str = typer.Option(VARIANT_MODEL),
    target_words: int = typer.Option(TARGET_WORDS),
    concurrency: int = typer.Option(16),
    temperature: float = typer.Option(0.8),
    limit: int = typer.Option(0, help="Only the first N products (smoke test)"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENROUTER_API_KEY"),
) -> None:
    """Generate one description per (product, style)."""
    style_list = [s.strip() for s in styles.split(",") if s.strip()]
    for s in style_list:
        if s not in VARIANT_STYLES:
            raise typer.BadParameter(f"Unknown style {s}; choose from {list(VARIANT_STYLES)}")

    products = load_products(products_path)
    asins = list(products)[: limit or None]
    done = load_done(out)

    out.parent.mkdir(parents=True, exist_ok=True)
    sink = out.open("a", encoding="utf-8")

    def write(row: dict[str, Any]) -> None:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()

    # The untouched listing is the control arm; it costs no tokens.
    jobs: list[tuple[str, str]] = []
    for asin in asins:
        for style in style_list:
            if (asin, style) in done:
                continue
            if style == "original":
                text = products[asin]["description_text"]
                write(
                    {
                        "parent_asin": asin,
                        "style": "original",
                        "text": text,
                        "words": len(text.split()),
                        "model": "none",
                    }
                )
                continue
            jobs.append((asin, style))

    console.print(
        f"[cyan]{len(jobs)} variants to generate[/cyan] "
        f"({len(done)} already present) using {model}"
    )
    if not jobs:
        sink.close()
        return

    caller = Caller(model=model, concurrency=concurrency, temperature=temperature, max_tokens=500)

    async def work(job: tuple[str, str]) -> Optional[dict[str, Any]]:
        asin, style = job
        data = await caller.json(SYSTEM, build_prompt(products[asin], style, target_words))
        text = str(data.get("description", "")).strip()
        if len(text.split()) < 20:
            raise ValueError(f"suspiciously short variant for {asin}/{style}")
        return {
            "parent_asin": asin,
            "style": style,
            "text": text,
            "words": len(text.split()),
            "model": model,
        }

    asyncio.run(gather_with_progress(jobs, work, desc="variants", on_result=write))
    sink.close()

    total = len(load_done(out))
    console.print(f"[green]{total} variants on disk[/green] → {out}")


if __name__ == "__main__":
    app()
