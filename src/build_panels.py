"""Select study products and group them into blocked shortlist panels.

Panels are the unit of the experiment: a fixed set of PANEL_SIZE comparable
products shown together many times, with only the description variants
re-randomised between repeats. Products inside a panel share a leaf category
and sit next to each other in price, so product identity varies as little as
possible and the copy has room to move the ranking.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from src.config import (
    DATA_DIR,
    MAX_PRICE_RATIO,
    MIN_DESC_CHARS,
    MIN_RATINGS,
    N_PANELS,
    PANEL_SIZE,
    PRICE_MAX,
    PRICE_MIN,
    STUDY_CATEGORY,
    TITLE_EXCLUDE,
)

app = typer.Typer(add_completion=False)
console = Console()


def eligible(row: dict[str, Any], price_min: float, price_max: float) -> bool:
    price = row.get("price")
    if price is None or not (price_min <= price <= price_max):
        return False
    title = (row.get("title") or "").lower()
    if any(x in title for x in TITLE_EXCLUDE):
        return False
    return (
        row.get("drink_form") == "ready_to_drink"
        and (row.get("rating_number") or 0) >= MIN_RATINGS
        and len(row.get("description_text") or "") >= MIN_DESC_CHARS
        and row.get("average_rating") is not None
    )


_SIZE_TOKEN = re.compile(
    r"^(?:\d+[a-z]*|pack|pk|of|count|ct|oz|fl|ml|l|g|kg|lb|bottles?|cans?|boxes?)$"
)


def dedupe_key(row: dict[str, Any]) -> str:
    """Amazon carries the same drink at many pack sizes; two of them inside one
    panel would be a duplicate rather than a comparison. Strip size and pack
    tokens so `Vimto Cordial 725ml` and `Vimto Cordial 725ml (Pack of 2)`
    collapse to the same key."""
    title = re.sub(r"[^a-z0-9 ]", " ", (row.get("title") or "").lower())
    words = [w for w in title.split() if not _SIZE_TOKEN.match(w)]
    return f"{(row.get('store') or '').lower()}|{' '.join(words[:5])}"


def leaf_category(row: dict[str, Any]) -> str:
    cats = row.get("categories") or []
    return str(cats[-1]) if cats else "Uncategorised"


def slim(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": row["parent_asin"],
        "title": row["title"],
        "store": row.get("store"),
        "price": row["price"],
        "average_rating": row["average_rating"],
        "rating_number": row["rating_number"],
        "category": leaf_category(row),
        "features": (row.get("features") or [])[:8],
        "description_text": row["description_text"],
        "details": row.get("details") or {},
    }


@app.command()
def main(
    drinks: Path = typer.Option(DATA_DIR / "drinks.jsonl", help="Input drink catalogue"),
    out_products: Path = typer.Option(DATA_DIR / "study_products.jsonl"),
    out_panels: Path = typer.Option(DATA_DIR / "panels.json"),
    n_panels: int = typer.Option(N_PANELS),
    panel_size: int = typer.Option(PANEL_SIZE),
    category: str = typer.Option(STUDY_CATEGORY, help="Leaf category; empty string = all"),
    price_min: float = typer.Option(PRICE_MIN),
    price_max: float = typer.Option(PRICE_MAX),
    max_price_ratio: float = typer.Option(MAX_PRICE_RATIO),
    max_per_category: int = typer.Option(99, help="Cap panels drawn from one category"),
    seed: int = typer.Option(11),
) -> None:
    """Build blocked panels of comparable products."""
    rng = random.Random(seed)

    seen: set[str] = set()
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    with drinks.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            scanned += 1
            if not eligible(row, price_min, price_max):
                continue
            leaf = leaf_category(row)
            if category and leaf != category:
                continue
            key = dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            by_category[leaf].append(row)

    # Price-sorted consecutive chunks keep each panel within a narrow price band.
    # Thin categories can still produce a chunk spanning an order of magnitude,
    # which would let price rather than copy drive the ranking, so cap the ratio.
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    rejected = 0
    for category, rows in by_category.items():
        if len(rows) < panel_size:
            continue
        rows.sort(key=lambda r: r["price"])
        for i in range(0, len(rows) - panel_size + 1, panel_size):
            chunk = rows[i : i + panel_size]
            lo, hi = chunk[0]["price"], chunk[-1]["price"]
            if lo <= 0 or hi / lo > max_price_ratio:
                rejected += 1
                continue
            candidates.append((category, chunk))

    # Round-robin across categories so no single one dominates the study.
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for category, chunk in candidates:
        grouped[category].append(chunk)
    for chunks in grouped.values():
        rng.shuffle(chunks)

    chosen: list[tuple[str, list[dict[str, Any]]]] = []
    order = sorted(grouped, key=lambda c: -len(grouped[c]))
    taken: dict[str, int] = defaultdict(int)
    while len(chosen) < n_panels:
        progressed = False
        for category in order:
            if len(chosen) >= n_panels:
                break
            if taken[category] >= max_per_category or not grouped[category]:
                continue
            chosen.append((category, grouped[category].pop()))
            taken[category] += 1
            progressed = True
        if not progressed:
            break

    if len(chosen) < n_panels:
        console.print(
            f"[yellow]Only built {len(chosen)} of {n_panels} panels[/yellow] "
            f"— raise --max-per-category or lower --panel-size"
        )

    out_products.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    products: dict[str, dict[str, Any]] = {}
    for idx, (category, chunk) in enumerate(chosen):
        prices = [r["price"] for r in chunk]
        panels.append(
            {
                "panel_id": f"P{idx:03d}",
                "category": category,
                "price_min": round(min(prices), 2),
                "price_max": round(max(prices), 2),
                "parent_asins": [r["parent_asin"] for r in chunk],
            }
        )
        for r in chunk:
            products[r["parent_asin"]] = slim(r)

    with out_products.open("w", encoding="utf-8") as f:
        for row in products.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_panels.write_text(json.dumps(panels, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(
        f"[green]{len(panels)} panels, {len(products)} products[/green] "
        f"(scanned {scanned}, {len(seen)} eligible after dedupe)"
    )
    spread = defaultdict(int)
    for p in panels:
        spread[p["category"]] += 1
    for category, count in sorted(spread.items(), key=lambda kv: -kv[1]):
        console.print(f"  {category}: {count} panels")
    console.print(f"→ {out_products}\n→ {out_panels}")


if __name__ == "__main__":
    app()
