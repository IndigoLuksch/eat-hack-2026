"""Filter non-alcoholic drink products from Amazon Reviews 2023 grocery metadata."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import typer
from rich.console import Console
from tqdm import tqdm

from src.config import (
    ALCOHOL_EXCLUDE,
    ALCOHOL_L2,
    BEVERAGE_L2,
    DATA_DIR,
    DRINK_FORM_RULES,
    DRINK_TITLE_HINTS,
    EXTRA_DRINK_PATHS,
    HF_CACHE_DIR,
    HF_DATASET,
    META_JSONL,
    NON_ALCOHOLIC_MARKERS,
    NON_DRINK_EXCLUDE,
    NON_DRINK_FOOD_WORDS,
)

app = typer.Typer(add_completion=False)
console = Console()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


@lru_cache(maxsize=None)
def _phrase_re(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Word-boundary alternation, so 'cola' never matches 'chocolate'."""
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(p) for p in phrases))


def _has(text: str, phrases: tuple[str, ...]) -> bool:
    return _phrase_re(phrases).search(text) is not None


def _category_path(categories: Any) -> str:
    if not isinstance(categories, list):
        return ""
    return _norm(" > ".join(str(c) for c in categories if c))


def _join_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x)
    return str(value)


def drink_form(categories: Any, title: str) -> str:
    """Match the deepest category first; parent names like 'Bottled Beverages,
    Water & Drink Mixes' would otherwise shadow the specific leaf."""
    segments = [_norm(str(c)) for c in categories if c] if isinstance(categories, list) else []
    for segment in reversed(segments):
        for form, keys in DRINK_FORM_RULES:
            if any(k in segment for k in keys):
                return form
    if _has(title, DRINK_TITLE_HINTS):
        return "ready_to_drink"
    return "other"


def classify(item: dict[str, Any]) -> Optional[str]:
    """Return the drink form, or None if the item is not a non-alcoholic drink."""
    title = _norm(item.get("title") or "")
    if not title:
        return None

    categories = item.get("categories") or []
    path = _category_path(categories)
    level2 = _norm(str(categories[1])) if len(categories) > 1 else ""

    if _has(title, NON_DRINK_EXCLUDE):
        return None

    # De-alcoholised drinks are in scope even though they name an alcohol.
    declared_non_alcoholic = _has(f"{title} {path}", NON_ALCOHOLIC_MARKERS)
    if not declared_non_alcoholic:
        if level2 in ALCOHOL_L2 or _has(f"{title} {path}", ALCOHOL_EXCLUDE):
            return None

    is_drink = (
        level2 == BEVERAGE_L2
        or any(p in path for p in EXTRA_DRINK_PATHS)
        # ~68k items carry no categories; fall back to the title.
        or (
            not categories
            and _has(title, DRINK_TITLE_HINTS)
            and not _has(title, NON_DRINK_FOOD_WORDS)
        )
        # De-alcoholised beer/wine: a drink despite sitting under alcohol.
        or (declared_non_alcoholic and (level2 in ALCOHOL_L2 or _has(title, DRINK_TITLE_HINTS)))
    )
    if not is_drink:
        return None

    form = drink_form(categories, title)
    if declared_non_alcoholic and form == "other":
        form = "ready_to_drink"
    return form


def simplify(item: dict[str, Any], form: str) -> dict[str, Any]:
    price = item.get("price")
    if price in (None, "None", ""):
        price = None
    else:
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = None

    description = item.get("description") or []
    return {
        "parent_asin": item.get("parent_asin"),
        "title": item.get("title") or "",
        "store": item.get("store"),
        "average_rating": item.get("average_rating"),
        "rating_number": item.get("rating_number"),
        "price": price,
        "drink_form": form,
        "categories": item.get("categories") or [],
        "features": item.get("features") or [],
        "description": description,
        "description_text": _join_text(description).strip(),
        "details": item.get("details"),
    }


def resolve_meta_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    matches = sorted(HF_CACHE_DIR.rglob(Path(META_JSONL).name))
    if matches:
        return matches[0]
    from huggingface_hub import hf_hub_download

    console.print("[yellow]Metadata not cached; downloading (~1.4GB)[/yellow]")
    return Path(
        hf_hub_download(
            repo_id=HF_DATASET,
            filename=META_JSONL,
            repo_type="dataset",
            cache_dir=str(HF_CACHE_DIR),
        )
    )


def iter_grocery_meta(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@app.command()
def main(
    out: Path = typer.Option(DATA_DIR / "drinks.jsonl", help="Output JSONL path"),
    limit: int = typer.Option(0, help="Max drink products to keep (0 = all)"),
    min_rating_count: int = typer.Option(20, help="Minimum rating_number"),
    min_description_chars: int = typer.Option(40, help="Minimum description length"),
    forms: str = typer.Option(
        "",
        help="Comma-separated drink_form filter, e.g. 'ready_to_drink,mix_or_concentrate'",
    ),
    local_meta: Optional[Path] = typer.Option(
        None, help="Path to meta_Grocery_and_Gourmet_Food.jsonl"
    ),
) -> None:
    """Extract non-alcoholic drink products into a local catalogue."""
    meta_path = resolve_meta_path(local_meta)
    keep_forms = {f.strip() for f in forms.split(",") if f.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    drinks = 0
    kept = 0
    by_form: dict[str, int] = {}

    with out.open("w", encoding="utf-8") as f:
        for item in tqdm(iter_grocery_meta(meta_path), desc="scanning grocery meta"):
            scanned += 1
            form = classify(item)
            if form is None:
                continue
            drinks += 1
            if keep_forms and form not in keep_forms:
                continue
            row = simplify(item, form)
            if len(row["description_text"]) < min_description_chars:
                continue
            if (row.get("rating_number") or 0) < min_rating_count:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            by_form[form] = by_form.get(form, 0) + 1
            kept += 1
            if limit and kept >= limit:
                break

    console.print(
        f"[green]Wrote {kept} drinks[/green] "
        f"(scanned {scanned}, {drinks} classified as drinks) → {out}"
    )
    for form, count in sorted(by_form.items(), key=lambda kv: -kv[1]):
        console.print(f"  {form}: {count}")


if __name__ == "__main__":
    app()
