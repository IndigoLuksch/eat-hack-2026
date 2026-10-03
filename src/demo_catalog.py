"""Products the demo can show.

A case is one study product that sits in a ranking panel, has a photo, and has
a short feature list. Image URLs are committed so the service does not need
the grocery metadata file.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Optional

from src.config import DATA_DIR, INTENTS
from src.jsonl import read_jsonl

FEATURE_MIN = 2
FEATURE_MAX = 6
FEATURE_MAX_CHARS = 160

_ASIN = re.compile(r'"parent_asin"\s*:\s*"([^"]+)"')


def clean_features(features: Any) -> list[str]:
    if not isinstance(features, list):
        return []
    return [str(item).strip() for item in features if str(item).strip()]


def short_features(features: Any) -> bool:
    """A handful of short bullets. Long marketing paragraphs stay off the page."""
    cleaned = clean_features(features)
    if not FEATURE_MIN <= len(cleaned) <= FEATURE_MAX:
        return False
    return all(len(item) <= FEATURE_MAX_CHARS for item in cleaned)


def image_url(images: Any) -> Optional[str]:
    """Prefer the main photo's hi-res URL, then its large URL, then any other."""
    if not isinstance(images, list):
        return None
    dicts = [img for img in images if isinstance(img, dict)]
    ordered = [img for img in dicts if img.get("variant") == "MAIN"]
    ordered += [img for img in dicts if img not in ordered]
    for img in ordered:
        for key in ("hi_res", "large"):
            url = img.get(key)
            if isinstance(url, str) and url.startswith("http"):
                return url
    return None


def index_images(meta_path: Path, asins: set[str]) -> dict[str, str]:
    """Scan grocery metadata once and keep a URL for each requested ASIN."""
    found: dict[str, str] = {}
    wanted = set(asins)
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(wanted):
                break
            match = _ASIN.search(line)
            if match is None or match.group(1) not in wanted or match.group(1) in found:
                continue
            url = image_url(json.loads(line).get("images"))
            if url:
                found[match.group(1)] = url
    return found


def load_images(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(asin): url for asin, url in raw.items() if isinstance(url, str) and url}


def load_holdout(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [str(asin) for asin in raw]


class Catalog:
    def __init__(
        self,
        products: dict[str, dict[str, Any]],
        panels: list[dict[str, Any]],
        images: dict[str, str],
        holdout: list[str],
    ) -> None:
        self.products = products
        self.panels = panels
        self.images = images
        self.holdout = holdout
        self.panel_of = {
            asin: panel for panel in panels for asin in panel.get("parent_asins") or []
        }

    def eligible(self) -> list[dict[str, Any]]:
        """Panel products with a photo and a short feature list.

        When the holdout file lists products, those are preferred. An empty
        holdout, or a holdout with nothing eligible, falls back to the full pool.
        """
        pool = []
        for asin, product in self.products.items():
            if asin not in self.panel_of or asin not in self.images:
                continue
            if not short_features(product.get("features")):
                continue
            pool.append(product)
        if not self.holdout:
            return pool
        preferred = [product for product in pool if product["parent_asin"] in set(self.holdout)]
        return preferred or pool

    def choose(
        self,
        rng: random.Random,
        exclude: Optional[str] = None,
    ) -> dict[str, Any]:
        pool = self.eligible()
        if exclude:
            fresh = [product for product in pool if product["parent_asin"] != exclude]
            if fresh:
                pool = fresh
        if not pool:
            raise LookupError("No demo products are available.")
        product = rng.choice(pool)
        return {
            "parent_asin": product["parent_asin"],
            "title": product.get("title") or "",
            "image": self.images[product["parent_asin"]],
            "features": clean_features(product.get("features")),
            "intent": rng.choice(list(INTENTS)),
        }

    def panel_asins(self, asin: str) -> list[str]:
        panel = self.panel_of.get(asin)
        if panel is None:
            raise KeyError(asin)
        return list(panel["parent_asins"])


def load_catalog(
    products_path: Path = DATA_DIR / "study_products.jsonl",
    panels_path: Path = DATA_DIR / "panels.json",
    images_path: Path = DATA_DIR / "product_images.json",
    holdout_path: Path = DATA_DIR / "holdout_asins.json",
) -> Catalog:
    products = {row["parent_asin"]: row for row in read_jsonl(products_path)}
    panels = json.loads(panels_path.read_text(encoding="utf-8"))
    return Catalog(products, panels, load_images(images_path), load_holdout(holdout_path))


def write_image_index(
    out: Path = DATA_DIR / "product_images.json",
    products_path: Path = DATA_DIR / "study_products.jsonl",
    meta_path: Optional[Path] = None,
) -> dict[str, str]:
    """Build the committed image map from the cached Amazon metadata."""
    from src.extract_drinks import resolve_meta_path

    asins = {row["parent_asin"] for row in read_jsonl(products_path)}
    found = index_images(resolve_meta_path(meta_path), asins)
    out.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return found


if __name__ == "__main__":
    images = write_image_index()
    print(f"{len(images)} image urls → {DATA_DIR / 'product_images.json'}")
