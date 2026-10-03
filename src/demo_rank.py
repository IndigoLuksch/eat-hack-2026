"""Compare a visitor's description with the model's on one product's panel.

Three repetitions. Within a repetition both rankings share a display order and
the same seven competitor descriptions. Only the target product's text changes.
"""

from __future__ import annotations

import random
from typing import Any

N_REPS = 3


def shared_orders(asins: list[str], reps: int, rng: random.Random) -> list[list[str]]:
    """One shuffled display order per repetition, shared by both descriptions."""
    orders = []
    for _ in range(reps):
        order = list(asins)
        rng.shuffle(order)
        orders.append(order)
    return orders


def build_cards(
    order: list[str],
    products: dict[str, dict[str, Any]],
    target_asin: str,
    description: str,
) -> list[dict[str, Any]]:
    """Option cards in the training shape. Competitors keep their original copy."""
    cards = []
    for position, asin in enumerate(order):
        product = products[asin]
        text = description if asin == target_asin else (product.get("description_text") or "")
        price = product.get("price") or 0
        cards.append(
            {
                "option_id": f"O{position + 1}",
                "parent_asin": asin,
                "position": position,
                "title": product.get("title") or "",
                "brand": product.get("store") or "Unknown",
                "price": float(price),
                "rating": product.get("average_rating"),
                "reviews": product.get("rating_number"),
                "text": text,
            }
        )
    return cards


def target_option_id(cards: list[dict[str, Any]], target_asin: str) -> str:
    for card in cards:
        if card["parent_asin"] == target_asin:
            return str(card["option_id"])
    raise KeyError(target_asin)


def place(ranking: list[str], option_id: str) -> int:
    """1 is the top of the list."""
    return ranking.index(option_id) + 1


def summarise_places(user_places: list[int], model_places: list[int]) -> dict[str, Any]:
    """Lower average place wins. Equal averages are a tie."""
    if len(user_places) != len(model_places) or not user_places:
        raise ValueError("both sides need the same non-empty list of places")
    user_mean = sum(user_places) / len(user_places)
    model_mean = sum(model_places) / len(model_places)
    if user_mean == model_mean:
        winner = "tie"
    elif user_mean < model_mean:
        winner = "user"
    else:
        winner = "model"
    return {
        "user": {"places": list(user_places), "mean": user_mean},
        "model": {"places": list(model_places), "mean": model_mean},
        "winner": winner,
    }
