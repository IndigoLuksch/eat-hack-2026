"""Demo ranking and catalog selection. No API calls and no model."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from src.demo_catalog import Catalog, image_url, load_catalog, short_features
from src.demo_model import _OpenRouterGenerator, resolve_adapter
from src.demo_rank import build_cards, place, shared_orders, summarise_places, target_option_id


def _product(asin: str, features: list[str] | None = None, description: str = "original copy") -> dict:
    return {
        "parent_asin": asin,
        "title": f"Juice {asin}",
        "store": "Brand",
        "price": 12.5,
        "average_rating": 4.6,
        "rating_number": 80,
        "features": ["Pressed apples", "No concentrate"] if features is None else features,
        "description_text": description,
    }


class AdapterPath(unittest.TestCase):
    def test_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(resolve_adapter(tmp), Path(tmp))

    def test_missing_path_is_not_downloaded(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_adapter("/no/such/adapter")


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs: dict = {}

    def create(self, **kwargs: object) -> _FakeResponse:
        self.kwargs = kwargs
        return _FakeResponse(self.content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = self


class OpenRouterWriter(unittest.TestCase):
    def test_uses_the_training_prompt_and_pins_length(self) -> None:
        client = _FakeClient("Pressed apple juice with nothing added.")
        writer = _OpenRouterGenerator("google/gemini-2.5-flash-lite", client=client)
        text = writer.generate(
            [
                {"role": "system", "content": "Write descriptions."},
                {"role": "user", "content": "Title: Apple juice"},
            ],
            max_new_tokens=180,
            temperature=0.7,
        )
        self.assertEqual(text, "Pressed apple juice with nothing added.")
        sent = client.completions.kwargs
        self.assertEqual(sent["model"], "google/gemini-2.5-flash-lite")
        self.assertEqual(sent["messages"][0]["content"], "Write descriptions.")
        self.assertIn("Title: Apple juice", sent["messages"][1]["content"])
        self.assertIn("about 70 words", sent["messages"][1]["content"])


class ImageAndFeatures(unittest.TestCase):
    def test_prefers_main_hi_res_then_large(self) -> None:
        images = [
            {"variant": "PT01", "hi_res": "https://example.com/side.jpg", "large": "https://example.com/side-l.jpg"},
            {"variant": "MAIN", "hi_res": None, "large": "https://example.com/main.jpg"},
        ]
        self.assertEqual(image_url(images), "https://example.com/main.jpg")
        images[1]["hi_res"] = "https://example.com/main-hi.jpg"
        self.assertEqual(image_url(images), "https://example.com/main-hi.jpg")

    def test_short_feature_band(self) -> None:
        self.assertTrue(short_features(["one", "two"]))
        self.assertTrue(short_features(["one"] * 6))
        self.assertFalse(short_features(["only"]))
        self.assertFalse(short_features(["one"] * 7))
        self.assertFalse(short_features(["one", "x" * 161]))


class CatalogChoice(unittest.TestCase):
    def _catalog(self, holdout: list[str] | None = None) -> Catalog:
        products = {
            "A": _product("A"),
            "B": _product("B", features=["too long " * 30, "still long"]),
            "C": _product("C", features=[]),
            "D": _product("D"),
        }
        panels = [{"panel_id": "P0", "parent_asins": ["A", "B", "C", "D"]}]
        images = {"A": "https://example.com/a.jpg", "B": "https://example.com/b.jpg", "D": "https://example.com/d.jpg"}
        return Catalog(products, panels, images, holdout or [])

    def test_skips_products_without_a_photo_or_short_features(self) -> None:
        asins = {item["parent_asin"] for item in self._catalog().eligible()}
        self.assertEqual(asins, {"A", "D"})

    def test_prefers_holdout_when_any_are_eligible(self) -> None:
        chosen = {self._catalog(["D"]).choose(random.Random(1))["parent_asin"] for _ in range(12)}
        self.assertEqual(chosen, {"D"})

    def test_empty_holdout_uses_the_full_pool(self) -> None:
        chosen = {self._catalog([]).choose(random.Random(i))["parent_asin"] for i in range(20)}
        self.assertEqual(chosen, {"A", "D"})

    def test_exclude_avoids_an_immediate_repeat(self) -> None:
        catalog = self._catalog()
        self.assertEqual(catalog.choose(random.Random(0), exclude="A")["parent_asin"], "D")

    def test_committed_catalog_has_cases(self) -> None:
        catalog = load_catalog()
        self.assertGreaterEqual(len(catalog.eligible()), 8)
        case = catalog.choose(random.Random(0))
        self.assertTrue(case["image"].startswith("https://"))
        self.assertGreaterEqual(len(case["features"]), 2)
        self.assertEqual(len(catalog.panel_asins(case["parent_asin"])), 8)


class PairedRanking(unittest.TestCase):
    def test_reps_share_an_order_and_differ_across_reps(self) -> None:
        asins = [f"A{i}" for i in range(8)]
        orders = shared_orders(asins, 3, random.Random(4))
        self.assertEqual(len(orders), 3)
        self.assertEqual(sorted(orders[0]), asins)
        self.assertNotEqual(orders[0], orders[1])

    def test_only_the_target_text_changes(self) -> None:
        asins = ["T", "C1", "C2"]
        products = {asin: _product(asin, description=f"orig {asin}") for asin in asins}
        order = ["C1", "T", "C2"]
        user = build_cards(order, products, "T", "user copy")
        model = build_cards(order, products, "T", "model copy")
        self.assertEqual([card["option_id"] for card in user], ["O1", "O2", "O3"])
        self.assertEqual(user[0]["text"], "orig C1")
        self.assertEqual(user[1]["text"], "user copy")
        self.assertEqual(model[1]["text"], "model copy")
        self.assertEqual(user[2]["text"], model[2]["text"])
        self.assertEqual(target_option_id(user, "T"), "O2")

    def test_place_is_one_at_the_top(self) -> None:
        self.assertEqual(place(["O3", "O1", "O2"], "O1"), 2)

    def test_lower_mean_wins_and_equals_tie(self) -> None:
        self.assertEqual(summarise_places([1, 2, 3], [4, 4, 4])["winner"], "user")
        self.assertEqual(summarise_places([2, 2, 2], [1, 2, 2])["winner"], "model")
        tied = summarise_places([1, 4], [2, 3])
        self.assertEqual(tied["winner"], "tie")
        self.assertEqual(tied["user"]["mean"], 2.5)


class RankStream(unittest.TestCase):
    def test_stream_reports_places_for_both_descriptions(self) -> None:
        try:
            from fastapi.testclient import TestClient
            import src.demo_server as server
        except ImportError:
            self.skipTest("demo dependencies are not installed")

        class FakeModel:
            error = None

            def wait_ready(self, timeout: float) -> bool:
                return True

            def generate(self, messages: list, max_new_tokens: int = 180, temperature: float = 0.7) -> str:
                return "Model copy about this juice."

        async def fake_rank(caller: object, intent_key: str, cards: list) -> list:
            def sort_key(card: dict) -> int:
                if str(card["text"]).startswith("USER"):
                    return 0
                if str(card["text"]).startswith("Model"):
                    return 2
                return 1

            return [card["option_id"] for card in sorted(cards, key=sort_key)]

        server._start_model = lambda: (
            setattr(server, "catalog", server.load_catalog()),
            setattr(server, "model", FakeModel()),
        )
        server.rank_cards = fake_rank

        with TestClient(server.app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            first = client.get("/api/case").json()
            second = client.get("/api/case", params={"exclude": first["parent_asin"]}).json()
            self.assertNotEqual(first["parent_asin"], second["parent_asin"])
            self.assertNotIn("description_text", first)
            self.assertIn("features", first)
            response = client.post(
                "/api/rank",
                json={"parent_asin": first["parent_asin"], "intent": "health", "text": "USER copy of the drink"},
            )
            self.assertEqual(response.status_code, 200)
            body = response.text
            self.assertIn('"phase": "writing"', body)
            self.assertIn('"phase": "ranking"', body)
            self.assertIn('"phase": "done"', body)
            done = json.loads(body.split("data: ")[-1])
            self.assertEqual(done["result"]["winner"], "user")
            self.assertEqual(done["result"]["user"]["places"], [1, 1, 1])
            self.assertEqual(done["result"]["model"]["text"], "Model copy about this juice.")
            self.assertTrue(all(place == 8 for place in done["result"]["model"]["places"]))
