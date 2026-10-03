"""Unit tests for the measurement and training-data pipeline.

These cover the deterministic pieces: variant checks, trial construction,
scoring, the SFT/DPO gates, and the fine-tune backend guard. They do not
call OpenRouter or load a model.
"""

from __future__ import annotations

import platform
import sys
import unittest

from src.build_finetune_data import (
    build_records,
    holdout_count,
    prompt_messages,
    select_holdout,
    split_by_asin,
)
from src.config import INTENTS, MAIN_RUN_ARMS, SFT_SYSTEM
from src.evaluate import assign_sources, holdout_panels, summarise, trial_winner, wilson
from src.finetune import reject_unsloth_on_apple_silicon
from src.generate_variants import validate_variant, word_bounds
from src.run_ranking import draw_assignment, parse_ranking, trial_grid, trial_id
from src.score_variants import normalised_rank, score_rankings, shrink, winning_generated_arm

MAIN = ["direct", "sensory", "quantified", "use_case", "assurance", "original"]


def _cell(scores: list[float]) -> list[dict]:
    return [
        {"arm": arm, "score": score, "n": 10, "cell_mean": score, "pooled_mean": score}
        for arm, score in zip(MAIN, scores)
    ]


def _product(asin: str) -> dict:
    return {
        "parent_asin": asin,
        "title": f"Juice {asin}",
        "store": "Brand",
        "price": 12.5,
        "average_rating": 4.8,
        "rating_number": 100,
        "features": ["Vitamin C"],
        "description_text": "SECRET_ORIGINAL",
    }


class VariantChecks(unittest.TestCase):
    def test_word_band_is_ten_percent(self) -> None:
        self.assertEqual(word_bounds(70), (63, 77))

    def test_accepts_grounded_copy_inside_the_band(self) -> None:
        text = " ".join(["bright"] * 69 + ["4.8"])
        source = "Rating: 4.8 from 100 reviews"
        self.assertIsNone(validate_variant(text, source, "sensory", 70))

    def test_rejects_length_markdown_alcohol_and_invented_numbers(self) -> None:
        source = "Rating: 4.8"
        self.assertIsNotNone(validate_variant(" ".join(["bright"] * 62), source, "direct"))
        self.assertIsNotNone(validate_variant(" ".join(["bright"] * 78), source, "direct"))
        padded = " ".join(["# Heading"] + ["bright"] * 69)
        self.assertIn("markdown", validate_variant(padded, source, "direct") or "")
        alcohol = " ".join(["wine"] + ["bright"] * 69)
        self.assertIn("alcohol", validate_variant(alcohol, source, "direct") or "")
        invented = " ".join(["contains", "999"] + ["bright"] * 68)
        self.assertIn("999", validate_variant(invented, source, "quantified") or "")

    def test_probe_rejects_digits(self) -> None:
        text = " ".join(["pleasant"] * 69 + ["12"])
        self.assertIsNotNone(validate_variant(text, "12 pack", "probe"))


class TrialConstruction(unittest.TestCase):
    def test_trial_id_matches_the_plan(self) -> None:
        self.assertEqual(trial_id("health", "P003", 42), "health-P003-00042")

    def test_grid_balances_intents_and_is_unique(self) -> None:
        intents = ["general", "health", "flavour"]
        panels = [f"P{i:03d}" for i in range(25)]
        grid = trial_grid(24000, intents, panels)
        self.assertEqual(grid[0], ("general", "P000", 0))
        self.assertEqual(grid[1], ("health", "P000", 0))
        self.assertEqual(grid[3], ("general", "P001", 0))
        self.assertEqual(grid[75], ("general", "P000", 1))
        counts = {intent: 0 for intent in intents}
        ids = []
        for intent, panel_id, rep in grid:
            counts[intent] += 1
            ids.append(trial_id(intent, panel_id, rep))
        self.assertEqual(counts, {intent: 8000 for intent in intents})
        self.assertEqual(len(ids), len(set(ids)))

    def test_assignment_is_deterministic_and_covers_the_panel(self) -> None:
        asins = [f"B{i}" for i in range(8)]
        arms = list(MAIN_RUN_ARMS)
        first, order = draw_assignment("health-P003-00042", asins, arms)
        again, order_again = draw_assignment("health-P003-00042", asins, arms)
        self.assertEqual(first, again)
        self.assertEqual(order, order_again)
        self.assertEqual(sorted(order), asins)
        self.assertTrue(set(first.values()) <= set(arms))

    def test_ranking_must_be_a_permutation(self) -> None:
        ids = [f"O{i}" for i in range(1, 9)]
        self.assertEqual(parse_ranking({"ranking": list(reversed(ids))}, ids), list(reversed(ids)))
        self.assertIsNone(parse_ranking({"ranking": ids[:-1]}, ids))
        self.assertIsNone(parse_ranking({"ranking": ids + ["O1"]}, ids))
        self.assertIsNone(parse_ranking({"ranking": "O1"}, ids))
        mixed = ["option o1", "2", "O3", "O4", "O5", "O6", "O7", "O8"]
        self.assertEqual(parse_ranking({"ranking": mixed}, ids), ids)


class Scoring(unittest.TestCase):
    def test_normalised_rank_and_shrinkage(self) -> None:
        self.assertEqual(normalised_rank(0, 8), 0.0)
        self.assertEqual(normalised_rank(7, 8), 1.0)
        self.assertAlmostEqual(shrink(0.2, 25, 0.8, prior=25), 0.5)

    def test_leaderboard_orders_probe_last_when_it_loses(self) -> None:
        options = [
            {"option_id": "O1", "parent_asin": "A", "arm": "direct"},
            {"option_id": "O2", "parent_asin": "B", "arm": "original"},
            {"option_id": "O3", "parent_asin": "C", "arm": "probe"},
        ]
        trials = [
            {"intent": "health", "ranking": ["O1", "O2", "O3"], "options": options}
            for _ in range(5)
        ]
        scores, board, used = score_rankings(trials, prior=25)
        self.assertEqual(used, 5)
        self.assertEqual([row["arm"] for row in board["health"]], ["direct", "original", "probe"])
        self.assertEqual(board["health"][1]["position"], 2)
        self.assertEqual(board["health"][2]["position"], 3)
        self.assertAlmostEqual(scores["A"]["health"][0]["score"], 0.0)
        self.assertEqual(winning_generated_arm(scores, "health"), "direct")


class FinetuneData(unittest.TestCase):
    def _bundle(self, min_gap: float = 0.05):
        scores = {
            "P1": {"health": _cell([0.10, 0.20, 0.30, 0.40, 0.50, 0.60])},
            "P2": {"health": _cell([0.40, 0.45, 0.50, 0.55, 0.60, 0.10])},
        }
        products = {asin: _product(asin) for asin in scores}
        variants = {
            (asin, "health", arm): f"{arm} copy {asin}"
            for asin in scores
            for arm in MAIN
        }
        return build_records(
            scores,
            products,
            variants,
            holdout_n=0,
            min_gap=min_gap,
        )

    def test_rank2_gate_and_gap(self) -> None:
        sft, dpo, holdout, stats = self._bundle(0.05)
        self.assertEqual(holdout, [])
        self.assertEqual(stats["sft_examples"], 3)
        self.assertEqual(stats["sft_rank2_dropped"], 1)
        self.assertEqual(stats["dpo_pairs"], 4)
        p2 = [row for row in sft if row["parent_asin"] == "P2"]
        self.assertEqual([row["arm"] for row in p2], ["original"])
        user = p2[0]["messages"][1]["content"]
        self.assertNotIn("SECRET_ORIGINAL", user)
        self.assertIn("Vitamin C", user)
        self.assertEqual(p2[0]["messages"][0]["content"], SFT_SYSTEM)
        self.assertEqual(p2[0]["messages"][2]["content"], "original copy P2")

        _sft, dpo_tight, _holdout, stats_tight = self._bundle(0.20)
        self.assertEqual(stats_tight["dpo_pairs"], 3)
        self.assertTrue(all(row["gap"] > 0.20 for row in dpo_tight))

    def test_holdout_is_by_product_and_stable(self) -> None:
        self.assertEqual(holdout_count(200, 20), 20)
        self.assertEqual(holdout_count(8, 20), 0)
        asins = [f"B{i:03d}" for i in range(200)]
        first = select_holdout(asins, 20, seed=11)
        second = select_holdout(asins, 20, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len(set(first)), 20)

    def test_train_eval_split_does_not_share_a_product(self) -> None:
        rows = []
        for asin in ("A", "B", "C", "D"):
            for intent in ("health", "flavour"):
                rows.append({"parent_asin": asin, "intent": intent})
        train, held = split_by_asin(rows, 0.25, seed=11)
        self.assertFalse({row["parent_asin"] for row in train} & {row["parent_asin"] for row in held})
        self.assertEqual(len(train) + len(held), len(rows))

    def test_prompt_omits_the_original_description(self) -> None:
        messages = prompt_messages(_product("P1"), "flavour")
        self.assertEqual(len(messages), 2)
        self.assertNotIn("SECRET_ORIGINAL", messages[1]["content"])
        self.assertNotIn("4.8", messages[1]["content"])
        self.assertNotIn("100", messages[1]["content"])
        self.assertIn("Flavour", messages[1]["content"])


class Evaluation(unittest.TestCase):
    def test_sources_rotate_evenly(self) -> None:
        asins = [f"A{i}" for i in range(8)]
        seen = {asin: set() for asin in asins}
        for rep in range(3):
            assigned = assign_sources(asins, rep)
            counts: dict[str, int] = {}
            for asin, source in assigned.items():
                seen[asin].add(source)
                counts[source] = counts.get(source, 0) + 1
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for asin in asins:
            self.assertEqual(seen[asin], {"finetuned", "prompt_only", "original"})

    def test_winner_and_wilson_interval(self) -> None:
        asins = [f"A{i}" for i in range(8)]
        assigned = assign_sources(asins, 0)
        options = []
        groups = {"finetuned": [], "prompt_only": [], "original": []}
        for index, asin in enumerate(asins):
            option_id = f"O{index + 1}"
            source = assigned[asin]
            groups[source].append(option_id)
            options.append(
                {"option_id": option_id, "parent_asin": asin, "source": source}
            )
        ranking = groups["finetuned"] + groups["prompt_only"] + groups["original"]
        self.assertEqual(trial_winner(options, ranking), "finetuned")
        rate, low, high = wilson(80, 100)
        self.assertAlmostEqual(rate, 0.8)
        self.assertLess(low, rate)
        self.assertGreater(high, rate)
        report = summarise(
            [{"intent": "health", "options": options, "ranking": ranking} for _ in range(4)]
        )
        self.assertEqual(report["overall"]["wins"]["finetuned"], 4)
        self.assertEqual(report["by_intent"]["health"]["n_trials"], 4)

    def test_holdout_panel_uses_the_tight_price_window(self) -> None:
        products = {}
        prices = [1, 2, 3, 4, 5, 6, 7, 8, 20, 20.2, 20.4, 20.6, 20.8, 21, 21.2, 21.4]
        for index, price in enumerate(prices):
            asin = f"A{index}"
            products[asin] = {"parent_asin": asin, "price": price}
        panels = holdout_panels(products, list(products))
        self.assertEqual(len(panels), 1)
        chosen = [products[asin]["price"] for asin in panels[0]["parent_asins"]]
        self.assertLess(max(chosen) / min(chosen), 1.1)


class BackendGuard(unittest.TestCase):
    def test_unsloth_is_rejected_on_apple_silicon(self) -> None:
        apple = platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
        if not apple:
            reject_unsloth_on_apple_silicon()
            return
        with self.assertRaises(SystemExit) as caught:
            reject_unsloth_on_apple_silicon()
        message = str(caught.exception)
        self.assertIn("trl", message)
        self.assertIn("mlx", message)

    def test_training_modules_do_not_import_backends(self) -> None:
        import src.evaluate  # noqa: F401
        import src.finetune  # noqa: F401

        for name in ("unsloth", "trl", "peft", "torch", "mlx", "mlx_lm"):
            self.assertNotIn(name, sys.modules)


class Intents(unittest.TestCase):
    def test_three_study_intents(self) -> None:
        self.assertEqual(list(INTENTS), ["general", "health", "flavour"])
        for spec in INTENTS.values():
            self.assertTrue(spec["shopper"])
            self.assertTrue(spec["brief"])
        self.assertEqual(len(MAIN_RUN_ARMS), 6)
        self.assertIn("original", MAIN_RUN_ARMS)
        self.assertNotIn("probe", MAIN_RUN_ARMS)


if __name__ == "__main__":
    unittest.main()
