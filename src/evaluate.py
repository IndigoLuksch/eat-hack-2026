"""Compare the fine-tune with a prompt-only baseline and the original copy.

For every held-out product and intent, three descriptions are prepared: one
from the fine-tuned model, one from a strong model told the winning rhetorical
form in plain English, and the untouched Amazon description. Fresh ranking
trials then rotate those three across held-out shortlists. Win rates and mean
normalised ranks are reported with 95% confidence intervals.

The prompt-only baseline is the control that makes the fine-tune's claim
meaningful. If it matches the fine-tune, the fine-tune added nothing.

mlx adapters are loaded through mlx_lm. PEFT adapters are loaded through peft.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from src.build_finetune_data import prompt_messages, sft_user
from src.generate_variants import validate_variant
from src.config import (
    ARMS,
    BASE_MODEL,
    CONCURRENCY,
    DATA_DIR,
    INTENTS,
    PANEL_SIZE,
    RANK_MODEL,
    ROOT,
    SFT_SYSTEM,
    VARIANT_MODEL,
)
from src.jsonl import read_jsonl
from src.llm import Caller, run_bounded
from src.run_ranking import RANK_SYSTEM, parse_ranking, rank_user, trial_grid
from src.score_variants import mean_ci, normalised_rank, winning_generated_arm

app = typer.Typer(add_completion=False)
console = Console()

SOURCES = ("finetuned", "prompt_only", "original")
RANK_ATTEMPTS = 3


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point rate, lower, upper) for a 95% Wilson interval."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    proportion = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (proportion + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(proportion * (1.0 - proportion) / n + z2 / (4.0 * n * n)) / denom
    return proportion, max(0.0, centre - margin), min(1.0, centre + margin)


def assign_sources(asins: list[str], rep: int) -> dict[str, str]:
    """Rotate the three description sources so each product cycles through them."""
    return {asin: SOURCES[(index + rep) % len(SOURCES)] for index, asin in enumerate(asins)}


def holdout_panels(
    products: dict[str, dict[str, Any]],
    holdout: list[str],
    panel_size: int = PANEL_SIZE,
    max_ratio: float = 1.25,
) -> list[dict[str, Any]]:
    """Tightest price windows among the held-out products.

    A random 10% of the catalogue does not fall inside the original panels, so
    chopping the sorted list into blocks of eight can span a 2× price range.
    Prefer the narrowest window, and keep a second only while it stays tight.
    """
    rows = [products[asin] for asin in holdout if asin in products]
    rows.sort(key=lambda row: (float(row["price"]), row["parent_asin"]))
    n = len(rows)
    if n < panel_size:
        return []
    windows = []
    for start in range(n - panel_size + 1):
        chunk = rows[start : start + panel_size]
        lo = float(chunk[0]["price"])
        hi = float(chunk[-1]["price"])
        ratio = hi / lo if lo else float("inf")
        windows.append((ratio, start, chunk))
    windows.sort(key=lambda item: (item[0], item[1]))
    panels = []
    used: set[int] = set()
    for ratio, start, chunk in windows:
        indexes = range(start, start + panel_size)
        if any(index in used for index in indexes):
            continue
        if panels and ratio > max_ratio:
            continue
        used.update(indexes)
        lo = float(chunk[0]["price"])
        hi = float(chunk[-1]["price"])
        panels.append(
            {
                "panel_id": f"H{len(panels):03d}",
                "parent_asins": [row["parent_asin"] for row in chunk],
                "price_min": lo,
                "price_max": hi,
            }
        )
    return panels


def prompt_only_messages(product: dict[str, Any], intent_key: str, arm: str) -> list[dict[str, str]]:
    user = (
        f"{sft_user(product, intent_key)}\n\n"
        f"Style: {ARMS[arm]}\n"
        "Write one product description in that style, about 70 words. "
        "Use only the facts above. Do not invent ingredients, certifications, "
        "health claims, awards or numbers. Never mention alcohol. "
        "No markdown or headings."
    )
    return [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": user},
    ]


def condition_means(options: list[dict[str, Any]], ranking: list[str]) -> dict[str, float]:
    rank_of = {option_id: index for index, option_id in enumerate(ranking)}
    buckets: dict[str, list[float]] = defaultdict(list)
    n_options = len(ranking)
    for option in options:
        buckets[option["source"]].append(normalised_rank(rank_of[option["option_id"]], n_options))
    return {source: sum(values) / len(values) for source, values in buckets.items()}


def trial_winner(options: list[dict[str, Any]], ranking: list[str]) -> Optional[str]:
    """Source with the best mean normalised rank. None when two sources tie."""
    means = condition_means(options, ranking)
    if not means:
        return None
    best = min(means.values())
    leaders = [source for source, value in means.items() if math.isclose(value, best, abs_tol=1e-9)]
    if len(leaders) != 1:
        return None
    return leaders[0]


def summarise(trials: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    for trial in trials:
        ranking = trial.get("ranking")
        options = trial.get("options")
        if not isinstance(ranking, list) or not isinstance(options, list):
            continue
        if len(ranking) < 2:
            continue
        ids = [option.get("option_id") for option in options]
        if set(ids) != set(ranking):
            continue
        usable.append(trial)

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        wins = {source: 0 for source in SOURCES}
        ties = 0
        ranks: dict[str, list[float]] = {source: [] for source in SOURCES}
        for trial in group:
            winner = trial_winner(trial["options"], trial["ranking"])
            if winner is None:
                ties += 1
            else:
                wins[winner] += 1
            for source, value in condition_means(trial["options"], trial["ranking"]).items():
                if source in ranks:
                    ranks[source].append(value)
        n = len(group)
        return {
            "n_trials": n,
            "ties": ties,
            "wins": wins,
            "win_rate": {
                source: dict(
                    zip(("rate", "ci_low", "ci_high"), wilson(wins[source], n))
                )
                for source in SOURCES
            },
            "mean_normalised_rank": {
                source: _rank_report(values) for source, values in ranks.items()
            },
        }

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in usable:
        by_intent[trial.get("intent", "")].append(trial)
    return {
        "n_trials": len(usable),
        "overall": block(usable),
        "by_intent": {intent: block(group) for intent, group in sorted(by_intent.items())},
    }


def _rank_report(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean, lo, hi = mean_ci(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)}


def _print_report(report: dict[str, Any]) -> None:
    console.print(
        "Mean normalised rank is 0 at the top of the list and 1 at the bottom. "
        "A win is the source with the best mean rank inside a trial."
    )
    blocks = [("overall", report["overall"])] + list(report["by_intent"].items())
    for name, block in blocks:
        table = Table(title=f"{name} ({block['n_trials']} trials, {block['ties']} ties)")
        table.add_column("source")
        table.add_column("win rate", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("mean rank", justify="right")
        table.add_column("rank 95% CI", justify="right")
        for source in SOURCES:
            rate = block["win_rate"][source]
            rank = block["mean_normalised_rank"][source]
            table.add_row(
                source,
                f"{rate['rate']:.3f}",
                f"[{rate['ci_low']:.3f}, {rate['ci_high']:.3f}]",
                f"{rank['mean']:.3f}",
                f"[{rank['ci_low']:.3f}, {rank['ci_high']:.3f}]",
            )
        console.print(table)


def _generate_peft(
    base_model: str,
    adapter: Path,
    batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_dir = adapter if (adapter / "tokenizer_config.json").exists() else Path(base_model)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    device = next(model.parameters()).device
    texts: list[str] = []
    for messages in batches:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return texts


def _generate_unsloth(
    base_model: str,
    adapter: Path,
    batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    from src.finetune import reject_unsloth_on_apple_silicon

    reject_unsloth_on_apple_silicon()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter))
    FastLanguageModel.for_inference(model)
    return _generate_peft_with_model(model, tokenizer, batches, max_new_tokens, temperature)


def _generate_peft_with_model(
    model: Any,
    tokenizer: Any,
    batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    texts: list[str] = []
    for messages in batches:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return texts


def _generate_mlx(
    base_model: str,
    adapter: Path,
    batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    from mlx_lm import generate, load

    model, tokenizer = load(base_model, adapter_path=str(adapter))
    texts: list[str] = []
    for messages in batches:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
                temp=temperature,
            )
        except TypeError:
            text = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
        if isinstance(text, str) and text.startswith(prompt):
            text = text[len(prompt) :]
        texts.append(str(text).strip())
    return texts


def generate_finetuned(
    backend: str,
    base_model: str,
    adapter: Path,
    batches: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    try:
        if backend == "mlx":
            return _generate_mlx(base_model, adapter, batches, max_new_tokens, temperature)
        if backend == "unsloth":
            return _generate_unsloth(base_model, adapter, batches, max_new_tokens, temperature)
        return _generate_peft(base_model, adapter, batches, max_new_tokens, temperature)
    except ImportError as exc:
        raise SystemExit(
            f"Backend {backend!r} is not installed ({exc}). See requirements-train.txt."
        ) from exc


def _load_descriptions(path: Path) -> dict[tuple[str, str, str], str]:
    found: dict[tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        key = (row.get("parent_asin"), row.get("intent"), row.get("source"))
        text = row.get("text")
        if all(key) and text:
            found[(str(key[0]), str(key[1]), str(key[2]))] = text
    return found


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


@app.command()
def main(
    n_trials: int = typer.Option(300, help="Ranking trials across held-out panels"),
    backend: str = typer.Option("trl", help="How to load the fine-tuned adapter: trl, unsloth, or mlx"),
    adapter: Optional[Path] = typer.Option(None, help="Adapter directory. Defaults to the SFT adapter for --backend"),
    base_model: str = typer.Option(BASE_MODEL),
    prompt_model: str = typer.Option(VARIANT_MODEL, help="Strong model used for the prompt-only baseline"),
    rank_model: str = typer.Option(RANK_MODEL),
    concurrency: int = typer.Option(CONCURRENCY),
    max_minutes: Optional[float] = typer.Option(None, help="Stop API calls cleanly after this many minutes"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip ranking trials already on disk"),
    temperature: float = typer.Option(0.7, help="Sampling temperature for generated descriptions"),
    rank_temperature: float = typer.Option(0.0),
    holdout_path: Path = typer.Option(DATA_DIR / "holdout_asins.json", "--holdout"),
    scores_path: Path = typer.Option(DATA_DIR / "variant_scores.json", "--scores"),
    products_path: Path = typer.Option(DATA_DIR / "study_products.jsonl", "--products"),
    descriptions_path: Path = typer.Option(DATA_DIR / "eval_descriptions.jsonl", "--descriptions"),
    rankings_path: Path = typer.Option(DATA_DIR / "eval_rankings.jsonl", "--rankings"),
    report_path: Path = typer.Option(DATA_DIR / "eval_report.json", "--report"),
    api_key: Optional[str] = typer.Option(None, envvar="OPENROUTER_API_KEY"),
) -> None:
    """Generate the three eval descriptions and rank them on held-out shortlists."""
    if backend not in {"trl", "unsloth", "mlx"}:
        raise typer.BadParameter("backend must be trl, unsloth, or mlx")
    if not holdout_path.exists():
        raise typer.BadParameter(f"{holdout_path} is missing. Run build_finetune_data first.")
    if not scores_path.exists():
        raise typer.BadParameter(f"{scores_path} is missing. Run score_variants first.")

    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    products = {row["parent_asin"]: row for row in read_jsonl(products_path)}
    adapter_path = adapter or (ROOT / "outputs" / "adapters" / backend / "sft")
    if not adapter_path.exists():
        raise typer.BadParameter(
            f"Fine-tuned adapter not found at {adapter_path}. "
            "Train with finetune.py or pass --adapter."
        )

    winning = {intent: winning_generated_arm(scores, intent) for intent in INTENTS}
    missing = [intent for intent, arm in winning.items() if not arm]
    if missing:
        raise typer.BadParameter(
            f"No winning generated arm for {', '.join(missing)}. Score a run that includes those intents."
        )
    for intent, arm in winning.items():
        console.print(f"Prompt-only style for {intent}: [cyan]{arm}[/cyan] — {ARMS[arm]}")

    panels = holdout_panels(products, holdout)
    if not panels:
        raise typer.BadParameter(
            f"Need at least {PANEL_SIZE} held-out products to build a shortlist; found {len(holdout)}."
        )
    placed = {asin for panel in panels for asin in panel["parent_asins"]}
    console.print(
        f"{len(panels)} held-out panels, {len(holdout) - len(placed)} held-out products left over"
    )
    for panel in panels:
        lo = float(panel["price_min"])
        hi = float(panel["price_max"])
        console.print(f"  {panel['panel_id']} ${lo:.2f}–${hi:.2f} ({hi / lo:.2f}×)")

    descriptions_path.parent.mkdir(parents=True, exist_ok=True)
    if not descriptions_path.exists():
        descriptions_path.touch()
    have = _load_descriptions(descriptions_path)
    deadline = None if max_minutes is None else time.monotonic() + max_minutes * 60

    for asin in holdout:
        product = products.get(asin)
        if product is None:
            continue
        for intent_key in INTENTS:
            key = (asin, intent_key, "original")
            if key in have:
                continue
            text = product.get("description_text") or ""
            row = {
                "parent_asin": asin,
                "intent": intent_key,
                "source": "original",
                "text": text,
                "model": "none",
            }
            _append(descriptions_path, row)
            have[key] = text

    prompt_jobs = []
    for asin in holdout:
        if asin not in products:
            continue
        for intent_key, arm in winning.items():
            if (asin, intent_key, "prompt_only") in have:
                continue
            prompt_jobs.append((asin, intent_key, arm))

    if prompt_jobs:
        caller = Caller(
            model=prompt_model,
            concurrency=concurrency,
            temperature=temperature,
            max_tokens=400,
            api_key=api_key,
        )

        async def write_prompt(job: tuple[str, str, str]) -> Optional[dict[str, Any]]:
            asin, intent_key, arm = job
            messages = prompt_only_messages(products[asin], intent_key, arm)
            facts = sft_user(products[asin], intent_key)
            prompt = messages[1]["content"]
            feedback: Optional[str] = None
            last = "no attempt"
            for _ in range(4):
                user = prompt + '\nReturn JSON: {"description": "..."}'
                if feedback:
                    user += f"\nPrevious attempt rejected: {feedback}. Rewrite it."
                data = await caller.json(messages[0]["content"], user)
                text = str(data.get("description") or data.get("text") or "").strip()
                reason = validate_variant(text, facts, arm)
                if reason is None:
                    return {
                        "parent_asin": asin,
                        "intent": intent_key,
                        "source": "prompt_only",
                        "arm": arm,
                        "text": text,
                        "model": prompt_model,
                    }
                last = reason
                feedback = reason
            raise ValueError(f"{asin}/{intent_key}: {last}")

        def store(row: dict[str, Any]) -> None:
            _append(descriptions_path, row)
            have[(row["parent_asin"], row["intent"], row["source"])] = row["text"]

        asyncio.run(
            run_bounded(
                prompt_jobs,
                write_prompt,
                concurrency=concurrency,
                desc="prompt-only",
                deadline=deadline,
                on_result=store,
            )
        )

    finetune_jobs = []
    for asin in holdout:
        if asin not in products:
            continue
        for intent_key in INTENTS:
            if (asin, intent_key, "finetuned") not in have:
                finetune_jobs.append((asin, intent_key))
    clock_stopped = deadline is not None and time.monotonic() >= deadline
    if clock_stopped and finetune_jobs:
        console.print("[yellow]Stopped on the clock before fine-tuned generation.[/yellow]")
        finetune_jobs = []
    if finetune_jobs:
        console.print(f"[cyan]{len(finetune_jobs)} fine-tuned descriptions[/cyan] via {backend}")
        messages = [prompt_messages(products[asin], intent_key) for asin, intent_key in finetune_jobs]
        texts = generate_finetuned(backend, base_model, adapter_path, messages, 180, temperature)
        for (asin, intent_key), text in zip(finetune_jobs, texts):
            if not text:
                continue
            row = {
                "parent_asin": asin,
                "intent": intent_key,
                "source": "finetuned",
                "text": text,
                "model": str(adapter_path),
            }
            _append(descriptions_path, row)
            have[(asin, intent_key, "finetuned")] = text

    needed = [
        (asin, intent_key, source)
        for asin in placed
        for intent_key in INTENTS
        for source in SOURCES
        if (asin, intent_key, source) not in have
    ]
    if needed and not clock_stopped:
        raise typer.BadParameter(
            f"{len(needed)} held-out descriptions are still missing, so ranking did not start."
        )
    if needed and clock_stopped:
        console.print("[yellow]Stopped on the clock with descriptions still missing.[/yellow]")
        return

    if not resume and rankings_path.exists():
        rankings_path.unlink()
    done = {row["trial_id"] for row in read_jsonl(rankings_path) if row.get("trial_id")}
    panel_ids = [panel["panel_id"] for panel in panels]
    by_id = {panel["panel_id"]: panel for panel in panels}
    grid = trial_grid(n_trials, list(INTENTS), panel_ids)
    pending = [
        (intent, panel_id, rep)
        for intent, panel_id, rep in grid
        if f"eval-{intent}-{panel_id}-{rep:05d}" not in done
    ]
    console.print(f"[cyan]{len(pending)} ranking trials[/cyan] ({len(done)} already on disk)")
    rankings_path.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        ranker = Caller(
            model=rank_model,
            concurrency=concurrency,
            temperature=rank_temperature,
            max_tokens=300,
            api_key=api_key,
        )
        drops: list[str] = []

        async def rank_one(spec: tuple[str, str, int]) -> Optional[dict[str, Any]]:
            intent_key, panel_id, rep = spec
            tid = f"eval-{intent_key}-{panel_id}-{rep:05d}"
            asins = list(by_id[panel_id]["parent_asins"])
            sources = assign_sources(asins, rep)
            order = list(asins)
            random.Random(tid).shuffle(order)
            cards = []
            for position, asin in enumerate(order):
                source = sources[asin]
                text = have.get((asin, intent_key, source))
                if not text:
                    drops.append(tid)
                    return None
                product = products[asin]
                cards.append(
                    {
                        "option_id": f"O{position + 1}",
                        "parent_asin": asin,
                        "source": source,
                        "position": position,
                        "title": product.get("title") or "",
                        "brand": product.get("store") or "Unknown",
                        "price": float(product.get("price") or 0),
                        "rating": product.get("average_rating"),
                        "reviews": product.get("rating_number"),
                        "text": text,
                    }
                )
            user = rank_user(INTENTS[intent_key]["shopper"], cards)
            option_ids = [card["option_id"] for card in cards]
            prompt = user
            for _ in range(RANK_ATTEMPTS):
                try:
                    payload = await ranker.json(RANK_SYSTEM, prompt)
                except Exception:  # noqa: BLE001
                    continue
                ranking = parse_ranking(payload, option_ids)
                if ranking is None:
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
                    "model": rank_model,
                    "options": [
                        {
                            "option_id": card["option_id"],
                            "parent_asin": card["parent_asin"],
                            "source": card["source"],
                            "position": card["position"],
                        }
                        for card in cards
                    ],
                    "ranking": ranking,
                }
            drops.append(tid)
            return None

        asyncio.run(
            run_bounded(
                pending,
                rank_one,
                concurrency=concurrency,
                desc="eval rankings",
                deadline=deadline,
                on_result=lambda row: _append(rankings_path, row),
            )
        )
        if drops:
            console.print(f"[yellow]Dropped {len(drops)} eval trials[/yellow]")

    report = summarise(read_jsonl(rankings_path))
    _print_report(report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]report[/green] → {report_path}")


if __name__ == "__main__":
    app()
