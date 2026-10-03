# Project plan — intent-conditioned drink description optimiser

Fine-tune a generative model to write non-alcoholic drink product descriptions
that AI shopping agents rank highly, conditioned on a stated shopper intent.

Measurement loop: show an agent a shortlist of 8 comparable products under one
intent, each product carrying a randomly assigned description variant, record
the returned ranking, repeat tens of thousands of times, then recover per
product and intent which variant wins. Train on the winners.

---

## 1. Current state

Working and verified — do not rewrite:


| File                       | Status                                                                              |
| -------------------------- | ----------------------------------------------------------------------------------- |
| `src/config.py`            | Study parameters, styles, intents. Needs the intent rework in §3.                   |
| `src/download_data.py`     | Downloads Amazon Reviews 2023 grocery metadata (1.38 GB) to `.hf_cache`. Done.      |
| `src/extract_drinks.py`    | 603,274 items → 27,580 classified non-alcoholic drinks. Done.                       |
| `src/build_panels.py`      | 25 panels × 8 products = 200 Fruit Juice products, $7–$40. Done.                    |
| `src/llm.py`               | Async OpenRouter caller, bounded concurrency, retries, JSON repair, progress. Done. |
| `src/generate_variants.py` | Written for intent-agnostic variants. **Must be reworked** per §4.                  |


Committed data artifacts (so the run can happen on another machine without the
1.38 GB download): `data/study_products.jsonl`, `data/panels.json`.

To build: `src/run_ranking.py`, `src/score_variants.py`,
`src/build_finetune_data.py`, `src/finetune.py`, `src/evaluate.py`.

---



## 2. Parameters


| Parameter                                    | Value                                                           |
| -------------------------------------------- | --------------------------------------------------------------- |
| Category / price band                        | Fruit Juice, $7–$40                                             |
| Products                                     | 200                                                             |
| Panels                                       | 25 fixed panels of 8 comparable products                        |
| Within-panel price ratio                     | median 1.05×, max 1.13×                                         |
| Intents                                      | 3 (see §3)                                                      |
| Variants per product per intent              | 6 (`original` + 5 generated)                                    |
| Descriptions generated                       | 200 × 3 × 5 = 3,000                                             |
| Shortlist size                               | 8                                                               |
| Total ranking trials                         | 24,000 (8,000 per intent)                                       |
| Observations per product-intent-variant cell | ~53                                                             |
| Expected per-cell recovery                   | ~70% (simulated)                                                |
| Product × intent cells                       | 600                                                             |
| SFT examples                                 | 1,200 (top-2 arms per cell)                                     |
| DPO pairs                                    | ≤1,200 (up to 2 per cell, gated on score gap)                   |
| Shrinkage prior                              | 25 pseudo-observations                                          |
| Generation model                             | `stealth/space-bunny-alpha` via OpenRouter (free; reasoning effort low) |
| Ranking model                                | `openai/gpt-4.1-mini` via OpenRouter                            |
| Fine-tune base                               | `Qwen/Qwen3-4B-Instruct-2507` — verify exact repo id before use |


Trial budget is tunable: 15,000 gives ~69% recovery, 36,000 gives ~73%. The run
must be resumable so it can be stopped on a clock rather than a target.

---



## 3. Intents

Three intents, each a stable key plus a shopper-facing brief. Replace
`CUSTOMER_INTENTS` in `config.py` with exactly these three.

```python
INTENTS = {
    "general": {
        "label": "General",
        "shopper": "I'm browsing for a good fruit juice. Show me the best option.",
        "brief": "Broad ecommerce audience. The strongest balanced canonical description.",
    },
    "health": {
        "label": "Health",
        "shopper": "I care about what's in my drinks — nutrition, ingredients, nothing artificial.",
        "brief": "Shopper prioritises nutritional characteristics, ingredients and everyday refreshment.",
    },
    "flavour": {
        "label": "Flavour",
        "shopper": "I want something that genuinely tastes great.",
        "brief": "Shopper primarily cares about taste, aroma, texture and sensory experience.",
    },
}
```

`shopper` goes in the ranking prompt. `brief` goes in the generation prompt.
`premium`, `social_occasion` and `alcohol_replacement` are deferred — see §10.

---



## 4. Variant generation (`src/generate_variants.py`, rework)

Variants become intent-specific. For each product × intent, produce 6 arms:

- `original` — the untouched Amazon description. Identical across all three
intents; it is the real-world baseline the fine-tune must beat. No API call.
- Five generated arms that hold the intent topic fixed and vary **rhetorical
form**, so the winner is attributable to a controlled dimension rather than
to sampling luck:


| Arm          | Instruction                                                                    |
| ------------ | ------------------------------------------------------------------------------ |
| `direct`     | Lead with the single strongest intent-relevant fact, plainly stated.           |
| `sensory`    | Concrete sensory language — taste, aroma, texture, temperature.                |
| `quantified` | Foreground numbers: volume, servings, percentages, counts.                     |
| `use_case`   | Concrete situations and moments where this drink fits.                         |
| `assurance`  | Trust signals present in the source: brand provenance, rating, certifications. |


Plus `probe` (deliberately vague, no concrete facts) generated for the pilot
only, as a manipulation check.

**Grounding rules, enforced in the system prompt and re-checked after
generation:** use only facts present in the listing; never invent ingredients,
certifications, health claims, awards or nutritional numbers; never mention
alcohol; no markdown or headings.

**Length control:** pin every generated arm to 70 words ±10%. Length is the
strongest single driver of LLM preference and will swamp the style effect if
it drifts. Reject and regenerate anything outside the band. The `original` arm
is intentionally not length-matched — it is the baseline, not a style.

Output `data/variants.jsonl`, one row per product × intent × arm:

```json
{"parent_asin": "B0...", "intent": "health", "arm": "direct",
 "text": "...", "words": 71, "model": "stealth/space-bunny-alpha"}
```

Resumable: skip `(parent_asin, intent, arm)` triples already on disk.

---



## 5. Ranking run (`src/run_ranking.py`, new)

The core measurement. Must survive 24,000 calls unattended.

**Trial construction.** Deterministic from a seed so a trial can be
reconstructed and resumed:

- `trial_id = f"{intent}-{panel_id}-{rep:05d}"`
- Intent cycles so all three stay balanced if the run is cut short.
- Panel cycles through the 25 panels.
- Seed a `random.Random` on `trial_id`, then draw: one arm per product
(uniform over the 6), and a shuffled display order.
- Display order must be randomised every trial — position bias in list ranking
is large and is not otherwise controlled.

**Prompt.** System prompt casts the model as a shopping assistant returning a
full ranking. User prompt contains the intent `shopper` line and 8 option
cards, each with `option_id`, title, brand, price, rating, review count, and
the assigned variant text. Require JSON:

```json
{"ranking": ["O3", "O7", "O1", "O5", "O2", "O8", "O4", "O6"]}
```

**Validation.** The returned ranking must be a permutation of all 8 option ids.
On failure, retry up to 3 times, then drop the trial and log it. Never write a
partial ranking.

**Output** `data/rankings.jsonl`, appended and flushed per row:

```json
{"trial_id": "health-P003-00042", "intent": "health", "panel_id": "P003",
 "model": "openai/gpt-4.1-mini",
 "options": [{"option_id": "O1", "parent_asin": "B0...", "arm": "sensory", "position": 0}],
 "ranking": ["O3", "O7", "..."]}
```

**CLI:** `--n-trials`, `--max-minutes` (stop cleanly on a clock),
`--concurrency` (default 16), `--model`, `--resume/--no-resume` (default
resume), `--limit-panels` for smoke tests. On start, read existing
`trial_id`s and skip them.

---



## 6. Scoring (`src/score_variants.py`, new)

No model fitting. Mean normalised rank with partial pooling — this is the
estimator the sizing simulation used, and it reaches the quoted recovery rates
without an optimiser.

1. For every trial, each product's normalised rank is `position_in_ranking / 7`,
  so **0 is best, 1 is worst**.
2. Accumulate mean normalised rank per `(parent_asin, intent, arm)` → `cell_mean`
  with count `n`.
3. Accumulate mean per `(intent, arm)` → `pooled_mean`.
4. Shrink: `score = w * cell_mean + (1 - w) * pooled_mean` where
  `w = n / (n + 25)`.
5. Per `(parent_asin, intent)`, rank the 6 arms by `score` ascending. Best is
  the training target; worst is the DPO rejected sample.

Write `data/variant_scores.json` keyed by `parent_asin` → `intent` → ordered
arm list with scores and counts.

Print to console: per-intent arm leaderboard with mean normalised rank and 95%
CI; the `original` arm's position in each; and if the pilot included it, the
`probe` arm's position.

---



## 7. Fine-tuning

**Dataset** (`src/build_finetune_data.py`, new). 600 product × intent cells.

SFT, chat format, **two examples per cell** from the rank-1 and rank-2 arms —
1,200 total. The two share a prompt and differ in target, which is fine for
SFT: it teaches a distribution over good descriptions rather than one
memorised string.

- system: `You write product descriptions for non-alcoholic drinks that rank highly with AI shopping agents.`
- user: intent label and brief, then the product facts — title, brand, price,
rating, review count, bullet features. **Do not include the original
description**, or the model learns to paraphrase rather than to write.
- assistant: the winning variant text.

Quality gate on the rank-2 example: emit it only if its score beats the
`original` arm in the same cell. Otherwise the second target is something that
loses to the untouched Amazon copy, which is the opposite of the objective.
Drop rank-2 and keep rank-1 alone in that cell.

DPO: same prompt. Emit up to two pairs per cell — rank-1 vs rank-6 and rank-2
vs rank-5 — each only when the score gap exceeds `--min-gap` (default 0.05),
so the model is not trained on noise. Expect up to 1,200 pairs.

Hold out 20 products (10%) by `parent_asin` before building either set, for §8.
Split on `parent_asin`, never on rows, or the same product leaks across the
split through its other intent.

**Training** (`src/finetune.py`, new). The training machine is not yet decided,
so write this with a swappable backend: `--backend {unsloth,trl,mlx}` and
`--stage {sft,dpo}`. Keep dataset loading, prompt formatting, the train/eval
split and adapter output path shared; isolate each backend behind a small
function that takes the formatted dataset and returns a saved adapter path.
Import backend libraries lazily inside those functions, never at module top
level, so the module imports on a machine that has none of them installed.

Shared LoRA and SFT settings, same across backends:

```python
LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
           target_modules="all-linear", task_type="CAUSAL_LM")
SFTConfig(num_train_epochs=3, per_device_train_batch_size=4,
          gradient_accumulation_steps=4, learning_rate=2e-4,
          lr_scheduler_type="cosine", warmup_ratio=0.03,
          bf16=True, max_length=1024, logging_steps=10)
```

DPO stage: initialise from the SFT adapter, `beta=0.1`,
`learning_rate=5e-6`, 1 epoch.


| Backend   | Target                                       | Notes                                                                                                                                  |
| --------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `unsloth` | Rented CUDA GPU (Colab T4/L4, Modal, RunPod) | `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`, then hand the model to TRL's `SFTTrainer`. Fastest and lowest memory.     |
| `trl`     | CUDA or Apple MPS                            | Plain `transformers` + `peft` + TRL. Works everywhere; set `bf16=False, fp16=False` on MPS. The portable fallback.                     |
| `mlx`     | Apple Silicon                                | `mlx_lm.lora`. Fastest on the M3 Max but a separate adapter format, so `evaluate.py` must load it through `mlx_lm` rather than `peft`. |


> **Unsloth is CUDA-only — it does not support Apple Silicon.** Selecting
> `--backend unsloth` on the M3 Max must fail fast with a clear message
> pointing at `trl` or `mlx`, not fall through to a confusing import error.

**Cost.** 1,200 examples × ~700 tokens × 3 epochs is about 2.5M tokens:
a few minutes on any CUDA GPU, well under an hour on the M3 Max.

---



## 8. Evaluation (`src/evaluate.py`, new)

The result is only meaningful against baselines. For each of the 20 held-out
products × 3 intents, generate a description with each of:

1. the fine-tuned model,
2. the **prompt-only baseline** — a strong model prompted with a plain-English
  description of the winning style found in §6,
3. the original Amazon description.

Then run fresh ranking trials on held-out panels where each product carries one
of the three, rotating assignment, and report win rates with confidence
intervals.

The prompt-only baseline is the control that makes the project's claim
meaningful. If a prompt matches the fine-tune, the fine-tune added nothing.

---



## 9. Runbook

```bash
# one-time, only if data/ artifacts are missing
python -m src.download_data --which meta
python -m src.extract_drinks --out data/drinks.jsonl
python -m src.build_panels

# API stages (needs OPENROUTER_API_KEY in .env)
python -m src.generate_variants --limit 8            # smoke test
python -m src.generate_variants                      # 3,000 descriptions, ~15 min
python -m src.run_ranking --n-trials 200 --limit-panels 3   # smoke test
python -m src.run_ranking --n-trials 2000            # pilot, check probe arm
python -m src.run_ranking --n-trials 24000 --max-minutes 60 # main run, resumable

# analysis and training
python -m src.score_variants
python -m src.build_finetune_data
python -m src.finetune --stage sft
python -m src.evaluate
```

Smoke-test the whole chain end to end at tiny scale before the main run.
Discovering a JSON parsing bug at trial 20,000 is the failure mode that costs
the session.

---



## 10. Open decisions and risks

1. **1,200 SFT examples is workable but not generous.** It comes from 600
  cells × top-2 arms, so the prompts are only 600 distinct products-plus-intent
   combinations. If the fine-tune overfits — watch for it reproducing training
   descriptions near-verbatim on held-out products — the fix is to widen to a
   second juice-adjacent category for ~400 products rather than to mine more
   arms per cell.
2. **Deferred intents.** `premium`, `social_occasion` and `alcohol_replacement`
  are good intents but poor fits for the Fruit Juice slice — no de-alcoholised
   products are in it, and most of the catalogue is not plausibly premium or
   party-oriented. They become viable if the category widens.
3. **Effect size is unmeasured.** Every sizing number assumes the copy effect
  is a quarter of the spread in product quality. The 2,000-trial pilot
   measures it. If the `probe` arm does not rank last, the experiment is not
   sensitive to copy at all and the design needs revisiting before the main run.
4. **Rate limits, not cost, are the constraint.** 24,000 ranking calls is about
   $20 on `gpt-4.1-mini`. Check the OpenRouter rate limit before launching;
   it determines whether the main run takes 40 minutes or 4 hours.
5. **Verify the Qwen3 repo id** on Hugging Face before writing training code.

