# Shelf Shift — product descriptions optimised for AI shopping agents

**EAT_HACK 2026 · Track 1: Human Truth**

When an AI shopping agent browses the web, it does not look at product images. It focuses on text descriptions, so we asked ourselves: how can brands optimise their product descriptions for shopping agents?

Using 200 real Amazon listings, we constructed catalogues of near-identical fruit juices. Each product has 6 alternate descriptions, and a random one is chosen for each catalogue. AI shopping agents were given customer requests (general, healthy choice, or best flavour) and were asked to rank products in 10,854 trials.

Next, we used the winning descriptions to fine-tune Qwen3-4B LoRA. We also added a new input channel: customer request. Now, this custom LLM outputs product descriptions optimised for a customer’s request and targeted at AI shopping agents.

The demo website allows you to write your own product description and see how well it performs compared to our model and Claude Opus: https://eat-hack-demo.onrender.com/

---

## Results

The untouched Amazon description finished **last of six arms** on all three shopper intents (no overlapping 95% CIs).

| Shopper intent | Best rhetorical form | Mean normalised rank | `original` | Products where rewriting won |
| --- | --- | --- | --- | --- |
| Flavour | `sensory` | 0.448 `[0.439, 0.457]` | 0.586 `[0.576, 0.595]` | 97 / 104 (93%) |
| General | `sensory` | 0.459 `[0.450, 0.468]` | 0.575 `[0.566, 0.584]` | 97 / 104 (93%) |
| Health | `assurance` | 0.482 `[0.472, 0.491]` | 0.547 `[0.537, 0.557]` | 94 / 104 (90%) |

Normalised rank: 0 = top of shortlist, 1 = bottom. On an 8-product shortlist, the flavour gap (0.138) is roughly one full place — from a rewrite that adds no new facts.

Two findings beyond “LLMs prefer better copy”:

1. **Winning form depends on intent, and the order inverts.** `sensory` ranks first for flavour/general but fourth for health; `assurance` (brand provenance, certifications already in the listing) ranks first for health and fourth for flavour.
2. **The leaderboard is an average, not a universal style.** Even on flavour, `sensory` is best for only 46 / 104 products (9 / 104 on health). The model learns the per-product pattern.

---

## Method

Attribution is the hard part: real shortlists confound copy with price, brand, and quality. We remove competing explanations:

| Control | Design |
| --- | --- |
| Within-product comparison | Same product appears with different descriptions; brand and quality cancel out |
| Near-identical competitors | Panels of 8 Fruit Juice products, price band $7–$40; median within-panel price ratio **1.05×** |
| Length pinned | Generated arms: 70 words ±10%; rejected otherwise |
| Copy-only ranking | Title, brand, price, rating, review count stripped from option cards |
| Display order reshuffled | Every trial; controls position bias |
| Random arm assignment | Seeded from trial id for exact reconstruction |

**Arms.** Six per product × intent. `original` is untouched Amazon copy (not length-matched — the real-world baseline). The other five fix intent topic and vary rhetorical form: `direct`, `sensory`, `quantified`, `use_case`, `assurance`.

**Grounding.** Generated copy is constrained to the listing. Numbers absent from the source are rejected programmatically, as are markdown, alcohol mentions, and out-of-band length. No invented certifications or health claims.

**Scoring.** Mean normalised rank with partial pooling (25-observation prior toward intent-by-arm mean). Median observations per cell: 48.

**Splits.** 25 panels → Half A (13 panels, 104 products) for measurement; Half B (12 panels, 96 products) held back. 10% of products held out by `parent_asin` before training so products cannot leak across intents.

---

## Model

Winning arms become supervised targets (best arm per product × intent; second-best only if it also beats `original`). DPO pairs: best-vs-worst and second-vs-fifth, gated on a minimum score gap. Base model: `Qwen/Qwen3-4B-Instruct-2507` with LoRA.

Fine-tuning is compared against a **prompt-only baseline** (a strong model told the winning style in plain English) on held-out products via `evaluate.py`. If prompting matches the fine-tune, the fine-tune added nothing.

### Demo

[Live demo](https://eat-hack-demo.onrender.com/) — write a description for a real product and shopper request; compare average rank against our model and Claude Opus across five ranking rounds with shared display orders.

Writer backend is swappable via `DEMO_BACKEND` (`mlx` / `trl` + `DEMO_ADAPTER`, or `openrouter`).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add OPENROUTER_API_KEY
```

**Demo** (committed data only):

```bash
pip install -r requirements-demo.txt
uvicorn src.demo_server:app --port 8000   # → http://127.0.0.1:8000
```

**Reproduce findings** from committed artifacts (`data/study_products.jsonl`, `data/panels.json`, `data/variants.jsonl`, `data/rankings_a.jsonl`):

```bash
python -m src.score_variants --rankings data/rankings_a.jsonl
```

**Full pipeline** (from source):

```bash
python -m src.download_data --which meta     # 1.38 GB → .hf_cache
python -m src.extract_drinks                 # 603,274 items → 27,580 drinks
python -m src.build_panels                   # → 25 panels × 8 products

python -m src.generate_variants              # 3,000 descriptions, ~15 min
python -m src.run_ranking --n-trials 12000 --max-minutes 60 \
    --panels data/half_a/panels.json \
    --variants data/half_a/variants.jsonl \
    --out data/rankings_a.jsonl
python -m src.score_variants --rankings data/rankings_a.jsonl

python -m src.build_finetune_data
python -m src.finetune --stage sft
python -m src.evaluate
```

Smoke-test first (`--limit 8`, `--n-trials 200 --limit-panels 3`). `run_ranking` is resumable by trial id and stops on `--max-minutes`. ~$20 / 24,000 trials on `gpt-4.1-mini`; rate limits bind before cost.

```bash
pytest   # 35 tests: trial construction, scoring, grounding, splits, demo
```

### Layout

```
src/
  config.py                # study parameters, arms, intents
  download_data.py         # Hugging Face → .hf_cache
  extract_drinks.py        # Amazon grocery metadata → drink catalogue
  build_panels.py          # catalogue → price-blocked panels
  llm.py                   # async OpenRouter: concurrency, retries, JSON repair
  generate_variants.py     # product × intent × arm → grounded, length-pinned copy
  run_ranking.py           # resumable ranking trials
  score_variants.py        # mean normalised rank with shrinkage
  build_finetune_data.py   # SFT + DPO sets, product-level holdout
  finetune.py              # Qwen3-4B LoRA (unsloth / TRL / mlx)
  evaluate.py              # fine-tune vs prompt-only vs original
  demo_server.py           # head-to-head demo (FastAPI + SSE)
demo/                      # demo page
data/                      # committed products, panels, variants, half-A rankings
tests/                     # pytest
```

`PLAN.md` is the working day-of spec. Training deps: `requirements-train.txt` (lazy imports; Unsloth is CUDA-only).

---

## Data

[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), `raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl` — 603,274 grocery items (CC, research use).

Classification uses Amazon’s category tree (level 2: `Beverages` vs `Alcoholic Beverages`); keyword fallback only for ~68k uncategorised items. De-alcoholised drinks kept; equipment, mixes, concentrates filtered. 27,580 drinks survive; study uses **200 Fruit Juice** products (≥50 ratings, description ≥120 characters).

All evaluated copy is real listing text or generated strictly from it.

---

## Limitations

| Limitation | Implication |
| --- | --- |
| Single ranker (`openai/gpt-4.1-mini`) | Transfer across Claude, Gemini, open-weights is untested |
| One category / price band | Fruit juice $7–$40; pipeline is category-agnostic, finding is not yet |
| Stated vs revealed intent | Three intents are authored prompts, not mined search logs |
| Optimisation can be gamed | Grounding is regex + system prompt; production needs claim verification and policy |
| Measurement cost | ~$20/hour per 24k trials; the model transfers — validate cross-category generalisation |
| Privacy | Public listing metadata + synthetic agent behaviour only; real session data would change that |
