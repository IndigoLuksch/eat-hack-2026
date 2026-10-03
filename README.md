# EAT_HACK 2026 — Intent-conditioned description optimiser

Fine-tune a model to write non-alcoholic drink product descriptions that AI
shopping agents rank highly, conditioned on a stated shopper intent.

**[PLAN.md](PLAN.md) is the implementation spec.** It carries the parameters,
data schemas, module-by-module instructions, runbook and open decisions. Start
there.

## How it works

An agent is shown a shortlist of 8 comparable fruit juices under one shopper
intent. Each product carries one of 6 randomly assigned description variants.
The agent returns a full ranking. Repeated ~24,000 times, this recovers which
variant wins for each product and intent, and those winners become the
fine-tuning set.

Products inside a shortlist are deliberately near-identical — same category,
median price ratio 1.05× — so the description has room to move the ranking.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add OPENROUTER_API_KEY
```

`data/study_products.jsonl` (200 products) and `data/panels.json` (25 panels)
are committed, so the API stages run without the 1.38 GB source download.

## Pipeline

```bash
# Rebuild the catalogue from source — only if the committed artifacts are gone
python -m src.download_data --which meta     # 1.38 GB → .hf_cache
python -m src.extract_drinks                 # 603k items → 27,580 drinks
python -m src.build_panels                   # → 25 panels × 8 products

# API stages, via OpenRouter
python -m src.generate_variants              # 3,000 descriptions
python -m src.run_ranking --n-trials 24000   # resumable, stop any time
python -m src.score_variants                 # mean normalised rank + shrinkage
python -m src.build_finetune_data            # SFT + DPO sets
python -m src.finetune --stage sft           # Qwen3-4B + LoRA via TRL
python -m src.evaluate                       # vs original and prompt-only baseline
```

Smoke-test the chain at tiny scale (`--limit 8`, `--n-trials 200`) before the
main run.

## Layout

```
src/
  config.py             # study parameters, variant arms, intents
  download_data.py      # Hugging Face → .hf_cache
  extract_drinks.py     # Amazon grocery metadata → drink catalogue
  build_panels.py       # catalogue → blocked shortlist panels
  llm.py                # async OpenRouter caller: concurrency, retries, JSON repair
  generate_variants.py  # product × intent × arm → description
data/
  study_products.jsonl  # committed: the 200 study products
  panels.json           # committed: the 25 shortlist panels
```

Still to build: `run_ranking.py`, `score_variants.py`, `build_finetune_data.py`,
`finetune.py`, `evaluate.py`. See [PLAN.md](PLAN.md) §5–8.

## Source data

[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
`raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl`. Drinks are
classified from Amazon's category tree rather than keyword matching — level 2
cleanly separates `Beverages` (95,811) from `Alcoholic Beverages` (2,682).
De-alcoholised drinks are deliberately kept.
