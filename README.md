# Shelf Shift — what AI shopping agents actually reward in product copy

**EAT_HACK 2026 · Track 1: Human Truth**

When an AI agent shops on your behalf, it never sees the packaging, the end cap
or the photography. It reads the product description. So the description stops
being marketing and starts being the entire sales surface — and nobody knows
what it rewards.

We ran a randomised controlled experiment on that question: **10,854 ranking
trials** across 104 real Amazon fruit juices, then used the result to train a
model to write copy that places higher on an agent's shortlist.

---

## What we found

The untouched Amazon description finished **last of six arms on all three
shopper intents.** Not close to last — last, with no overlapping confidence
intervals.

| Shopper intent | Best rhetorical form | Mean normalised rank | `original` | Products where rewriting won |
| --- | --- | --- | --- | --- |
| Flavour | `sensory` | 0.448 `[0.439, 0.457]` | 0.586 `[0.576, 0.595]` | 97 / 104 (93%) |
| General | `sensory` | 0.459 `[0.450, 0.468]` | 0.575 `[0.566, 0.584]` | 97 / 104 (93%) |
| Health | `assurance` | 0.482 `[0.472, 0.491]` | 0.547 `[0.537, 0.557]` | 94 / 104 (90%) |

Normalised rank runs 0 (top of the shortlist) to 1 (bottom), so lower is
better. On an 8-product shortlist, the flavour gap of 0.138 is worth **about
one full place**, from a rewrite that adds no new facts.

Two things make this more than "LLMs like better copy":

**The winning form depends on the intent, and the ordering inverts.** `sensory`
is first for flavour and general, but *fourth* for health. `assurance` — brand
provenance and certifications already present in the listing — is first for
health and fourth for flavour. An agent asked for a healthy drink and an agent
asked for a tasty drink are not reading the same text in the same way, even
when the product is identical.

**It is not one style for everyone.** Even on flavour, where `sensory` wins
the leaderboard, it is the best arm for only 46 of 104 products — and for just
9 of 104 on health. The per-product pattern is what the model learns; the
leaderboard is only the average.

---

## The experiment

The hard part is attribution. Real shortlists differ in price, brand, rating
and quality all at once, so an observed ranking says nothing about the copy. We
removed every competing explanation:

- **Within-product comparison.** The same product appears thousands of times
  carrying different descriptions. Each product is its own control, so brand
  equity and intrinsic quality cancel out.
- **Near-identical competitors.** Panels of 8 products from one leaf category
  (Fruit Juice) in one price band ($7–$40), median within-panel price ratio
  **1.05×**. The copy has room to move the ranking because nothing else does.
- **Length is pinned.** Every generated arm is held to 70 words ±10%, rejected
  and regenerated otherwise. Length is the single strongest driver of LLM
  preference and would otherwise swamp the style effect entirely.
- **Only the description is shown.** Title, brand, price, rating and review
  count are stripped from the option cards. The agent ranks on copy alone.
- **Display order reshuffled every trial.** Position bias in list ranking is
  large and is not otherwise controlled.
- **Arms assigned at random per trial**, seeded from the trial id so any trial
  can be reconstructed exactly.

Six arms per product × intent. `original` is the untouched Amazon copy and is
deliberately *not* length-matched — it is the real-world baseline, not a style.
The other five hold the intent topic fixed and vary only rhetorical form:
`direct`, `sensory`, `quantified`, `use_case`, `assurance`.

Generated copy is **grounded in the listing**: numbers that do not appear in
the source are rejected programmatically, along with markdown, alcohol
mentions and anything outside the word band. No invented certifications, no
invented health claims.

Scoring is mean normalised rank with partial pooling — each cell is shrunk
toward its intent-by-arm pooled mean with a 25-observation prior, so a product
seen 12 times does not outvote one seen 60. Median observations per cell: 48.

The 25 panels are split into two halves. Half A (13 panels, 104 products) is
the measurement set reported above. Half B (12 panels, 96 products) is held
back, and 10% of products are held out by `parent_asin` before any training
set is built, so a product cannot leak across the split through its other
intent.

---

## The product

Measuring the effect is step one. The deliverable is a **description writer
conditioned on shopper intent**, trained on the arms that actually won.

Each product × intent cell contributes its best arm as a supervised target,
plus its second-best only when that arm also beats the untouched Amazon copy.
DPO pairs are best-versus-worst and second-versus-fifth, gated on a minimum
score gap so the model is not trained on noise. The base model is
`Qwen/Qwen3-4B-Instruct-2507` with LoRA.

The point of fine-tuning rather than prompting: the per-product winner is not
reducible to a style guide. `evaluate.py` runs the fine-tune against a
**prompt-only baseline** — a strong model told in plain English what the
winning style was — on held-out products. If a prompt matches the fine-tune,
the fine-tune added nothing, and we would rather know.

### The demo: can you beat it?

`demo/` is a head-to-head. You are shown a real product, its photo, its bullet
features and one shopper's request. You write a description. The model writes
one from the same facts.

Both go into the same panel against the same seven competitors — carrying
their real Amazon copy — and are ranked five times with shared display orders,
so the only thing that differs between your run and the model's is your
sentence. You get both average ranks with error bars, and who won each round.

It is the experiment, played as a game. It also makes the finding immediate in
a way a table does not: most people write something that loses to the model,
and then want to know why.

> The demo's writer is swappable via `DEMO_BACKEND`. Set it to `mlx` or `trl`
> with `DEMO_ADAPTER` pointing at the trained adapter; `openrouter` runs a
> hosted model against the identical training prompt.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add OPENROUTER_API_KEY
```

**The demo**, which needs only the committed data:

```bash
pip install -r requirements-demo.txt
uvicorn src.demo_server:app --port 8000   # → http://127.0.0.1:8000
```

**The pipeline.** `data/study_products.jsonl`, `data/panels.json`,
`data/variants.jsonl` and `data/rankings_a.jsonl` are committed, so you can
re-derive the findings without the 1.38 GB source download or any API spend:

```bash
python -m src.score_variants --rankings data/rankings_a.jsonl
```

To run it from scratch:

```bash
# Rebuild the catalogue from source — only if the committed artifacts are gone
python -m src.download_data --which meta     # 1.38 GB → .hf_cache
python -m src.extract_drinks                 # 603,274 items → 27,580 drinks
python -m src.build_panels                   # → 25 panels × 8 products

# API stages, via OpenRouter
python -m src.generate_variants                             # 3,000 descriptions, ~15 min
python -m src.run_ranking --n-trials 12000 --max-minutes 60 \
    --panels data/half_a/panels.json \
    --variants data/half_a/variants.jsonl \
    --out data/rankings_a.jsonl                             # resumable, stop any time
python -m src.score_variants --rankings data/rankings_a.jsonl

# Training
python -m src.build_finetune_data
python -m src.finetune --stage sft
python -m src.evaluate
```

Smoke-test the chain at tiny scale (`--limit 8`, `--n-trials 200
--limit-panels 3`) before a main run. Discovering a JSON parsing bug at trial
10,000 is the failure mode that costs the day.

`run_ranking` is resumable by trial id and stops cleanly on `--max-minutes`, so
the budget is a clock rather than a target. 24,000 ranking calls on
`gpt-4.1-mini` is roughly $20; rate limits, not cost, are the binding
constraint.

### Layout

```
src/
  config.py                # study parameters, the six arms, the three intents
  download_data.py         # Hugging Face → .hf_cache
  extract_drinks.py        # Amazon grocery metadata → drink catalogue
  build_panels.py          # catalogue → price-blocked shortlist panels
  llm.py                   # async OpenRouter caller: concurrency, retries, JSON repair
  generate_variants.py     # product × intent × arm → grounded, length-pinned copy
  run_ranking.py           # resumable shopping-agent ranking trials
  score_variants.py        # mean normalised rank with shrinkage
  build_finetune_data.py   # SFT + DPO sets, product-level holdout
  finetune.py              # Qwen3-4B LoRA via unsloth, TRL, or mlx
  evaluate.py              # fine-tune vs prompt-only baseline vs original
  demo_server.py           # the head-to-head demo, FastAPI + SSE
demo/                      # the demo page
data/                      # committed: products, panels, variants, half-A rankings
tests/                     # pytest: trial construction, scoring, demo
```

`PLAN.md` is the working spec we built from on the day — study sizing, schemas
and open risks. It is kept as written rather than tidied up after the fact.

Training dependencies are in `requirements-train.txt` and imported lazily, only
when a backend is selected, so the module loads on a machine with none of them.
Unsloth is CUDA-only.

35 tests cover trial construction and determinism, ranking validation, the
shrinkage estimator, the grounding and length checks, dataset splitting and the
demo's pairing logic:

```bash
pytest
```

---

## Data

[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
`raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl` — 603,274 grocery
items, CC-licensed for research use.

Drinks are classified from Amazon's own category tree rather than by keyword
matching: level 2 of the taxonomy cleanly separates `Beverages` (95,811) from
`Alcoholic Beverages` (2,682), with keyword fallback only for the ~68k items
that carry no categories at all. De-alcoholised drinks are deliberately kept in
scope; equipment, mixes and concentrates are filtered out. 27,580 drinks
survive, and the study uses 200 Fruit Juice products with at least 50 ratings
and a description of at least 120 characters.

All copy evaluated here is real listing copy or generated strictly from it. No
synthetic products, no synthetic shoppers.

---

## Limitations, and what it would take to make this real

**One agent is not all agents.** Every ranking here comes from
`openai/gpt-4.1-mini`. The effect is large and consistent, but we have not
shown it transfers across model families, and that is the first thing we would
test: the same panels against Claude, Gemini and an open-weights ranker. If the
winning form differs by model, the product becomes "optimise for the agent your
customer uses", which is a more interesting product, not a worse one.

**One category, one price band.** Fruit juice at $7–$40 was chosen to make the
measurement clean. Whether `sensory` wins in household cleaning is an open
question. The pipeline is category-agnostic; the finding is not yet.

**Stated intent is not revealed intent.** Our three intents are prompts we
wrote, not observed shopper language. Real deployment would mine them from
search logs.

**This is an optimisation target, and optimisation targets get gamed.** We
constrain generation to facts in the listing and reject invented numbers
mechanically, but "truthful and grounded" is enforced by a regex and a system
prompt, not a guarantee. A production version needs claim-level verification
against a product data feed, and a retailer deploying this needs a policy on
what counts as legitimate optimisation versus manipulating a buyer's agent.
That question is going to matter commercially long before it is settled.

**Scale and cost.** Measurement is the expensive part: ~$20 and an hour of
wall-clock per 24,000 trials, which covers 200 products. For a 50,000-SKU
retailer that is the wrong shape. The fix is that the *model* is the
transferable asset — you pay for measurement once per category, then write for
free. Validating how far one category's model generalises is what decides
whether this is a product or a consultancy.

**Privacy and data ownership.** Nothing here touches personal data: it is
public listing metadata plus synthetic agent behaviour. A version trained on a
retailer's real sessions would not have that property, and the honest version
of this product keeps the measurement loop agent-only for exactly that reason.
