from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HF_CACHE_DIR = ROOT / os.getenv("HF_HOME", ".hf_cache")
HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
META_JSONL = "raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl"
REVIEWS_JSONL = "raw/review_categories/Grocery_and_Gourmet_Food.jsonl"

# --- OpenRouter -------------------------------------------------------------
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
VARIANT_MODEL = os.getenv("VARIANT_MODEL", "anthropic/claude-3.5-sonnet")
RANK_MODEL = os.getenv("RANK_MODEL", "openai/gpt-4o-mini")
CONCURRENCY = int(os.getenv("CONCURRENCY", "16"))

# --- Study parameters -------------------------------------------------------
# Deliberately narrow: one leaf category inside one price band. Narrowing does
# not improve the estimator (it is already within-product), but it concentrates
# a fixed trial budget onto fewer cells, which raises per-product precision.
STUDY_CATEGORY = "Fruit Juice"
PRICE_MIN = 7.0
PRICE_MAX = 40.0
PANEL_SIZE = 8  # products shown per shortlist
N_PANELS = 25  # 25 x 8 = 200 products
MAX_PRICE_RATIO = 2.5  # cap on max/min price inside one panel
MIN_RATINGS = 50
MIN_DESC_CHARS = 120
SHRINKAGE_PRIOR = 25  # pseudo-observations pulling a cell toward the pooled mean

# Amazon's leaf categories leak adjacent products; these are not drinks you
# would shortlist against a juice.
TITLE_EXCLUDE = (
    "powder",
    "drink mix",
    "concentrate",
    "syrup",
    "meal replacement",
    "glucose control",
    "baby",
    "infant",
    "formula",
    "puree",
    "capsule",
    "tablet",
    "supplement",
)

# Variant styles. "original" is the untouched control; "probe" is a deliberately
# weak arm used as a manipulation check — if it does not rank last, the
# experiment is not measuring copy at all.
VARIANT_STYLES: dict[str, str] = {
    "original": "",
    "sensory": (
        "Lead with honest sensory detail: flavour, aroma, carbonation, mouthfeel, "
        "how it tastes chilled. Vivid but never invented."
    ),
    "spec_dense": (
        "Lead with quantified, scannable specification: pack size, volume per unit, "
        "calories, sugar, caffeine, servings. Facts over adjectives."
    ),
    "constraint_match": (
        "Lead with explicit dietary and lifestyle flags a shopper might filter on: "
        "sugar-free, vegan, gluten-free, organic, non-GMO, caffeine-free, kosher. "
        "State only flags supported by the source text; say nothing about the rest."
    ),
    "occasion": (
        "Lead with concrete use cases and moments: workouts, lunchboxes, desk work, "
        "road trips, hosting. Make the fit to a situation obvious."
    ),
    "social_proof": (
        "Lead with popularity and reviewer consensus, using only the rating and "
        "review count supplied. No invented awards, rankings or endorsements."
    ),
    "probe": (
        "Write a deliberately vague, low-information description. Generic praise "
        "only, no concrete facts, no numbers, no dietary flags. This is a control "
        "arm and is expected to perform badly."
    ),
}

# Variants actually shown in the main run. The probe is pilot-only.
MAIN_RUN_STYLES = ("original", "sensory", "spec_dense", "constraint_match", "occasion", "social_proof")

TARGET_WORDS = 70  # all variants pinned to this length so length is not a confound

CUSTOMER_INTENTS: dict[str, str] = {
    "everyday_refresh": "I want a refreshing non-alcoholic drink for everyday use at home.",
    "low_sugar": "I'm cutting down on sugar. Find me something with little or no sugar.",
    "kids_family": "I need something my kids will actually drink, for school lunchboxes.",
    "afternoon_lift": "I want a natural pick-me-up to get through the afternoon slump.",
    "sports_hydration": "I want something to rehydrate with after a hard workout.",
    "clean_label": "I prefer organic, natural, clean-label drinks without artificial additives.",
}

# Amazon's grocery taxonomy is `Grocery & Gourmet Food > <level 2> > ...`.
# Level 2 is a reliable drink signal, so it drives classification; keyword
# matching is only a fallback for the ~68k items with no categories at all.
BEVERAGE_L2 = "beverages"
ALCOHOL_L2 = ("alcoholic beverages", "home brewing & winemaking")
# Plant/dairy milks live outside the Beverages branch but are drinks.
EXTRA_DRINK_PATHS = ("plant-based milk", "milk & cream > dairy milk")

# Ordered most-specific-first; first match wins.
DRINK_FORM_RULES = (
    (
        "ready_to_drink",
        (
            "juices",
            "soft drinks",
            "energy drinks",
            "sports drinks",
            "meal replacement & protein drinks",
            "iced tea",
            "kombucha",
            "iced coffee & cold-brew",
            "ready to drink liquid coffee",
            "plant-based milk",
            "dairy milk",
            "water",
        ),
    ),
    (
        "mix_or_concentrate",
        (
            "powdered drink mixes & flavorings",
            "syrups",
            "concentrates",
            "cocoa",
            "malted drinks",
            "hot chocolate",
            "cocktail mixers",
            "instant coffee",
            "milk tea mix",
            "bubble tea kits",
        ),
    ),
    (
        "brewing_ingredient",
        (
            "ground coffee",
            "whole coffee beans",
            "single-serve capsules & pods",
            "coffee substitutes",
            "tapioca pearls",
            "tea",
            "coffee",
        ),
    ),
)

DRINK_TITLE_HINTS = (
    "juice",
    "soda",
    "sparkling water",
    "mineral water",
    "bottled water",
    "iced tea",
    "green tea",
    "black tea",
    "herbal tea",
    "kombucha",
    "lemonade",
    "smoothie",
    "energy drink",
    "sports drink",
    "electrolyte",
    "cola",
    "ginger ale",
    "tonic water",
    "coconut water",
    "oat milk",
    "almond milk",
    "soy milk",
    "cold brew",
    "coffee drink",
    "matcha",
    "yerba mate",
    "fruit punch",
    "nectar",
)

# Matched on word boundaries against title + categories only. Terms that are
# ambiguous in drink names are deliberately absent: "ale" (ginger ale), "proof"
# (leak-proof), "ipa" and "sake" (too short / too common in prose).
ALCOHOL_EXCLUDE = (
    "wine",
    "beer",
    "lager",
    "stout",
    "cider",
    "champagne",
    "prosecco",
    "vodka",
    "whiskey",
    "whisky",
    "bourbon",
    "rum",
    "gin",
    "tequila",
    "liqueur",
    "brandy",
    "cognac",
    "mezcal",
    "hard seltzer",
    "alcoholic",
    "abv",
)

# Override the alcohol block: de-alcoholised wine/beer are in scope.
NON_ALCOHOLIC_MARKERS = (
    "non-alcoholic",
    "non alcoholic",
    "nonalcoholic",
    "alcohol-free",
    "alcohol free",
    "dealcoholized",
    "de-alcoholized",
    "zero proof",
    "zero-proof",
    "mocktail",
)

# Equipment and ingredients that are not themselves a drink.
NON_DRINK_EXCLUDE = (
    "mug",
    "tumbler",
    "travel cup",
    "machine",
    "maker",
    "brewer",
    "kettle",
    "filter",
    "filters",
    "frother",
    "grinder",
    "straw",
    "straws",
    "coaster",
    "dispenser",
    "water bottle",
    "gift basket",
)

# Applied only to items with no categories, where a flavour word in the title
# ("matcha candy", "smoothie bowl") is the sole drink signal.
NON_DRINK_FOOD_WORDS = (
    "candy",
    "candies",
    "bowl",
    "cookie",
    "cookies",
    "granola",
    "cereal",
    "oatmeal",
    "bar",
    "bars",
    "cake",
    "ice cream",
    "yogurt",
    "snack",
    "chips",
)
