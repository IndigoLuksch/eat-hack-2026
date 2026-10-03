"""The demo page and the ranking API.

The process binds its port before ranking starts. BotBait is an OpenRouter
model with the style prompt, so it does not wait on a local weight load.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.build_finetune_data import botbait_messages, prompt_messages
from src.config import BASELINE_MODEL, BOTBAIT_MODEL, INTENTS, RANK_MODEL, ROOT
from src.demo_catalog import Catalog, load_catalog
from src.demo_model import _ModalGenerator, _OpenRouterGenerator
from src.demo_rank import (
    N_REPS,
    SOURCES,
    build_cards,
    place,
    shared_orders,
    summarise_places,
    target_option_id,
)
from src.llm import Caller
from src.run_ranking import RANK_SYSTEM, parse_ranking, rank_user

log = logging.getLogger("demo")

DEMO_DIR = ROOT / "demo"
MAX_WORDS = 400
RANK_ATTEMPTS = 3
# Some proxies buffer the first kilobytes of a stream. A comment flushes them.
_PADDING = ":" + (" " * 2048) + "\n\n"

catalog: Catalog
model: Any
qwen: Any
baseline: _OpenRouterGenerator


def _qwen_writer() -> Any:
    """Prefer the Modal-hosted Qwen3-4B base; fall back to OpenRouter."""
    import os

    url = os.getenv("DEMO_MODAL_QWEN_URL", "").strip()
    if url:
        return _ModalGenerator(url, os.getenv("DEMO_MODAL_TOKEN", ""))
    from src.config import BASE_QWEN_MODEL

    return _OpenRouterGenerator(
        BASE_QWEN_MODEL,
        extra_body={"reasoning": {"effort": "none", "exclude": True}},
    )


class _ReadyWriter:
    """BotBait writer. Ready immediately; no local model load."""

    def __init__(self, generator: _OpenRouterGenerator) -> None:
        self._generator = generator
        self.error: Optional[str] = None

    def wait_ready(self, timeout: float) -> bool:
        return True

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 180,
        temperature: float = 0.7,
    ) -> str:
        text = self._generator.generate(messages, max_new_tokens, temperature)
        if not str(text).strip():
            raise RuntimeError("model returned an empty description")
        return str(text).strip()


def _start_model() -> None:
    global catalog, model, qwen, baseline
    catalog = load_catalog()
    # BotBait is the prompted OpenRouter model, not the Modal fine-tune.
    model = _ReadyWriter(_OpenRouterGenerator(BOTBAIT_MODEL))
    # Same OpenRouter key the ranker needs; construct here so a missing key
    # fails before the first visitor submits.
    qwen = _qwen_writer()
    baseline = _OpenRouterGenerator(BASELINE_MODEL)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _start_model()
    yield


app = FastAPI(lifespan=_lifespan)


def _intents() -> list[dict[str, str]]:
    return [
        {"key": key, "label": spec["label"], "shopper": spec["shopper"]}
        for key, spec in INTENTS.items()
    ]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@app.get("/style.css")
def css() -> FileResponse:
    return FileResponse(DEMO_DIR / "style.css", media_type="text/css")


@app.get("/api/case")
def case(exclude: Optional[str] = None) -> dict[str, Any]:
    chosen = catalog.choose(random.Random(), exclude or None)
    return {**chosen, "intents": _intents()}


class RankBody(BaseModel):
    parent_asin: str
    intent: str
    text: str = Field(min_length=1)


async def rank_cards(caller: Caller, intent_key: str, cards: list[dict[str, Any]]) -> list[str]:
    """One shopping-agent ranking, using the training prompt and parser."""
    user = rank_user(INTENTS[intent_key]["shopper"], cards)
    option_ids = [card["option_id"] for card in cards]
    prompt = user
    last = "ranking failed"
    for _ in range(RANK_ATTEMPTS):
        try:
            payload = await caller.json(RANK_SYSTEM, prompt)
        except Exception as exc:  # noqa: BLE001 - the caller already retried transport errors
            last = str(exc)
            continue
        ranking = parse_ranking(payload, option_ids)
        if ranking is None:
            last = "ranking is not a permutation of the option ids"
            prompt = (
                f"{user}\n"
                "The previous reply was rejected. Return JSON whose ranking "
                f"contains each of these ids exactly once, best first: {', '.join(option_ids)}."
            )
            continue
        return ranking
    raise RuntimeError(last)


@app.post("/api/rank")
async def rank(body: RankBody) -> StreamingResponse:
    text = " ".join(body.text.split())
    if body.intent not in INTENTS:
        raise HTTPException(status_code=400, detail="Unknown intent.")
    if not text or len(text.split()) > MAX_WORDS:
        raise HTTPException(status_code=400, detail="Description length is off.")
    if body.parent_asin not in catalog.panel_of:
        raise HTTPException(status_code=404, detail="Unknown product.")

    async def events() -> AsyncIterator[str]:
        yield _PADDING
        yield _sse({"phase": "writing"})
        writers: list[asyncio.Task] = []
        tasks: list[asyncio.Task] = []
        try:
            while not model.wait_ready(2):
                yield _sse({"phase": "writing"})
            if model.error:
                log.error("model unavailable: %s", model.error)
                yield _sse({"phase": "error", "message": "Model is not available."})
                return

            product = catalog.products[body.parent_asin]
            plain = prompt_messages(product, body.intent)
            writers = [
                asyncio.create_task(asyncio.to_thread(model.generate, botbait_messages(product, body.intent))),
                asyncio.create_task(asyncio.to_thread(qwen.generate, plain)),
                asyncio.create_task(asyncio.to_thread(baseline.generate, plain)),
            ]
            pending_writers = set(writers)
            while pending_writers:
                finished, pending_writers = await asyncio.wait(
                    pending_writers,
                    timeout=2,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not finished:
                    yield _sse({"phase": "writing"})
            texts_by_writer: dict[str, str] = {}
            for name, task in zip(("model", "qwen", "opus"), writers):
                try:
                    texts_by_writer[name] = task.result()
                except Exception:
                    log.exception("description failed for %s", name)
                    yield _sse({"phase": "error", "message": "Could not write a description."})
                    return
            model_text = texts_by_writer["model"]
            qwen_text = texts_by_writer["qwen"]
            opus_text = texts_by_writer["opus"]

            asins = catalog.panel_asins(body.parent_asin)
            orders = shared_orders(asins, N_REPS, random.Random())
            caller = Caller(
                model=RANK_MODEL,
                concurrency=N_REPS * len(SOURCES),
                temperature=0.0,
                max_tokens=300,
            )
            yield _sse({"phase": "ranking", "done": 0, "total": N_REPS})

            places: dict[str, list[Optional[int]]] = {
                source: [None] * N_REPS for source in SOURCES
            }
            texts = {
                "user": text,
                "model": model_text,
                "qwen": qwen_text,
                "opus": opus_text,
            }

            def runs_done() -> int:
                return sum(
                    1
                    for index in range(N_REPS)
                    if all(places[source][index] is not None for source in SOURCES)
                )

            async def one(rep: int, source: str, description: str, order: list[str]) -> None:
                cards = build_cards(order, catalog.products, body.parent_asin, description)
                ranking = await rank_cards(caller, body.intent, cards)
                places[source][rep] = place(
                    ranking, target_option_id(cards, body.parent_asin)
                )

            for rep, order in enumerate(orders):
                for source in SOURCES:
                    tasks.append(
                        asyncio.create_task(one(rep, source, texts[source], order))
                    )

            pending = set(tasks)
            while pending:
                finished, pending = await asyncio.wait(
                    pending,
                    timeout=2,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not finished:
                    yield _sse({"phase": "ranking", "done": runs_done(), "total": N_REPS})
                    continue
                for task in finished:
                    task.result()
                yield _sse({"phase": "ranking", "done": runs_done(), "total": N_REPS})

            summary = summarise_places(
                {source: [int(slot) for slot in places[source]] for source in SOURCES}
            )
            for source in SOURCES:
                summary[source]["text"] = texts[source]
            yield _sse({"phase": "done", "result": summary})
        except Exception:
            log.exception("rank failed")
            yield _sse({"phase": "error", "message": "Ranking failed."})
        finally:
            for task in writers:
                if not task.done():
                    task.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
