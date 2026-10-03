"""Concurrent OpenRouter client shared by variant generation and ranking."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable, Iterable, Optional, TypeVar

from openai import AsyncOpenAI
from tqdm import tqdm

from src.config import (
    CONCURRENCY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)

T = TypeVar("T")

# OpenRouter asks callers to identify themselves; harmless if unset.
_HEADERS = {
    "HTTP-Referer": "https://github.com/eat-hack/description-optimiser",
    "X-Title": "EAT_HACK description optimiser",
}


def make_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, default_headers=_HEADERS)


def extract_json(text: str) -> dict[str, Any]:
    """Models wrap JSON in prose or fences often enough to be worth handling."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))


class Caller:
    """Rate-limited chat caller with bounded retries.

    Retries cover transient 429s and malformed JSON alike; both are common
    enough over tens of thousands of calls that an unretried run will not
    finish cleanly.
    """

    def __init__(
        self,
        model: str,
        concurrency: int = CONCURRENCY,
        temperature: float = 0.7,
        max_tokens: int = 700,
        max_attempts: int = 5,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        # Built on first use, inside the running loop. Creating the HTTP client
        # or the semaphore beforehand binds them to a different loop on Python 3.9.
        self._api_key = api_key
        self._concurrency = concurrency
        self.client: Optional[AsyncOpenAI] = None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.reasoning_effort = reasoning_effort
        self._sem: Optional[asyncio.Semaphore] = None

    def _bind(self) -> asyncio.Semaphore:
        if self.client is None:
            self.client = make_client(self._api_key)
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._concurrency)
        return self._sem

    def _request_kwargs(self) -> dict[str, Any]:
        if not self.reasoning_effort:
            return {}
        # Exclude the trace so the completion is the JSON we asked for.
        return {
            "extra_body": {
                "reasoning": {"effort": self.reasoning_effort, "exclude": True},
            }
        }

    async def text(self, system: str, user: str) -> str:
        async with self._bind():
            last: Exception | None = None
            for attempt in range(self.max_attempts):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        **self._request_kwargs(),
                    )
                    content = resp.choices[0].message.content
                    if not content:
                        raise ValueError("empty completion")
                    return content
                except Exception as exc:  # noqa: BLE001 - retry on anything transient
                    last = exc
                    await asyncio.sleep(min(2**attempt, 30))
            raise RuntimeError(f"all {self.max_attempts} attempts failed: {last}")

    async def json(self, system: str, user: str) -> dict[str, Any]:
        async with self._bind():
            last: Exception | None = None
            for attempt in range(self.max_attempts):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        **self._request_kwargs(),
                    )
                    content = resp.choices[0].message.content
                    if not content:
                        raise ValueError("empty completion")
                    return extract_json(content)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    await asyncio.sleep(min(2**attempt, 30))
            raise RuntimeError(f"all {self.max_attempts} attempts failed: {last}")


async def gather_with_progress(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[Any]],
    desc: str,
    on_result: Optional[Callable[[Any], None]] = None,
) -> list[Any]:
    """Run `worker` over `items`, streaming results to `on_result` as they land.

    Results are handled on completion rather than at the end so a long run can
    checkpoint to disk and be resumed after an interruption.
    """
    items = list(items)
    results: list[Any] = []
    bar = tqdm(total=len(items), desc=desc)

    async def wrapped(item: T) -> None:
        try:
            out = await worker(item)
        except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
            bar.write(f"  skipped: {exc}")
            out = None
        if out is not None:
            results.append(out)
            if on_result is not None:
                on_result(out)
        bar.update(1)

    await asyncio.gather(*(wrapped(i) for i in items))
    bar.close()
    return results


async def run_bounded(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[Any]],
    *,
    concurrency: int,
    desc: str,
    deadline: Optional[float] = None,
    on_result: Optional[Callable[[Any], None]] = None,
) -> tuple[list[Any], int]:
    """Run `worker` with at most `concurrency` calls in flight.

    `deadline` is a `time.monotonic()` timestamp. A call already in flight is
    allowed to finish; nothing new starts after the deadline. Returns the
    collected results and how many items never started.
    """
    queue: asyncio.Queue[T] = asyncio.Queue()
    pending = list(items)
    for item in pending:
        queue.put_nowait(item)
    results: list[Any] = []
    bar = tqdm(total=len(pending), desc=desc)

    async def consume() -> None:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                return
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                out = await worker(item)
            except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
                bar.write(f"  skipped: {exc}")
                out = None
            if out is not None:
                results.append(out)
                if on_result is not None:
                    on_result(out)
            bar.update(1)

    await asyncio.gather(*(consume() for _ in range(max(1, concurrency))))
    left = queue.qsize()
    bar.close()
    return results, left
