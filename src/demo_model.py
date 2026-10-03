"""Keep one description writer resident for the demo.

`openrouter` calls a cheap hosted model. That is the stand-in while the
fine-tune is still training. `mlx` is the Apple Silicon path. `trl` loads a
PEFT adapter with transformers, which is what a Linux box can run. The adapter
is a local directory or a Hugging Face repo id, downloaded on first use.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from src.config import BASE_MODEL, OPENROUTER_BASE_URL, VARIANT_MODEL
from src.llm import _HEADERS

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_adapter(spec: str) -> Path:
    """A local directory, or a Hugging Face repo id downloaded to the cache."""
    text = spec.strip()
    if not text:
        raise RuntimeError("Set DEMO_ADAPTER to a local directory or a Hugging Face repo id.")
    path = Path(text).expanduser()
    if path.exists():
        return path
    if not _REPO_ID.match(text):
        raise FileNotFoundError(f"Adapter not found at {path}.")
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    return Path(snapshot_download(repo_id=text, token=token))


class _PeftGenerator:
    def __init__(self, base_model: str, adapter: Path) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_dir = adapter if (adapter / "tokenizer_config.json").exists() else Path(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device = "cuda"
        else:
            # float16 keeps Qwen3-4B inside a 16 GB instance. float32 does not.
            dtype = torch.float16
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.to(device).eval()
        self.device = device

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class _MlxGenerator:
    def __init__(self, base_model: str, adapter: Path) -> None:
        from mlx_lm import load

        self.model, self.tokenizer = load(base_model, adapter_path=str(adapter))

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
                temp=temperature,
            )
        except TypeError:
            text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )
        if isinstance(text, str) and text.startswith(prompt):
            text = text[len(prompt) :]
        return str(text).strip()


class _OpenRouterGenerator:
    """Same training prompt as the fine-tune, answered by a hosted model."""

    def __init__(self, model: str, client: Any = None) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        from openai import OpenAI

        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key, default_headers=_HEADERS)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 180,
        temperature: float = 0.7,
    ) -> str:
        prompt = [dict(message) for message in messages]
        if prompt and prompt[-1].get("role") == "user":
            prompt[-1]["content"] = (
                f"{prompt[-1]['content']}\n"
                "Write about 70 words of flowing prose. No headings, markdown, or bullet characters."
            )
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max(max_new_tokens, 320),
                    messages=prompt,
                )
                content = resp.choices[0].message.content or ""
                text = content.strip()
                if text:
                    return text
                raise RuntimeError("empty completion")
            except Exception as exc:  # noqa: BLE001 - one transient failure should not end the demo
                last = exc
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"description request failed: {last}")


class ResidentModel:
    """Loads in the background. Requests wait until `ready` is set."""

    def __init__(self, base_model: str = BASE_MODEL) -> None:
        self.base_model = base_model
        self.error: Optional[str] = None
        self._generator: Any = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def load(self) -> None:
        try:
            backend = os.getenv("DEMO_BACKEND", "openrouter").strip() or "openrouter"
            if backend == "openrouter":
                model_name = os.getenv("DEMO_MODEL", "").strip() or VARIANT_MODEL
                self._generator = _OpenRouterGenerator(model_name)
                return
            spec = os.getenv("DEMO_ADAPTER", "").strip()
            if backend not in {"mlx", "trl"}:
                raise RuntimeError("Set DEMO_BACKEND to openrouter, mlx, or trl.")
            if not spec:
                raise RuntimeError("Set DEMO_ADAPTER to a local directory or a Hugging Face repo id.")
            adapter = resolve_adapter(spec)
            if backend == "mlx":
                self._generator = _MlxGenerator(self.base_model, adapter)
            else:
                self._generator = _PeftGenerator(self.base_model, adapter)
        except Exception as exc:  # noqa: BLE001 - surface a single message to the page
            self.error = str(exc)
        finally:
            self._ready.set()

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 180,
        temperature: float = 0.7,
    ) -> str:
        if not self.ready:
            raise RuntimeError("model is still loading")
        if self.error or self._generator is None:
            raise RuntimeError(self.error or "model failed to load")
        with self._lock:
            text = self._generator.generate(messages, max_new_tokens, temperature)
        if not str(text).strip():
            raise RuntimeError("model returned an empty description")
        return str(text).strip()
