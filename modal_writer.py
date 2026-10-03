"""Serve BotBait and the untuned Qwen3-4B base on Modal.

Deploy once:

    modal deploy modal_writer.py

Then set:
  DEMO_BACKEND=modal
  DEMO_MODAL_URL=<Writer.generate URL>
  DEMO_MODAL_QWEN_URL=<BaseQwen.generate URL>
"""

from __future__ import annotations

import modal
from pydantic import BaseModel, Field

FINETUNE_ID = "lollygag/qwen3-4b-juice-descriptions"
BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"
CACHE_DIR = "/cache/huggingface"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "huggingface_hub>=0.23.0",
        "fastapi[standard]>=0.115.0",
        "pydantic>=2.0",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_CACHE": CACHE_DIR})
)

volume = modal.Volume.from_name("botbait-hf-cache", create_if_missing=True)
app = modal.App("botbait-writer", image=image)


class GenerateRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    max_new_tokens: int = 180
    temperature: float = 0.7


def _load_causal_lm(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return tokenizer, model


def _generate_text(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
    import re

    import torch

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        raise RuntimeError("model returned an empty description")
    return text


@app.cls(
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: volume},
)
class Writer:
    """Merged BotBait fine-tune."""

    @modal.enter()
    def load(self) -> None:
        self.tokenizer, self.model = _load_causal_lm(FINETUNE_ID)

    @modal.fastapi_endpoint(method="POST")
    def generate(self, body: GenerateRequest) -> dict[str, str]:
        return {
            "text": _generate_text(
                self.tokenizer,
                self.model,
                body.messages,
                body.max_new_tokens,
                body.temperature,
            )
        }


@app.cls(
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: volume},
)
class BaseQwen:
    """Untuned Qwen3-4B-Instruct — the fine-tune's base model."""

    @modal.enter()
    def load(self) -> None:
        self.tokenizer, self.model = _load_causal_lm(BASE_ID)

    @modal.fastapi_endpoint(method="POST")
    def generate(self, body: GenerateRequest) -> dict[str, str]:
        return {
            "text": _generate_text(
                self.tokenizer,
                self.model,
                body.messages,
                body.max_new_tokens,
                body.temperature,
            )
        }
