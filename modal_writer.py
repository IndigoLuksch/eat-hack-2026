"""Serve the BotBait fine-tune on Modal.

Deploy once:

    modal deploy modal_writer.py

Then set DEMO_BACKEND=modal and DEMO_MODAL_URL to the printed HTTPS URL.
"""

from __future__ import annotations

import modal
from pydantic import BaseModel, Field

MODEL_ID = "lollygag/qwen3-4b-juice-descriptions"
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


@app.cls(
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: volume},
)
class Writer:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        import re

        import torch

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        if not text:
            raise RuntimeError("model returned an empty description")
        return text

    @modal.fastapi_endpoint(method="POST")
    def generate(self, body: GenerateRequest) -> dict[str, str]:
        """POST {"messages": [...], "max_new_tokens": 180, "temperature": 0.7}."""
        return {
            "text": self._generate(body.messages, body.max_new_tokens, body.temperature)
        }
