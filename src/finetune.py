"""Fine-tune Qwen3-4B-Instruct on the winning descriptions.

The dataset split and the adapter directory are shared. Each backend is a
small function that receives the already formatted rows and returns the saved
adapter path. Backend libraries are imported inside those functions, so this
module imports on a machine that has none of them installed.

Unsloth is CUDA-only. On Apple Silicon it fails immediately and names the
backends that do run there.

Shared recipe, from the study plan:

    LoRA r=32, alpha=64, dropout=0.05, target_modules=all-linear
    SFT: 3 epochs, batch 4, grad accumulation 4, lr 2e-4, cosine, warmup 0.03,
         bf16, max length 1024
    DPO: initialise from the SFT adapter, beta 0.1, lr 5e-6, 1 epoch

On Apple MPS the TRL backend forces bf16 and fp16 off. Install training
dependencies from requirements-train.txt.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import types
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from src.build_finetune_data import split_by_asin
from src.config import BASE_MODEL, DATA_DIR, HOLDOUT_SEED, ROOT
from src.jsonl import read_jsonl

app = typer.Typer(add_completion=False)
console = Console()

ADAPTER_ROOT = ROOT / "outputs" / "adapters"


def reject_unsloth_on_apple_silicon() -> None:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        raise SystemExit(
            "Unsloth is CUDA-only and does not support Apple Silicon. "
            "Use --backend trl or --backend mlx."
        )


def _precision(torch: Any) -> tuple[bool, bool]:
    """bf16, fp16. MPS cannot use either; CUDA uses bf16."""
    if torch.cuda.is_available():
        return True, False
    return False, False


def _as_dataset(stage: str, rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    if stage == "sft":
        columns = [{"messages": row["messages"]} for row in rows]
    else:
        columns = [
            {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
            for row in rows
        ]
    return Dataset.from_list(columns)


def _fit_trl(
    model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    stage: str,
    output_dir: Path,
    peft_config: Any,
) -> Path:
    import torch

    bf16, fp16 = _precision(torch)
    train_ds = _as_dataset(stage, train_rows)
    eval_ds = _as_dataset(stage, eval_rows) if len(eval_rows) >= 1 else None
    common = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=bf16,
        fp16=fp16,
        max_length=1024,
        logging_steps=10,
        report_to="none",
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        seed=HOLDOUT_SEED,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    if stage == "sft":
        from trl import SFTConfig, SFTTrainer

        args = SFTConfig(
            **common,
            num_train_epochs=3,
            learning_rate=2e-4,
            assistant_only_loss=True,
            loss_type="nll",
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
    else:
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(
            **common,
            beta=0.1,
            learning_rate=5e-6,
            num_train_epochs=1,
        )
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            model_adapter_name="policy",
            ref_adapter_name="reference",
        )
    # Checkpointing a frozen base yields no gradients unless inputs require them.
    trained = trainer.model
    config = getattr(trained, "config", None)
    if config is not None:
        config.use_cache = False
    base = trained.get_base_model() if hasattr(trained, "get_base_model") else trained
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _load_tokenizer_and_base(base_model: str, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16, _fp16 = _precision(torch)
    dtype = torch.bfloat16 if bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    # Leave placement to the Trainer. Moving onto MPS first breaks LoRA
    # wrapping and gradient checkpointing.
    return model, tokenizer


def _attach_dpo_adapters(model: Any, sft_adapter: Path) -> Any:
    from peft import PeftModel

    if not sft_adapter.exists():
        raise SystemExit(
            f"DPO starts from the SFT adapter, which was not found at {sft_adapter}. "
            "Run --stage sft for this backend first."
        )
    policy = PeftModel.from_pretrained(
        model,
        str(sft_adapter),
        is_trainable=True,
        adapter_name="policy",
    )
    policy.load_adapter(str(sft_adapter), adapter_name="reference", is_trainable=False)
    policy.set_adapter("policy")
    return policy


def _lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


def _require_batch(rows: list[dict[str, Any]], backend: str) -> None:
    if len(rows) < 4:
        raise SystemExit(
            f"{backend} trains with batch size 4, but the training split has "
            f"{len(rows)} rows."
        )


def train_trl(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    stage: str,
    output_dir: Path,
    base_model: str,
    sft_adapter: Optional[Path],
) -> Path:
    import torch

    _require_batch(train_rows, "trl")
    model, tokenizer = _load_tokenizer_and_base(base_model, torch)
    peft_config = None
    if stage == "dpo":
        model = _attach_dpo_adapters(model, Path(sft_adapter) if sft_adapter else Path())
    else:
        peft_config = _lora_config()
    return _fit_trl(model, tokenizer, train_rows, eval_rows, stage, output_dir, peft_config)


def train_unsloth(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    stage: str,
    output_dir: Path,
    base_model: str,
    sft_adapter: Optional[Path],
) -> Path:
    reject_unsloth_on_apple_silicon()
    from unsloth import FastLanguageModel

    _require_batch(train_rows, "unsloth")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    peft_config = None
    if stage == "dpo":
        model = _attach_dpo_adapters(model, Path(sft_adapter) if sft_adapter else Path())
    else:
        peft_config = _lora_config()
    return _fit_trl(model, tokenizer, train_rows, eval_rows, stage, output_dir, peft_config)


def _write_mlx_split(stage: str, rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if stage == "sft":
                payload = {"messages": row["messages"]}
            else:
                system = ""
                user = ""
                for message in row["prompt"]:
                    if message["role"] == "system":
                        system = message["content"]
                    elif message["role"] == "user":
                        user = message["content"]
                payload = {
                    "system": system,
                    "prompt": user,
                    "chosen": row["chosen"][0]["content"],
                    "rejected": row["rejected"][0]["content"],
                }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _mlx_iters(n_train: int, batch_size: int, epochs: int) -> int:
    if n_train < batch_size:
        raise SystemExit(
            f"mlx trains with batch size {batch_size}, but the training split has {n_train} rows."
        )
    return max(1, (n_train // batch_size) * epochs)


def train_mlx(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    stage: str,
    output_dir: Path,
    base_model: str,
    sft_adapter: Optional[Path],
) -> Path:
    try:
        import mlx_lm.lora as lora
    except ImportError as exc:
        raise SystemExit(
            f"mlx-lm is not installed ({exc}). See requirements-train.txt."
        ) from exc

    if stage == "dpo":
        try:
            import mlx_lm.dpo as dpo
        except ImportError as exc:
            raise SystemExit(
                "This mlx-lm install has no DPO trainer (mlx_lm.lora is SFT-only). "
                "Run the DPO stage with --backend trl, or install an mlx-lm build "
                "that provides mlx_lm.dpo."
            ) from exc
        if not hasattr(dpo, "run") or not hasattr(dpo, "CONFIG_DEFAULTS"):
            raise SystemExit(
                "mlx_lm.dpo is installed but has no run()/CONFIG_DEFAULTS entry point. "
                "Run the DPO stage with --backend trl."
            )
        runner = dpo
        adapter_file = Path(sft_adapter or "") / "adapters.safetensors"
        if not adapter_file.exists():
            raise SystemExit(
                f"DPO starts from the SFT adapter, which was not found at {adapter_file}. "
                "Run --stage sft --backend mlx first."
            )
    else:
        runner = lora
        adapter_file = None

    epochs = 3 if stage == "sft" else 1
    learning_rate = 2e-4 if stage == "sft" else 5e-6
    # Apple Silicon: batch 4 @ 1024 OOMs on Metal for Qwen3-4B LoRA.
    # Keep the effective batch (16) via grad accumulation.
    batch_size = 1
    grad_accumulation = 16
    max_seq_length = 768
    iters = _mlx_iters(len(train_rows), batch_size, epochs)
    updates = max(1, math.ceil(iters / grad_accumulation))
    data_dir = ROOT / "outputs" / "mlx_data" / stage
    _write_mlx_split(stage, train_rows, data_dir / "train.jsonl")
    # mlx refuses a validation set smaller than the batch.
    if len(eval_rows) >= batch_size:
        _write_mlx_split(stage, eval_rows, data_dir / "valid.jsonl")
    elif eval_rows:
        console.print(
            f"[yellow]Skipping mlx validation:[/yellow] {len(eval_rows)} rows is below batch size {batch_size}."
        )

    args = copy.deepcopy(runner.CONFIG_DEFAULTS)
    args.update(
        {
            "model": base_model,
            "train": True,
            "data": str(data_dir),
            "fine_tune_type": "lora",
            "optimizer": "adamw",
            "num_layers": -1,
            "batch_size": batch_size,
            "iters": iters,
            "learning_rate": learning_rate,
            "steps_per_report": 10,
            "steps_per_eval": max(10, iters),
            "grad_accumulation_steps": grad_accumulation,
            "adapter_path": str(output_dir),
            "save_every": iters,
            "max_seq_length": max_seq_length,
            "grad_checkpoint": True,
            "mask_prompt": True,
            "seed": HOLDOUT_SEED,
            "lora_parameters": {"rank": 32, "dropout": 0.05, "scale": 64.0},
            "lr_schedule": {
                "name": "cosine_decay",
                "arguments": [learning_rate, updates, 0.0],
                "warmup": max(1, int(round(0.03 * updates))),
            },
        }
    )
    if adapter_file is not None:
        args["resume_adapter_file"] = str(adapter_file)
        args["beta"] = 0.1
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.run(types.SimpleNamespace(**args))
    saved = output_dir / "adapters.safetensors"
    if not saved.exists():
        raise SystemExit(f"mlx training finished without writing {saved}")
    return output_dir


BACKENDS = {
    "trl": train_trl,
    "unsloth": train_unsloth,
    "mlx": train_mlx,
}


@app.command()
def main(
    stage: str = typer.Option("sft", help="sft or dpo"),
    backend: str = typer.Option("trl", help="unsloth (CUDA), trl (CUDA or MPS), or mlx (Apple Silicon)"),
    base_model: str = typer.Option(BASE_MODEL),
    data: Optional[Path] = typer.Option(None, help="Training JSONL. Defaults to data/sft.jsonl or data/dpo.jsonl"),
    sft_adapter: Optional[Path] = typer.Option(None, help="SFT adapter to initialise DPO from"),
    output: Optional[Path] = typer.Option(None, help="Adapter output directory"),
    eval_fraction: float = typer.Option(0.1, help="Fraction of training products held out for eval"),
    seed: int = typer.Option(HOLDOUT_SEED),
) -> None:
    """Train a LoRA adapter. SFT and DPO share the split and the output layout."""
    if stage not in {"sft", "dpo"}:
        raise typer.BadParameter("stage must be sft or dpo")
    if backend not in BACKENDS:
        raise typer.BadParameter("backend must be unsloth, trl, or mlx")
    if backend == "unsloth":
        reject_unsloth_on_apple_silicon()

    data_path = data or (DATA_DIR / ("dpo.jsonl" if stage == "dpo" else "sft.jsonl"))
    rows = read_jsonl(data_path)
    if not rows:
        raise typer.BadParameter(f"No rows in {data_path}. Run build_finetune_data first.")
    train_rows, eval_rows = split_by_asin(rows, eval_fraction, seed)
    output_dir = output or (ADAPTER_ROOT / backend / stage)
    adapter = sft_adapter or (ADAPTER_ROOT / backend / "sft")
    console.print(
        f"[cyan]{stage}[/cyan] via [cyan]{backend}[/cyan] on {base_model}: "
        f"{len(train_rows)} train rows, {len(eval_rows)} eval rows"
    )
    saved = BACKENDS[backend](train_rows, eval_rows, stage, output_dir, base_model, adapter)
    console.print(f"[green]adapter saved[/green] → {saved}")


if __name__ == "__main__":
    app()
