"""Download Amazon Reviews 2023 source files from the Hugging Face hub."""

from __future__ import annotations

from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from rich.console import Console

from src.config import HF_CACHE_DIR, HF_DATASET, META_JSONL, REVIEWS_JSONL

app = typer.Typer(add_completion=False)
console = Console()

FILES = {
    "meta": META_JSONL,
    "reviews": REVIEWS_JSONL,
}


def fetch(key: str, cache_dir: Path) -> Path:
    filename = FILES[key]
    console.print(f"[cyan]Downloading[/cyan] {filename}")
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    size_gb = Path(path).stat().st_size / 1e9
    console.print(f"[green]Ready[/green] {filename} ({size_gb:.2f} GB) → {path}")
    return Path(path)


@app.command()
def main(
    which: str = typer.Option("meta", help=f"Comma-separated: {','.join(FILES)}"),
    cache_dir: Path = typer.Option(HF_CACHE_DIR, help="Local download cache"),
) -> None:
    """Fetch raw category files; downloads resume if interrupted."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key in [k.strip() for k in which.split(",") if k.strip()]:
        if key not in FILES:
            raise typer.BadParameter(f"Unknown file {key}. Choose from {list(FILES)}")
        fetch(key, cache_dir)


if __name__ == "__main__":
    app()
