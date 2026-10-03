"""Vercel entrypoint. The demo routes live in demo_server."""

from src.demo_server import _start_model, app

_start_model()

__all__ = ["app"]
