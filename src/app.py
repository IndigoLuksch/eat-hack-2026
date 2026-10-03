"""Vercel entrypoint. The demo routes live in demo_server."""

from src.demo_server import app

__all__ = ["app"]
