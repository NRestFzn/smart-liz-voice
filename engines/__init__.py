"""Pluggable streaming TTS engines.

Each engine implements the StreamEngine protocol defined in `base.py` and is
selected at runtime via the `TTS_ENGINE` environment variable read in
`main.py`. See plan.md §13 (Phase 5) for the design rationale.
"""

from .base import StreamEngine

__all__ = ["StreamEngine"]
