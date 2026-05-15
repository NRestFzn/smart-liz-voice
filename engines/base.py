"""Common interface every streaming TTS engine must implement."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class StreamEngine(Protocol):
    """Streaming TTS engine contract.

    Implementations live in `xtts_engine.py` and `chattts_engine.py`.
    Both engines MUST yield 16-bit signed little-endian mono PCM at
    `sample_rate` Hz so the downstream WebSocket frames stay byte-identical
    regardless of which engine is loaded.
    """

    sample_rate: int

    def warm_up(self, speaker_path: Path) -> None:
        """Pre-load the model and any speaker conditioning for the default voice.

        Called once from the FastAPI startup hook. Implementations should make
        the first real request cheap — that means running at least one silent
        inference if the engine relies on CUDA-graph compilation.
        """
        ...

    def stream(
        self,
        text: str,
        speaker_path: Path,
        emotion: str = "HAPPY",
    ) -> Iterator[bytes]:
        """Yield raw PCM16-LE mono bytes for ``text``.

        The XTTS engine ignores ``emotion``. The ChatTTS engine maps it to a
        RefineText prompt. See plan.md §13 (Phase 5c).
        """
        ...
