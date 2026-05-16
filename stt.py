"""Speech-to-text via faster-whisper.

Lives in the same FastAPI process as the TTS engines so the GPU stays loaded
once and serves both directions of the voice pipeline. Loaded lazily on the
first /transcribe call (same pattern as the TTS engines in engines/).
"""

from __future__ import annotations

import io
import os
from typing import Any

from faster_whisper import WhisperModel


_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Lazy-load the faster-whisper model on first use."""
    global _model
    if _model is None:
        model_name = os.getenv("STT_MODEL", "base.en")
        device = os.getenv("STT_DEVICE", "auto")
        compute_type = os.getenv("STT_COMPUTE_TYPE", "default")
        print(f"[STT] loading faster-whisper {model_name} (device={device}, compute_type={compute_type})")
        _model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _model


def transcribe(audio: Any, language: str | None = "en") -> dict[str, Any]:
    """Run faster-whisper on a WAV file-like / path / float32 ndarray.

    Returns {"text": str, "language": str, "duration_ms": int}.
    """
    model = get_model()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=1,                       # greedy — fastest, fine for conversational use
        vad_filter=True,                   # trim leading/trailing silence
        condition_on_previous_text=False,  # each utterance is independent
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "duration_ms": int(info.duration * 1000),
    }


def transcribe_bytes(wav_bytes: bytes, language: str | None = "en") -> dict[str, Any]:
    """Convenience wrapper — accepts a WAV byte string from an HTTP body."""
    return transcribe(io.BytesIO(wav_bytes), language=language)
