"""XTTS v2 streaming engine — the Phase 1 implementation, repackaged.

Behavior identical to the previous inline implementation in `main.py`:
precomputed speaker conditioning + `inference_stream()` yielding float32
tensors converted to PCM16-LE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from TTS.api import TTS

# === HACK BUAT BYPASS PYTORCH 2.6+ SECURITY ===
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ==============================================


def _tensor_chunk_to_pcm16(chunk) -> bytes:
    if hasattr(chunk, "detach"):
        samples = chunk.detach().cpu().numpy()
    else:
        samples = np.asarray(chunk)

    samples = samples.squeeze()
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


class XTTSEngine:
    """XTTS v2 wrapper implementing the StreamEngine protocol."""

    sample_rate = int(os.getenv("TTS_STREAM_SAMPLE_RATE", "24000"))

    def __init__(self) -> None:
        model_name = os.getenv(
            "TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[XTTS] loading {model_name} on {device}")

        engine = TTS(model_name)
        if hasattr(engine, "to"):
            engine = engine.to(device)
        self.tts = engine
        self.device = device

        self._stream_chunk_size = int(os.getenv("TTS_STREAM_CHUNK_SIZE", "20"))
        self._stream_overlap_len = int(os.getenv("TTS_STREAM_OVERLAP_LEN", "1024"))
        self._speaker_cache: dict[str, tuple] = {}

    # -- StreamEngine protocol -------------------------------------------------

    def warm_up(self, speaker_path: Path) -> None:
        self._get_speaker_conditioning(speaker_path)
        print(f"[XTTS] precomputed conditioning for {speaker_path.name}")

    def stream(
        self,
        text: str,
        speaker_path: Path,
        emotion: str = "HAPPY",
    ) -> Iterator[bytes]:
        # XTTS ignores `emotion` — emotional handling is implicit in the model.
        gpt_cond_latent, speaker_embedding = self._get_speaker_conditioning(speaker_path)

        chunks = self.tts.synthesizer.tts_model.inference_stream(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=self._stream_chunk_size,
            overlap_wav_len=self._stream_overlap_len,
        )

        for chunk in chunks:
            pcm = _tensor_chunk_to_pcm16(chunk)
            if pcm:
                yield pcm

    # -- Internal --------------------------------------------------------------

    def _get_speaker_conditioning(self, speaker_path: Path):
        cache_key = str(speaker_path)
        cached = self._speaker_cache.get(cache_key)
        if cached is not None:
            return cached

        gpt_cond_latent, speaker_embedding = (
            self.tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[str(speaker_path)]
            )
        )
        self._speaker_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding

    # -- Legacy batch path (used by /synthesize) -------------------------------

    def tts_to_file(self, text: str, speaker_wav: str, file_path: str) -> None:
        """Backwards-compat shim for the legacy MP3 endpoint in `main.py`."""
        self.tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            file_path=file_path,
        )
