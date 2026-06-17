"""XTTS v2 streaming engine — the Phase 1 implementation, repackaged.

Behavior identical to the previous inline implementation in `main.py`:
precomputed speaker conditioning + `inference_stream()` yielding float32
tensors converted to PCM16-LE.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
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


def _split_for_batch_tts(text: str, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text) if s.strip()]
    sentence_chunks: list[str] = []

    for sentence in sentences or [text.strip()]:
        if len(sentence) <= max_chars:
            sentence_chunks.append(sentence)
            continue

        words = sentence.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and len(candidate) > max_chars:
                sentence_chunks.append(" ".join(current).rstrip(",;:") + ".")
                current = [word]
            else:
                current.append(word)
        if current:
            sentence_chunks.append(" ".join(current))

    packed: list[str] = []
    current = ""
    for chunk in sentence_chunks:
        candidate = f"{current} {chunk}".strip()
        if current and len(candidate) > max_chars:
            packed.append(current)
            current = chunk
        else:
            current = candidate
    if current:
        packed.append(current)

    return packed


def _reference_audio_paths(speaker_path: Path) -> list[Path]:
    env_paths = os.getenv("TTS_REFERENCE_WAVS")
    if env_paths:
        refs = [Path(value.strip()) for value in env_paths.split(";") if value.strip()]
        return [path for path in refs if path.exists()]

    prefix = re.sub(r"_\d+$", "", speaker_path.stem)
    if prefix != speaker_path.stem:
        refs = sorted(speaker_path.parent.glob(f"{prefix}*.wav"))
        if refs:
            return refs

    return [speaker_path]


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
        self._batch_speed = float(os.getenv("TTS_BATCH_SPEED", "1.0"))
        self._batch_max_chars = int(os.getenv("TTS_BATCH_MAX_CHARS", "120"))
        self._batch_pause_ms = int(os.getenv("TTS_BATCH_PAUSE_MS", "500"))
        self._temperature = float(os.getenv("TTS_TEMPERATURE", "0.78"))
        self._top_p = float(os.getenv("TTS_TOP_P", "0.88"))
        self._top_k = int(os.getenv("TTS_TOP_K", "50"))
        self._repetition_penalty = float(os.getenv("TTS_REPETITION_PENALTY", "8.0"))
        self._gpt_cond_len = int(os.getenv("TTS_GPT_COND_LEN", "12"))
        self._gpt_cond_chunk_len = int(os.getenv("TTS_GPT_COND_CHUNK_LEN", "6"))
        self._max_ref_len = int(os.getenv("TTS_MAX_REF_LEN", "15"))
        self._sound_norm_refs = os.getenv("TTS_SOUND_NORM_REFS", "0") == "1"
        self._speaker_cache: dict[str, tuple] = {}

    # -- StreamEngine protocol -------------------------------------------------

    def warm_up(self, speaker_path: Path) -> None:
        self._get_speaker_conditioning(speaker_path)
        refs = _reference_audio_paths(speaker_path)
        print(f"[XTTS] precomputed conditioning for {', '.join(path.name for path in refs)}")

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
        refs = _reference_audio_paths(speaker_path)
        cache_key = "|".join(str(path) for path in refs)
        cached = self._speaker_cache.get(cache_key)
        if cached is not None:
            return cached

        gpt_cond_latent, speaker_embedding = (
            self.tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[str(path) for path in refs],
                gpt_cond_len=self._gpt_cond_len,
                gpt_cond_chunk_len=self._gpt_cond_chunk_len,
                max_ref_length=self._max_ref_len,
                sound_norm_refs=self._sound_norm_refs,
            )
        )
        self._speaker_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
        return gpt_cond_latent, speaker_embedding

    # -- Legacy batch path (used by /synthesize) -------------------------------

    def tts_to_file(self, text: str, speaker_wav: str, file_path: str) -> None:
        """Batch synthesis using cached speaker conditioning."""
        speaker_path = Path(speaker_wav)
        gpt_cond_latent, speaker_embedding = self._get_speaker_conditioning(speaker_path)

        chunks = _split_for_batch_tts(text, self._batch_max_chars)
        pause = np.zeros(int(self.sample_rate * self._batch_pause_ms / 1000), dtype=np.float32)
        wav_parts: list[np.ndarray] = []

        for index, chunk in enumerate(chunks):
            result = self.tts.synthesizer.tts_model.inference(
                text=chunk,
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                speed=self._batch_speed,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                repetition_penalty=self._repetition_penalty,
            )
            wav_parts.append(np.asarray(result["wav"], dtype=np.float32))
            if index < len(chunks) - 1 and self._batch_pause_ms > 0:
                wav_parts.append(pause)

        wav = np.concatenate(wav_parts) if wav_parts else np.zeros(0, dtype=np.float32)
        sf.write(file_path, wav, self.sample_rate)
