"""ChatTTS streaming engine (Phase 5).

Conversational TTS with explicit emotion tags via RefineTextParams. Wraps
ChatTTS so it conforms to the same `StreamEngine` protocol as XTTS — the
WebSocket pipeline is unaware of which engine produced the PCM stream.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import numpy as np

# Word-level prosody tokens the LLM is allowed to emit inline. When any of
# these appear in the text, the refiner is skipped — otherwise it rewrites
# or removes them (see ChatTTS docs §"For word level manual control").
INLINE_PROSODY_RE = re.compile(r"\[(?:uv_break|lbreak|laugh)\]", re.IGNORECASE)

# ChatTTS is an optional dependency. Importing eagerly would crash boot when
# TTS_ENGINE=xtts even if ChatTTS is not installed. main.py only imports this
# module when TTS_ENGINE=chattts, so the import is fine here.
import ChatTTS  # type: ignore

try:
    import torchaudio  # type: ignore
except ImportError:  # pragma: no cover - torchaudio is part of the recommended stack
    torchaudio = None


# Emotion → RefineText tag mapping. See plan.md §13 (Phase 5c).
EMOTION_TAG_MAP: dict[str, str] = {
    "HAPPY":   "[oral_2][laugh_0][break_3]",
    "EXCITED": "[oral_2][laugh_1][break_2]",
    "SAD":     "[oral_3][break_5][uv_break]",
    "ANGRY":   "[oral_1][break_2]",
    "SHOCKED": "[oral_2][break_4][uv_break]",
}
DEFAULT_REFINE_TAG = "[oral_2][break_3]"


def _numpy_chunk_to_pcm16(chunk) -> bytes:
    """ChatTTS yields float32 ndarray; convert to little-endian s16 PCM."""
    samples = np.asarray(chunk).squeeze()
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def _load_audio_for_speaker(path: Path, target_sr: int):
    """Load a reference WAV resampled to ``target_sr`` (mono).

    ChatTTS's ``sample_audio_speaker`` expects a 1-D numpy array.
    """
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is required to extract ChatTTS speaker embeddings."
        )

    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


class ChatTTSEngine:
    """ChatTTS wrapper implementing the StreamEngine protocol."""

    sample_rate = 24000  # ChatTTS native output — matches XTTS to keep ESP32 happy

    def __init__(self) -> None:
        local_path = os.getenv("CHATTTS_MODEL_DIR", "./models/chattts")
        compile_flag = os.getenv("CHATTTS_COMPILE", "1") == "1"
        print(f"[ChatTTS] loading from {local_path} (compile={compile_flag})")

        self.chat = ChatTTS.Chat()
        load_ok = self.chat.load(
            source="local",
            local_path=local_path,
            compile=compile_flag,
        )
        if load_ok is False:
            raise RuntimeError(
                f"ChatTTS.load() returned False — check that {local_path} "
                "contains the asset/ and config/ directories. See README §2.1b."
            )

        self._refine_temp = float(os.getenv("CHATTTS_REFINE_TEMP", "0.3"))
        self._infer_temp = float(os.getenv("CHATTTS_INFER_TEMP", "0.3"))
        self._spk_smp_cache: dict[str, str] = {}

    # -- StreamEngine protocol -------------------------------------------------

    def warm_up(self, speaker_path: Path) -> None:
        spk_smp = self._get_spk_smp(speaker_path)
        # Drain a single silent inference to build CUDA graphs.
        for _ in self._raw_stream("Hello.", spk_smp, "[oral_2]"):
            pass
        print(f"[ChatTTS] warm-up complete for {speaker_path.name}")

    def stream(
        self,
        text: str,
        speaker_path: Path,
        emotion: str = "HAPPY",
    ) -> Iterator[bytes]:
        spk_smp = self._get_spk_smp(speaker_path)
        refine_prompt = EMOTION_TAG_MAP.get(emotion.upper(), DEFAULT_REFINE_TAG)
        yield from self._raw_stream(text, spk_smp, refine_prompt)

    # -- Internal --------------------------------------------------------------

    def _raw_stream(
        self,
        text: str,
        spk_smp: str,
        refine_prompt: str,
    ) -> Iterator[bytes]:
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_smp=spk_smp,
            temperature=self._infer_temp,
        )

        has_inline_tokens = bool(INLINE_PROSODY_RE.search(text))

        if has_inline_tokens:
            # Honor the LLM's inline [uv_break] / [lbreak] / [laugh] markers
            # verbatim — the refiner would otherwise rewrite or strip them.
            wav_iter = self.chat.infer(
                [text],
                stream=True,
                skip_refine_text=True,
                params_infer_code=params_infer_code,
            )
        else:
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt=refine_prompt,
                temperature=self._refine_temp,
            )
            wav_iter = self.chat.infer(
                [text],
                stream=True,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )

        for chunk in wav_iter:
            # ChatTTS shape: list-like of (n_speakers, samples) per chunk.
            wav = chunk[0] if isinstance(chunk, (list, tuple)) else chunk
            pcm = _numpy_chunk_to_pcm16(wav)
            if pcm:
                yield pcm

    def _get_spk_smp(self, speaker_path: Path) -> str:
        cache_key = str(speaker_path)
        cached = self._spk_smp_cache.get(cache_key)
        if cached is not None:
            return cached

        persisted = _load_persisted_embedding(speaker_path)
        if persisted is not None:
            self._spk_smp_cache[cache_key] = persisted
            return persisted

        audio = _load_audio_for_speaker(speaker_path, self.sample_rate)
        spk_smp = self.chat.sample_audio_speaker(audio)
        self._spk_smp_cache[cache_key] = spk_smp
        _persist_speaker_embedding(speaker_path, spk_smp)
        return spk_smp


# -- Speaker embedding persistence (on-disk cache) ------------------------------

def _embedding_path(speaker_path: Path) -> Path:
    return speaker_path.with_suffix(".spk_smp.txt")


def _persist_speaker_embedding(speaker_path: Path, spk_smp: str) -> None:
    target = _embedding_path(speaker_path)
    try:
        target.write_text(spk_smp, encoding="utf-8")
        print(f"[ChatTTS] saved speaker embedding to {target.name}")
    except OSError as exc:
        print(f"[ChatTTS] could not persist embedding: {exc}")


def _load_persisted_embedding(speaker_path: Path) -> str | None:
    target = _embedding_path(speaker_path)
    if not target.exists():
        return None
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        return None
