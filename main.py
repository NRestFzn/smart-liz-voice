"""FastAPI entrypoint for the streaming TTS service.

Thin shell: parses requests, resolves the speaker WAV, dispatches to the
selected engine (XTTS or ChatTTS — see plan.md §13), and streams PCM bytes
back over HTTP. Engine-specific code lives in `engines/`.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from engines.base import StreamEngine

import io
import speech_recognition as sr

# -----------------------------------------------------------------------------
# Paths / env
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
AUDIO_DIR = BASE_DIR / "public" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VOICE_DIR = BASE_DIR / "sample_voice"
DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "VO_Nicole_01.wav")
LEGACY_SPEAKER_REFERENCE = BASE_DIR / "voiceover.wav"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
MP3_BITRATE = os.getenv("TTS_MP3_BITRATE", "48k")
MP3_SAMPLE_RATE = os.getenv("TTS_MP3_SAMPLE_RATE", "24000")
TTS_LEADING_SILENCE_MS = int(os.getenv("TTS_LEADING_SILENCE_MS", "0"))
AUDIO_TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "900"))
TTS_WARM_UP_ON_STARTUP = os.getenv("TTS_WARM_UP_ON_STARTUP", "1") == "1"

TTS_ENGINE_NAME = os.getenv("TTS_ENGINE", "xtts").lower()

# -----------------------------------------------------------------------------
# Engine loader
# -----------------------------------------------------------------------------
def _load_engine() -> StreamEngine:
    if TTS_ENGINE_NAME in {"system", "sapi", "windows"}:
        from engines.system_engine import SystemTTSEngine
        return SystemTTSEngine()

    if TTS_ENGINE_NAME == "chattts":
        from engines.chattts_engine import ChatTTSEngine
        return ChatTTSEngine()

    try:
        from engines.xtts_engine import XTTSEngine
        return XTTSEngine()
    except Exception as exc:
        if os.getenv("TTS_ALLOW_SYSTEM_FALLBACK", "1") != "1":
            raise

        print(f"[TTS] XTTS unavailable, falling back to Windows System.Speech: {exc}")
        from engines.system_engine import SystemTTSEngine
        return SystemTTSEngine()


engine: StreamEngine | None = None


def get_engine() -> StreamEngine:
    """Lazy-load the configured engine on first use."""
    global engine
    if engine is None:
        engine = _load_engine()
    return engine


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI()
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


def ffmpeg_exe() -> str:
    env_ffmpeg = os.getenv("FFMPEG_BIN")
    if env_ffmpeg:
        return env_ffmpeg
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    return "ffmpeg"


def _available_voice_files() -> list[str]:
    if not SAMPLE_VOICE_DIR.exists():
        return []
    return sorted([p.name for p in SAMPLE_VOICE_DIR.glob("*.wav") if p.is_file()])


def resolve_speaker_wav(speaker_wav: str) -> Path:
    speaker_wav = (speaker_wav or "default").strip()

    if speaker_wav.lower() == "default":
        candidates = [DEFAULT_SPEAKER_WAV]
    else:
        candidates = [speaker_wav]

    for name in candidates:
        safe_name = Path(name).name
        if safe_name != name and speaker_wav.lower() != "default":
            raise HTTPException(
                status_code=400,
                detail="speaker_wav harus nama file saja (tanpa path).",
            )

        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}.wav"

        candidate_path = SAMPLE_VOICE_DIR / safe_name
        if candidate_path.exists():
            return candidate_path

    if speaker_wav.lower() == "default" and LEGACY_SPEAKER_REFERENCE.exists():
        return LEGACY_SPEAKER_REFERENCE

    available = _available_voice_files()
    detail = "File speaker_wav tidak ditemukan."
    if available:
        detail += f" Pilihan tersedia: {', '.join(available)}"
    raise HTTPException(status_code=404, detail=detail)


class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = "default"
    language: str = "en"
    emotion: str = "HAPPY"   # used by ChatTTS, ignored by XTTS


def cleanup_audio_files(keep_filename=None, expired_only=False) -> None:
    cutoff = time.time() - AUDIO_TTL_SECONDS if AUDIO_TTL_SECONDS > 0 else 0

    for path in AUDIO_DIR.glob("liz-*.*"):
        try:
            if keep_filename is not None and path.name == keep_filename:
                continue
            if expired_only and AUDIO_TTL_SECONDS > 0 and path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
        except OSError:
            pass


def make_audio_url(request: Request, filename: str) -> str:
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/audio/{filename}"
    return str(request.url_for("audio", path=filename))


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
    ]

    if TTS_LEADING_SILENCE_MS > 0:
        cmd.extend(["-af", f"adelay={TTS_LEADING_SILENCE_MS}:all=1"])

    cmd.extend([
        "-ar",
        MP3_SAMPLE_RATE,
        "-ac",
        "1",
        "-b:a",
        MP3_BITRATE,
        str(mp3_path),
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg failed to convert WAV to MP3"
        raise RuntimeError(detail)


def stream_pcm_chunks(text: str, speaker_path: Path, emotion: str) -> Iterator[bytes]:
    """Engine-agnostic streamer used by both the PCM and WAV endpoints."""
    yield from get_engine().stream(text, speaker_path, emotion)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    active_engine = get_engine()
    return {
        "ok": True,
        "configured_engine": TTS_ENGINE_NAME,
        "active_engine": active_engine.__class__.__name__,
        "sample_rate": active_engine.sample_rate,
        "audio_format": "mp3",
        "audio_dir": str(AUDIO_DIR),
        "voices_dir": str(SAMPLE_VOICE_DIR),
        "available_voices": _available_voice_files(),
        "leading_silence_ms": TTS_LEADING_SILENCE_MS,
        "storage_policy": "keep_latest_audio_only",
        "stt_model": os.getenv("STT_MODEL", "base.en"),
    }


@app.post("/transcribe")
async def transcribe_endpoint(request: Request):
    """
    Speech-to-text pake Google API.
    Menerima WAV dari Express/ESP32, ngembaliin {text, language}.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body.")

    try:
        # Bungkus raw bytes dari req.body jadi file-like object 
        # biar bisa dibaca sama library speech_recognition
        wav_io = io.BytesIO(body)
        recognizer = sr.Recognizer()

        # Load audio dari memory (wav_io)
        with sr.AudioFile(wav_io) as source:
            audio_content = recognizer.record(source)

        # Tembak ke Google API persis kayak script test lu kemarin
        # Set language id-ID biar paten bahasa Indonesia
        hasil_teks = recognizer.recognize_google(audio_content, language="id-ID")
        
        print(f"[STT] Berhasil nangkep: {hasil_teks}")
        
        # Return JSON sesuai ekspektasi ESP32 lu
        return {
            "text": hasil_teks,
            "language": "id"
        }

    except sr.UnknownValueError:
        print("[STT] Google API gak bisa nangkep suaranya (noise / gak jelas)")
        # Return text kosong biar ESP32 nggak error parsing
        return {"text": "", "language": "id"}
        
    except sr.RequestError as exc:
        print(f"[STT] Gagal request ke Google API: {exc}")
        raise HTTPException(status_code=502, detail="Google API Error")
        
    except Exception as exc:
        print(f"[STT] Internal error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/synthesize")
async def synthesize(request: Request, payload: TTSRequest):
    """Batch endpoint that generates a full WAV, then converts it to MP3."""
    try:
        cleanup_audio_files(expired_only=True)

        speaker_path = resolve_speaker_wav(payload.speaker_wav)

        audio_id = f"liz-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        wav_path = AUDIO_DIR / f"{audio_id}.wav"
        mp3_filename = f"{audio_id}.mp3"
        mp3_path = AUDIO_DIR / mp3_filename

        engine = get_engine()
        if not hasattr(engine, "tts_to_file"):
            raise HTTPException(
                status_code=503,
                detail="The selected TTS engine does not support batch /synthesize.",
            )

        engine.tts_to_file(  # type: ignore[attr-defined]
            text=payload.text,
            speaker_wav=str(speaker_path),
            file_path=str(wav_path),
        )

        convert_wav_to_mp3(wav_path, mp3_path)

        try:
            wav_path.unlink()
        except OSError:
            pass

        cleanup_audio_files(keep_filename=mp3_filename)

        return {
            "text": payload.text,
            "audio_url": make_audio_url(request, mp3_filename),
            "audio_path": f"/audio/{mp3_filename}",
            "audio_format": "audio/mpeg",
            "audio_bitrate": MP3_BITRATE,
            "expires_in_seconds": AUDIO_TTL_SECONDS,
            "storage_policy": "keep_latest_audio_only",
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/synthesize_stream")
async def synthesize_stream(payload: TTSRequest):
    """Streaming endpoint — yields raw PCM16 mono chunks at the engine's sample rate."""
    try:
        speaker_path = resolve_speaker_wav(payload.speaker_wav)
    except HTTPException:
        raise

    eng = get_engine()

    def generator():
        try:
            for pcm in stream_pcm_chunks(payload.text, speaker_path, payload.emotion):
                yield pcm
        except Exception as exc:
            print(f"Stream error: {exc}")
            return

    headers = {
        "X-Sample-Rate": str(eng.sample_rate),
        "X-Channels": "1",
        "X-Sample-Format": "s16le",
        "X-TTS-Engine": TTS_ENGINE_NAME,
        "Cache-Control": "no-store",
    }

    return StreamingResponse(generator(), media_type="audio/pcm", headers=headers)


@app.post("/synthesize_wav_stream")
async def synthesize_wav_stream(payload: TTSRequest):
    """Same as /synthesize_stream but with a streaming WAV RIFF header.

    Useful for testing the stream with media players that need a container.
    """
    try:
        speaker_path = resolve_speaker_wav(payload.speaker_wav)
    except HTTPException:
        raise

    eng = get_engine()

    def build_wav_header(sample_rate: int) -> bytes:
        return (
            b"RIFF"
            + struct.pack("<I", 0xFFFFFFFF)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data"
            + struct.pack("<I", 0xFFFFFFFF)
        )

    def generator():
        yield build_wav_header(eng.sample_rate)
        try:
            for pcm in stream_pcm_chunks(payload.text, speaker_path, payload.emotion):
                yield pcm
        except Exception as exc:
            print(f"Stream error: {exc}")
            return

    return StreamingResponse(generator(), media_type="audio/wav")


@app.on_event("startup")
def warm_up_default_speaker():
    if not TTS_WARM_UP_ON_STARTUP:
        print("[TTS] startup warm-up disabled; set TTS_WARM_UP_ON_STARTUP=1 to enable")
        return

    try:
        speaker_path = resolve_speaker_wav("default")
        active_engine = get_engine()
        active_engine.warm_up(speaker_path)
        print(
            f"[TTS] configured_engine={TTS_ENGINE_NAME} active_engine={active_engine.__class__.__name__} ready - warm-up complete for {speaker_path.name}"
        )
    except Exception as exc:
        print(f"[TTS] Warm-up skipped: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
