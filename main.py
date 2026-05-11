import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from TTS.api import TTS

# === HACK BUAT BYPASS PYTORCH 2.6+ SECURITY ===
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ==============================================

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "public" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_VOICE_DIR = BASE_DIR / "sample_voice"
DEFAULT_SPEAKER_WAV = os.getenv("DEFAULT_SPEAKER_WAV", "VO_Escoffier.wav")
LEGACY_SPEAKER_REFERENCE = BASE_DIR / "voiceover.wav"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
MP3_BITRATE = os.getenv("TTS_MP3_BITRATE", "48k")
MP3_SAMPLE_RATE = os.getenv("TTS_MP3_SAMPLE_RATE", "24000")
AUDIO_TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "900"))


def ffmpeg_exe() -> str:
    env_ffmpeg = os.getenv("FFMPEG_BIN")
    if env_ffmpeg:
        return env_ffmpeg

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg


app = FastAPI()
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ================================
# SET DEVICE (GPU)
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"

TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
tts = None


def get_tts() -> TTS:
    global tts
    if tts is None:
        engine = TTS(TTS_MODEL_NAME)
        if hasattr(engine, "to"):
            engine = engine.to(device)
        tts = engine
    return tts


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
            raise HTTPException(status_code=400, detail="speaker_wav harus nama file saja (tanpa path).")

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

    # If Express calls this service via localhost, this URL is only useful from
    # the same machine. Set PUBLIC_BASE_URL when the ESP must fetch it directly.
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
        "-ar",
        MP3_SAMPLE_RATE,
        "-ac",
        "1",
        "-b:a",
        MP3_BITRATE,
        str(mp3_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg failed to convert WAV to MP3"
        raise RuntimeError(detail)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": device,
        "audio_format": "mp3",
        "audio_dir": str(AUDIO_DIR),
        "voices_dir": str(SAMPLE_VOICE_DIR),
        "available_voices": _available_voice_files(),
        "storage_policy": "keep_latest_audio_only",
    }


@app.post("/synthesize")
async def synthesize(request: Request, payload: TTSRequest):
    try:
        cleanup_audio_files(expired_only=True)

        speaker_path = resolve_speaker_wav(payload.speaker_wav)

        audio_id = f"liz-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        wav_path = AUDIO_DIR / f"{audio_id}.wav"
        mp3_filename = f"{audio_id}.mp3"
        mp3_path = AUDIO_DIR / mp3_filename

        get_tts().tts_to_file(
            text=payload.text,
            speaker_wav=str(speaker_path),
            language="en",
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


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
