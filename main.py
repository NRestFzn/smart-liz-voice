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

SPEAKER_REFERENCE = BASE_DIR / "voiceover.wav"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
MP3_BITRATE = os.getenv("TTS_MP3_BITRATE", "48k")
MP3_SAMPLE_RATE = os.getenv("TTS_MP3_SAMPLE_RATE", "24000")
TTS_LEADING_SILENCE_MS = int(os.getenv("TTS_LEADING_SILENCE_MS", "250"))
AUDIO_TTL_SECONDS = int(os.getenv("AUDIO_TTL_SECONDS", "900"))


def ffmpeg_exe() -> str:
    env_ffmpeg = os.getenv("FFMPEG_BIN")
    if env_ffmpeg:
        return env_ffmpeg

    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg

    local_candidates = [
        BASE_DIR / "ffmpeg.exe",
        BASE_DIR / "ffmpeg" / "ffmpeg.exe",
        BASE_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
        BASE_DIR / "tools" / "ffmpeg" / "ffmpeg.exe",
        BASE_DIR / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]

    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "ffmpeg tidak ditemukan. Tambahkan ffmpeg ke PATH, set env FFMPEG_BIN, "
        "atau taruh ffmpeg.exe di folder project/ffmpeg/bin."
    )


app = FastAPI()
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# Load model XTTSv2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TTS model on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


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


@app.get("/health")
async def health():
    return {
        "ok": True,
        "device": device,
        "audio_format": "mp3",
        "audio_dir": str(AUDIO_DIR),
        "leading_silence_ms": TTS_LEADING_SILENCE_MS,
        "storage_policy": "keep_latest_audio_only",
    }


@app.post("/synthesize")
async def synthesize(request: Request, payload: TTSRequest):
    try:
        cleanup_audio_files(expired_only=True)

        if not SPEAKER_REFERENCE.exists():
            raise HTTPException(
                status_code=404,
                detail="File referensi suara 'voiceover.wav' nggak ketemu!",
            )

        audio_id = f"liz-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        wav_path = AUDIO_DIR / f"{audio_id}.wav"
        mp3_filename = f"{audio_id}.mp3"
        mp3_path = AUDIO_DIR / mp3_filename

        tts.tts_to_file(
            text=payload.text,
            speaker_wav=str(SPEAKER_REFERENCE),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
