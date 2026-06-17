"""Windows SAPI fallback TTS engine.

This engine is intentionally small and dependency-light. It lets the FastAPI
service run on machines where the XTTS/Torch stack is unavailable, such as a
Python version that is too new for Coqui TTS.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator


class SystemTTSEngine:
    sample_rate = int(os.getenv("TTS_STREAM_SAMPLE_RATE", "24000"))

    def warm_up(self, speaker_path: Path) -> None:
        print("[SystemTTS] ready via Windows System.Speech")

    def tts_to_file(self, text: str, speaker_wav: str, file_path: str) -> None:
        script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = {int(os.getenv("SYSTEM_TTS_RATE", "0"))}
$synth.Volume = {int(os.getenv("SYSTEM_TTS_VOLUME", "100"))}
$synth.SetOutputToWaveFile({self._ps_quote(str(Path(file_path).resolve()))})
$synth.Speak({self._ps_quote(text)})
$synth.Dispose()
"""
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "Windows SAPI synthesis failed"
            raise RuntimeError(detail)

    def stream(
        self,
        text: str,
        speaker_path: Path,
        emotion: str = "HAPPY",
    ) -> Iterator[bytes]:
        with tempfile.TemporaryDirectory(prefix="liz-system-tts-") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            self.tts_to_file(text, str(speaker_path), str(wav_path))

            cmd = [
                self._ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdout is not None
            try:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk
            finally:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(stderr.strip() or "ffmpeg failed to stream Windows SAPI audio")

    @staticmethod
    def _ffmpeg_exe() -> str:
        return os.getenv("FFMPEG_BIN", "ffmpeg")

    @staticmethod
    def _ps_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
