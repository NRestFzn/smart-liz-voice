from __future__ import annotations

import argparse
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from serial import Serial
from serial.tools import list_ports


DEFAULT_WAKE_WORD = os.getenv("VOICE_WAKE_WORD", "liz")


@dataclass
class VadConfig:
    sample_rate: int
    frame_ms: int
    preroll_ms: int
    min_speech_ms: int
    silence_ms: int
    max_seconds: float
    threshold: float
    threshold_multiplier: float


def list_audio_devices() -> None:
    print(sd.query_devices())


def list_serial_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        print(f"{port.device}: {port.description}")


def resolve_audio_device(value: str | None) -> int | str | None:
    if value is None or value.strip() == "":
        return None

    value = value.strip()
    if value.isdigit():
        return int(value)

    return value


def choose_serial_port(requested: str | None) -> str:
    if requested:
        return requested

    ports = list(list_ports.comports())
    candidates = [
        port
        for port in ports
        if re.search(
            r"(usb|uart|cp210|ch340|wch|silicon|esp|arduino)",
            f"{port.device} {port.description}",
            re.IGNORECASE,
        )
    ]

    if len(candidates) == 1:
        return candidates[0].device

    print("Could not choose ESP serial port automatically.")
    list_serial_ports()
    raise SystemExit("Run again with --port COMx, for example: --port COM7")


def serial_echo_worker(ser: Serial, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = ser.readline()
        except Exception:
            break

        if line:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[ESP] {text}")
        else:
            time.sleep(0.02)


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))


def calibrate_noise_floor(stream: sd.InputStream, frame_size: int, seconds: float) -> float:
    values: list[float] = []
    deadline = time.time() + seconds

    print("Calibrating mic noise... stay quiet for a moment.")
    while time.time() < deadline:
        data, overflowed = stream.read(frame_size)
        if overflowed:
            continue
        values.append(rms(data.reshape(-1)))

    if not values:
        return 0.01

    noise = float(np.median(np.array(values, dtype=np.float32)))
    print(f"Noise floor: {noise:.5f}")
    return noise


def record_utterance(stream: sd.InputStream, cfg: VadConfig) -> np.ndarray:
    frame_size = int(cfg.sample_rate * cfg.frame_ms / 1000)
    preroll_frames = max(1, cfg.preroll_ms // cfg.frame_ms)
    min_speech_frames = max(1, cfg.min_speech_ms // cfg.frame_ms)
    silence_frames = max(1, cfg.silence_ms // cfg.frame_ms)
    max_frames = max(1, int(cfg.max_seconds * 1000 / cfg.frame_ms))

    preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)

    while True:
        data, overflowed = stream.read(frame_size)
        if overflowed:
            continue

        frame = data.reshape(-1).astype(np.float32, copy=True)
        preroll.append(frame)

        if rms(frame) < cfg.threshold:
            continue

        print("Speech detected...")
        frames = list(preroll)
        silent_count = 0

        while len(frames) < max_frames:
            data, overflowed = stream.read(frame_size)
            if overflowed:
                continue

            frame = data.reshape(-1).astype(np.float32, copy=True)
            frames.append(frame)

            if rms(frame) < cfg.threshold:
                silent_count += 1
                if silent_count >= silence_frames and len(frames) >= min_speech_frames:
                    break
            else:
                silent_count = 0

        return np.concatenate(frames)


def transcribe(model: WhisperModel, audio: np.ndarray, language: str) -> str:
    segments, _info = model.transcribe(
        audio,
        language=language or None,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()
    return re.sub(r"\s+", " ", text)


def should_send(text: str, wake_word: str) -> bool:
    if not wake_word:
        return True
    return re.search(rf"\b{re.escape(wake_word)}\b", text, re.IGNORECASE) is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Listen to a USB microphone, transcribe speech, and send the text to the ESP serial chat input."
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit.")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit.")
    parser.add_argument("--device", help="Audio device index or name. Use --list-devices to find it.")
    parser.add_argument("--port", help="ESP serial port, for example COM7.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--model", default=os.getenv("VOICE_STT_MODEL", "base.en"))
    parser.add_argument("--language", default=os.getenv("VOICE_STT_LANGUAGE", "en"))
    parser.add_argument("--wake-word", default=DEFAULT_WAKE_WORD, help='Default: "liz". Use --wake-word "" to disable.')
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--preroll-ms", type=int, default=350)
    parser.add_argument("--min-speech-ms", type=int, default=450)
    parser.add_argument("--silence-ms", type=int, default=950)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.0, help="Manual RMS threshold. 0 = auto calibration.")
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--calibrate-seconds", type=float, default=1.2)
    parser.add_argument("--serial-wait", type=float, default=2.5, help="Wait after opening serial because ESP may reset.")
    parser.add_argument("--dry-run", action="store_true", help="Transcribe only, do not send to serial.")
    parser.add_argument("--no-echo", action="store_true", help="Do not print ESP serial output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    if args.list_ports:
        list_serial_ports()
        return 0

    audio_device = resolve_audio_device(args.device)
    serial_port = choose_serial_port(args.port) if not args.dry_run else None

    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    stop_event = threading.Event()
    ser: Serial | None = None
    if not args.dry_run:
        assert serial_port is not None
        print(f"Opening ESP serial: {serial_port} @ {args.baud}")
        ser = Serial(serial_port, args.baud, timeout=0.1, write_timeout=2)
        time.sleep(args.serial_wait)

        if not args.no_echo:
            threading.Thread(target=serial_echo_worker, args=(ser, stop_event), daemon=True).start()

    cfg = VadConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        preroll_ms=args.preroll_ms,
        min_speech_ms=args.min_speech_ms,
        silence_ms=args.silence_ms,
        max_seconds=args.max_seconds,
        threshold=args.threshold,
        threshold_multiplier=args.threshold_multiplier,
    )

    try:
        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(cfg.sample_rate * cfg.frame_ms / 1000),
            device=audio_device,
        ) as stream:
            if cfg.threshold <= 0:
                noise = calibrate_noise_floor(stream, int(cfg.sample_rate * cfg.frame_ms / 1000), args.calibrate_seconds)
                cfg.threshold = max(0.01, noise * cfg.threshold_multiplier)

            print(f"Listening. Threshold: {cfg.threshold:.5f}")
            if args.wake_word:
                print(f'Say "{args.wake_word}" in the sentence to send it to Liz.')

            while True:
                audio = record_utterance(stream, cfg)
                text = transcribe(model, audio, args.language)
                if not text:
                    continue

                print(f"[YOU] {text}")
                if not should_send(text, args.wake_word.strip()):
                    print("[ignored] wake word not found")
                    continue

                if ser is not None:
                    ser.write((text + "\n").encode("utf-8"))
                    ser.flush()
                    print("[sent to ESP]")

    except KeyboardInterrupt:
        print("\nStopping voice bridge.")
        return 0
    finally:
        stop_event.set()
        if ser is not None:
            ser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
