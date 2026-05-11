# USB Mic Voice Bridge

This bridge lets a USB microphone on the PC replace manual Serial Monitor typing.

Flow:

```text
USB mic -> local Whisper STT -> ESP serial input -> existing Liz API/audio flow
```

Install dependencies:

```powershell
cd D:\Code\Project\Expressive-Face\TTS-Engine
python -m pip install -r requirements-voice.txt
```

List microphones:

```powershell
python voice_mic_to_serial.py --list-devices
```

List ESP serial ports:

```powershell
python voice_mic_to_serial.py --list-ports
```

Run:

```powershell
python voice_mic_to_serial.py --port COM7
```

Say "Liz" in the sentence, for example:

```text
Liz, say hi.
Liz, I'm happy right now.
```

Notes:

- Close Arduino Serial Monitor before running this script, because only one app can use the COM port.
- The first run downloads the Whisper model. Use `--model tiny.en` for faster but less accurate transcription.
- Use `--wake-word ""` if you want every detected sentence sent to ESP.
- Use `--dry-run` to test microphone transcription without sending to ESP.
