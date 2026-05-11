import os
os.add_dll_directory(r"E:\Apps\ffmpeg-master-latest-win64-gpl-shared\bin")

import base64
import torch

# ================================
# FIX PyTorch load (XTTS issue)
# ================================
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS

app = FastAPI()

# ================================
# SET DEVICE (GPU)
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("========== GPU CHECK ==========")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
print("================================")

# ================================
# LOAD MODEL (GPU)
# ================================
tts = TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=(device == "cuda")
).to(device)

print(f"✅ XTTS loaded on {device}")

# ================================
# REQUEST MODEL
# ================================
class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = "default"

# ================================
# API ENDPOINT
# ================================
@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    try:
        os.makedirs("./output_voice", exist_ok=True)
        output_path = "./output_voice/output.wav"

        default_reference = "./sample_voice/VO_Lumine.wav"
        speaker_reference = (
            default_reference
            if request.speaker_wav in ("", "default")
            else request.speaker_wav
        )

        if not os.path.exists(speaker_reference):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"File referensi suara '{speaker_reference}' nggak ketemu! "
                    f"(default: '{default_reference}')"
                ),
            )

        print(f"\n🧠 Generating TTS on {device}...")
        
        # ================================
        # GENERATE AUDIO
        # ================================
        tts.tts_to_file(
            text=request.text,
            speaker_wav=speaker_reference,
            language="en",
            file_path=output_path
        )

        print("✅ Audio generated")

        # ================================
        # ENCODE BASE64
        # ================================
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "audio_base64": audio_base64
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)