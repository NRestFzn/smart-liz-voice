import os
os.add_dll_directory(r"D:\App\ffmpeg-master-latest-win64-gpl-shared\bin")
                     
import base64
import torch

# === HACK BUAT BYPASS PYTORCH 2.6+ SECURITY ===
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ==============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS

app = FastAPI()

# Load model XTTSv2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TTS model on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

class TTSRequest(BaseModel):
    text: str
    speaker_wav: str = "default"

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    try:
        output_path = "output.wav"
        
        # File referensi suara princess lu
        speaker_reference = "voiceover.wav" 
        
        if not os.path.exists(speaker_reference):
            raise HTTPException(status_code=404, detail="File referensi suara 'voiceover.wav' nggak ketemu!")

        # Generate audio
        tts.tts_to_file(
            text=request.text,
            speaker_wav=speaker_reference,
            language="en",
            file_path=output_path
        )

        # Encode ke Base64
        with open(output_path, "rb") as audio_file:
            audio_encoded = base64.b64encode(audio_file.read()).decode('utf-8')

        return {"audio_base64": audio_encoded}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)