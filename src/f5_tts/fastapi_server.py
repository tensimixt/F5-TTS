from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
import torch
import tempfile
import os
import soundfile as sf
import uvicorn
from pydantic import BaseModel
import numpy as np
from typing import Optional
import io

# Import the necessary functions from F5-TTS
from f5_tts.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text, 
    infer_process, remove_silence_from_audio
)

app = FastAPI(title="F5-TTS API", description="API for F5-TTS text-to-speech synthesis")

# Global variables to store models
tts_model = None
vocoder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TTSRequest(BaseModel):
    ref_text: str
    gen_text: str
    remove_silence: bool = True
    speed: float = 1.0
    nfe_step: int = 32
    model_name: str = "F5TTS_v1_Base"

@app.on_event("startup")
async def startup_event():
    global tts_model, vocoder
    # Load vocoder
    vocoder = load_vocoder()
    # Load TTS model (default to F5TTS_v1_Base)
    tts_model = load_model("F5TTS_v1_Base")

@app.post("/tts/")
async def generate_tts(
    ref_audio: UploadFile = File(...),
    request: TTSRequest = None,
    ref_text: str = Form(None),
    gen_text: str = Form(None),
    remove_silence: bool = Form(True),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    model_name: str = Form("F5TTS_v1_Base")
):
    # Use either form data or JSON request
    if request:
        ref_text = request.ref_text
        gen_text = request.gen_text
        remove_silence = request.remove_silence
        speed = request.speed
        nfe_step = request.nfe_step
        model_name = request.model_name
    
    # Read reference audio
    audio_data = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name
    
    # Preprocess reference audio and text
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        temp_audio_path, ref_text, device=device
    )
    
    # Generate audio
    audio_output = infer_process(
        ref_audio_processed, 
        ref_text_processed, 
        gen_text, 
        tts_model, 
        vocoder, 
        nfe_step=nfe_step, 
        speed=speed
    )
    
    # Remove silence if requested
    if remove_silence:
        audio_output = remove_silence_from_audio(audio_output)
    
    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
        sf.write(temp_output.name, audio_output, 24000)
        output_path = temp_output.name
    
    # Clean up the temporary reference audio file
    os.unlink(temp_audio_path)
    
    # Return the audio file
    return FileResponse(
        output_path, 
        media_type="audio/wav", 
        filename="generated_speech.wav",
        background=BackgroundTasks(lambda: os.unlink(output_path))
    )

@app.get("/models/")
async def list_models():
    return {
        "available_models": [
            "F5TTS_v1_Base",
            "F5TTS_Base",
            "E2TTS_Base"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, workers=1)
