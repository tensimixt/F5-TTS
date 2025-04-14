# fastapi_server.py
import random
import sys
from importlib.resources import files

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
from typing import Optional

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,  # This is the correct function name
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything
from f5_tts.model import DiT
from hydra.utils import get_class
from omegaconf import OmegaConf

app = FastAPI()

# Load models
model_name = "F5TTS_v1_Base"
model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load vocoder
vocoder = load_vocoder(mel_spec_type, device=device)

# Load model
repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
from cached_path import cached_path
ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))
ema_model = load_model(
    model_cls, 
    model_arc, 
    ckpt_file, 
    mel_spec_type, 
    "", 
    "euler", 
    True, 
    device
)

class TTSRequest(BaseModel):
    ref_text: Optional[str] = ""
    gen_text: str
    remove_silence: bool = False
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    sway_sampling_coef: float = -1
    cfg_strength: float = 2
    nfe_step: int = 32
    speed: float = 1.0
    seed: Optional[int] = None

@app.post("/tts")
async def tts(
    file: UploadFile = File(...),
    ref_text: str = Form(""),
    gen_text: str = Form(...),
    remove_silence: bool = Form(False),
    target_rms: float = Form(0.1),
    cross_fade_duration: float = Form(0.15),
    sway_sampling_coef: float = Form(-1),
    cfg_strength: float = Form(2),
    nfe_step: int = Form(32),
    speed: float = Form(1.0),
    seed: Optional[int] = Form(None)
):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await file.read())
        ref_file = temp_file.name
    
    # Set seed if provided
    if seed is not None:
        seed_everything(seed)
    else:
        seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
    
    # Process reference audio and text
    ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
    
    # Generate audio
    wav, sr, spec = infer_process(
        ref_file,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        mel_spec_type,
        show_info=print,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        device=device,
    )
    
    # Save output files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
        sf.write(wav_file.name, wav, sr)
        if remove_silence:
            remove_silence_for_generated_wav(wav_file.name)
        output_path = wav_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as spec_file:
        save_spectrogram(spec, spec_file.name)
        spec_path = spec_file.name
    
    # Return the audio file
    return FileResponse(
        output_path,
        media_type="audio/wav",
        headers={"X-Seed": str(seed), "X-Spectrogram": spec_path}
    )

@app.get("/")
def read_root():
    return {"message": "F5-TTS API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
