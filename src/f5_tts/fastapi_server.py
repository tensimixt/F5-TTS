# fastapi_server.py
import random
import sys
import os
from importlib.resources import files

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from typing import Optional

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model.utils import seed_everything
from f5_tts.model import DiT
from hydra.utils import get_class
from omegaconf import OmegaConf

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "f5tts_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    try:
        # Save uploaded file
        temp_ref_file = os.path.join(OUTPUT_DIR, f"ref_{random.randint(0, 100000)}.wav")
        with open(temp_ref_file, "wb") as f:
            f.write(await file.read())
        
        # Set seed if provided
        if seed is not None:
            seed_everything(seed)
        else:
            seed = random.randint(0, sys.maxsize)
            seed_everything(seed)
        
        # Process reference audio and text
        ref_file, ref_text = preprocess_ref_audio_text(temp_ref_file, ref_text)
        
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
        output_filename = f"output_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        sf.write(output_path, wav, sr)
        if remove_silence:
            remove_silence_for_generated_wav(output_path)
        
        spec_filename = f"spec_{seed}.png"
        spec_path = os.path.join(OUTPUT_DIR, spec_filename)
        save_spectrogram(spec, spec_path)
        
        # Return the audio file with appropriate headers
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_filename,
            headers={
                "X-Seed": str(seed),
                "X-Spectrogram": spec_filename,
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
    except Exception as e:
        import traceback
        print(f"Error in TTS endpoint: {e}")
        print(traceback.format_exc())
        return Response(
            content=str(e),
            status_code=500
        )
    finally:
        # Clean up reference file
        if 'temp_ref_file' in locals():
            try:
                os.unlink(temp_ref_file)
            except:
                pass

@app.get("/")
def read_root():
    return {"message": "F5-TTS API is running", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
