import io
import tempfile

import torch
from fastapi import APIRouter, Request, UploadFile

from speakers.loader import _get_model_speaker_profiles

router = APIRouter()


@router.post("/clone_speaker")
def predict_speaker(wav_file: UploadFile, request: Request):
    temp_audio_name = next(tempfile._get_candidate_names())
    with open(temp_audio_name, "wb") as temp, torch.inference_mode():
        temp.write(io.BytesIO(wav_file.file.read()).getbuffer())
        with request.app.state.model_lock:
            gpt_cond_latent, speaker_embedding = (
                request.app.state.model.get_conditioning_latents(temp_audio_name)
            )
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }


@router.get("/studio_speakers")
def get_speakers(request: Request):
    speaker_profiles = _get_model_speaker_profiles(request.app.state.model)
    speaker_profiles.update(request.app.state.external_speaker_profiles)
    return speaker_profiles

