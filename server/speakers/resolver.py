from pathlib import Path
from typing import List, Optional, Tuple

import torch
from fastapi import HTTPException

from speakers.loader import _get_model_speaker_profiles


def _resolve_speaker_profile(speaker_profile_id: str, model, external_speaker_profiles):
    normalized_speaker_profile_id = speaker_profile_id.strip().replace("\\", "/")
    if not normalized_speaker_profile_id:
        raise HTTPException(
            status_code=422, detail="speaker_profile_id must be a non-empty string"
        )

    profiles = _get_model_speaker_profiles(model)
    profiles.update(external_speaker_profiles)
    profile = profiles.get(normalized_speaker_profile_id)
    if profile is None:
        profile = profiles.get(Path(normalized_speaker_profile_id).name)
    if profile is None:
        available_profiles = sorted(profiles.keys())
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown speaker_profile_id '{normalized_speaker_profile_id}'",
                "available_speaker_profiles": available_profiles,
            },
        )
    return profile


def _resolve_conditioning_inputs(
    *,
    model,
    external_speaker_profiles,
    speaker_profile_id: Optional[str],
    speaker_embedding: Optional[List[float]],
    gpt_cond_latent: Optional[List[List[float]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if speaker_profile_id:
        profile = _resolve_speaker_profile(
            speaker_profile_id, model, external_speaker_profiles
        )
        speaker_embedding = profile["speaker_embedding"]
        gpt_cond_latent = profile["gpt_cond_latent"]

    if speaker_embedding is None or gpt_cond_latent is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide either speaker_profile_id or both speaker_embedding "
                "and gpt_cond_latent."
            ),
        )

    resolved_speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).unsqueeze(-1)
    resolved_gpt_cond_latent = (
        torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    )
    return resolved_speaker_embedding, resolved_gpt_cond_latent

