from fastapi import APIRouter, HTTPException, Request

from ..state import get_app_state


router = APIRouter()


@router.get("/speaker_profiles")
def list_speaker_profiles(request: Request):
    state = get_app_state(request)
    return state.speaker_profiles.list_profiles()


@router.get("/speaker_profiles/{profile_id}")
def get_speaker_profile(profile_id: str, request: Request):
    state = get_app_state(request)
    try:
        profile = state.speaker_profiles.get_profile(profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Speaker profile '{profile_id}' was not found")
    return profile


@router.delete("/speaker_profiles/{profile_id}")
def delete_speaker_profile(profile_id: str, request: Request):
    state = get_app_state(request)
    try:
        deleted = state.speaker_profiles.delete_profile(profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Speaker profile '{profile_id}' was not found")
    return {"status": "ok", "speaker_profile_id": profile_id}
