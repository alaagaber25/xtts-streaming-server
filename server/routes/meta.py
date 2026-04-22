from fastapi import APIRouter, Request

from ..state import get_app_state


router = APIRouter()


@router.get("/studio_speakers")
def get_speakers(request: Request):
    state = get_app_state(request)
    return state.service.get_studio_speakers()


@router.get("/languages")
def get_languages(request: Request):
    state = get_app_state(request)
    return state.service.get_languages()


@router.get("/health")
def health(request: Request):
    state = get_app_state(request)
    return {
        "status": "ok",
        "device": state.service.device.type,
        "model_path": state.service.model_path,
        "speaker_profiles_path": str(state.settings.speaker_profiles_path),
        "speaker_profile_count": len(state.speaker_profiles.list_profiles()),
        "queue_size": state.scheduler.request_queue.qsize(),
        "open_sessions": len(state.scheduler.sessions),
        "batch_collection_window": state.settings.batch_collection_window,
        "max_batch_size": state.settings.max_batch_size,
        "gpu_worker_count": state.settings.gpu_worker_count,
    }
