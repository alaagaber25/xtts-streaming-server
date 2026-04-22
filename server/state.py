import logging
from dataclasses import dataclass

from fastapi import Request

from .runtime import InferenceScheduler
from .speaker_profiles import SpeakerProfileStore
from .settings import Settings
from .xtts_service import XTTSService


@dataclass
class AppState:
    settings: Settings
    service: XTTSService
    scheduler: InferenceScheduler
    speaker_profiles: SpeakerProfileStore
    logger: logging.Logger


def get_app_state(request: Request) -> AppState:
    return request.app.state.xtts_state
