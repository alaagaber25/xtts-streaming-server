from threading import Lock

import torch
from fastapi import FastAPI

import core.patches  # noqa: F401
from core.config import Settings
from models.xtts_loader import load_model
from routers.languages import router as languages_router
from routers.speakers import router as speakers_router
from routers.tts import router as tts_router
from speakers.loader import _load_external_speaker_profiles

settings = Settings()

torch.set_num_threads(settings.num_threads)

xtts_config, model = load_model()
external_speaker_profiles = _load_external_speaker_profiles()

print("Running XTTS Server ...", flush=True)

app = FastAPI(
    title="XTTS Streaming server",
    description="""XTTS Streaming server""",
    version="0.0.1",
    docs_url="/",
)
app.state.xtts_config = xtts_config
app.state.model = model
app.state.model_lock = Lock()
app.state.external_speaker_profiles = external_speaker_profiles
app.include_router(tts_router)
app.include_router(speakers_router)
app.include_router(languages_router)
