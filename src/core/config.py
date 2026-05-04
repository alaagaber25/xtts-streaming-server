import os
from functools import cache
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    use_cpu: bool = False
    num_threads: int = Field(default_factory=lambda: os.cpu_count() or 1)
    use_deepspeed: bool = False
    custom_model_path: str | None = None
    speaker_profiles_path: str = "speaker_profiles"
    save_tts_outputs: bool = False
    tts_outputs_path: str = "tts_outputs"
    xtts_port: int = 8004

    @property
    def resolved_custom_model_path(self) -> Path | None:
        if not self.custom_model_path:
            return None
        return _resolve_repo_path(self.custom_model_path)

    @property
    def resolved_speaker_profiles_path(self) -> Path:
        return _resolve_repo_path(self.speaker_profiles_path)

    @property
    def resolved_tts_outputs_path(self) -> Path:
        return _resolve_repo_path(self.tts_outputs_path)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu" if self.use_cpu else "cuda")


@cache
def Settings():
    return _Settings()


settings = Settings()

USE_CPU = settings.use_cpu
NUM_THREADS = settings.num_threads
USE_DEEPSPEED = settings.use_deepspeed
CUSTOM_MODEL_PATH = settings.resolved_custom_model_path
SPEAKER_PROFILES_PATH = settings.resolved_speaker_profiles_path
SAVE_TTS_OUTPUTS = settings.save_tts_outputs
TTS_OUTPUTS_PATH = settings.resolved_tts_outputs_path
XTTS_PORT = settings.xtts_port
DEVICE = settings.device

if DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")
