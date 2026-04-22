import os
import logging
import re
from pathlib import Path

import torch
from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = DEFAULT_PROJECT_ROOT / ".env"
PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf8",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    num_threads: PositiveInt = Field(default_factory=lambda: os.cpu_count() or 1, validation_alias="NUM_THREADS")
    use_cpu: bool = Field(default=False, validation_alias="USE_CPU")
    custom_model_path: Path = Field(default=Path("/app/tts_models"), validation_alias="CUSTOM_MODEL_PATH")
    speaker_profiles_path: Path = Field(
        default=DEFAULT_PROJECT_ROOT / "speaker_profiles",
        validation_alias="SPEAKER_PROFILES_PATH",
    )
    default_speaker_profile_id: str | None = Field(default=None, validation_alias="DEFAULT_SPEAKER_PROFILE_ID")
    batch_collection_window: PositiveFloat = Field(default=0.03, validation_alias="BATCH_COLLECTION_WINDOW")
    max_batch_size: PositiveInt = Field(default=4, validation_alias="MAX_BATCH_SIZE")
    queue_poll_interval: PositiveFloat = Field(default=0.005, validation_alias="QUEUE_POLL_INTERVAL")
    gpu_worker_count: PositiveInt = Field(default=1, validation_alias="GPU_WORKER_COUNT")

    @field_validator("speaker_profiles_path", mode="before")
    @classmethod
    def resolve_speaker_profiles_path(cls, value: Path | str | None) -> Path:
        resolved = value if value is not None else DEFAULT_PROJECT_ROOT / "speaker_profiles"
        return _resolve_path_env(resolved, DEFAULT_PROJECT_ROOT)

    @field_validator("custom_model_path", mode="before")
    @classmethod
    def resolve_custom_model_path(cls, value: Path | str) -> Path:
        return _resolve_path_env(value, DEFAULT_PROJECT_ROOT)

    @field_validator("default_speaker_profile_id", mode="before")
    @classmethod
    def normalize_default_speaker_profile_id(cls, value: str | None) -> str | None:
        normalized = _normalize_optional_env(value)
        if normalized is None:
            return None
        if not PROFILE_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "DEFAULT_SPEAKER_PROFILE_ID must contain only letters, numbers, underscores, or hyphens"
            )
        return normalized


def load_settings(logger: logging.Logger | None = None) -> Settings:
    settings = Settings()
    if settings.gpu_worker_count != 1:
        if logger is not None:
            logger.warning(
                "GPU_WORKER_COUNT > 1 is not supported with the shared XTTS model instance; using 1 worker."
            )
        settings = settings.model_copy(update={"gpu_worker_count": 1})
    return settings


def resolve_device(settings: Settings, logger: logging.Logger | None = None) -> torch.device:
    torch.set_num_threads(settings.num_threads)
    if settings.use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if logger is not None:
        logger.warning("CUDA unavailable; falling back to CPU.")
    return torch.device("cpu")


def _normalize_optional_env(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_path_env(value: Path | str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
