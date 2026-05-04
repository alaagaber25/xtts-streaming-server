import logging
from typing import List, Optional

from pydantic import BaseModel, field_validator

from text.normalization import normalize_tts_text

logger = logging.getLogger(__name__)


class _NormalizedTextInput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def normalize_text(cls, text: str) -> str:
        normalized_text = normalize_tts_text(text)
        logger.info(
            f"Normalized TTS text for {cls.__name__}: \nbefore:{text} \nafter={normalized_text}"
        )
        return normalized_text


class StreamingInputs(_NormalizedTextInput):
    speaker_embedding: Optional[List[float]] = None
    gpt_cond_latent: Optional[List[List[float]]] = None
    speaker_profile_id: Optional[str] = None
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"


class TTSInputs(_NormalizedTextInput):
    speaker_embedding: Optional[List[float]] = None
    gpt_cond_latent: Optional[List[List[float]]] = None
    speaker_profile_id: Optional[str] = None
    language: str
