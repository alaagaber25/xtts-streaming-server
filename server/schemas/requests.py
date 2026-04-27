from typing import List, Optional

from pydantic import BaseModel


class StreamingInputs(BaseModel):
    speaker_embedding: Optional[List[float]] = None
    gpt_cond_latent: Optional[List[List[float]]] = None
    speaker_profile_id: Optional[str] = None
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"


class TTSInputs(BaseModel):
    speaker_embedding: Optional[List[float]] = None
    gpt_cond_latent: Optional[List[List[float]]] = None
    speaker_profile_id: Optional[str] = None
    text: str
    language: str

