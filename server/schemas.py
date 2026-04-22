from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


PROFILE_ID_REGEX = r"^[A-Za-z0-9_-]+$"
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ProfileId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, pattern=PROFILE_ID_REGEX)]
PositiveStreamChunkSize = Annotated[int, Field(gt=0)]


class XTTSBaseModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class SpeakerProfileSummary(XTTSBaseModel):
    id: ProfileId
    name: NonEmptyStr
    description: str | None = None
    created_at: NonEmptyStr

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return cls._normalize_optional_text(value)


class SpeakerProfile(SpeakerProfileSummary):
    speaker_embedding: list[float]
    gpt_cond_latent: list[list[float]]

    def to_summary(self) -> SpeakerProfileSummary:
        return SpeakerProfileSummary(
            id=self.id,
            name=self.name,
            description=self.description,
            created_at=self.created_at,
        )


class BaseTTSInputs(XTTSBaseModel):
    speaker_profile_id: ProfileId | None = None
    speaker_embedding: list[float] | None = None
    gpt_cond_latent: list[list[float]] | None = None
    text: NonEmptyStr
    language: NonEmptyStr

    @field_validator("speaker_profile_id", mode="before")
    @classmethod
    def normalize_optional_profile_id(cls, value: str | None) -> str | None:
        return cls._normalize_optional_text(value)

    @model_validator(mode="after")
    def validate_voice_source(self) -> "BaseTTSInputs":
        has_embedding = self.speaker_embedding is not None
        has_latent = self.gpt_cond_latent is not None
        if has_embedding != has_latent:
            raise ValueError(
                "speaker_embedding and gpt_cond_latent must be provided together"
            )

        return self


class StreamingInputs(BaseTTSInputs):
    add_wav_header: bool = True
    stream_chunk_size: PositiveStreamChunkSize = 20


class TTSInputs(BaseTTSInputs):
    pass
