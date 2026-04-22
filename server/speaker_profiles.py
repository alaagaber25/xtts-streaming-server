import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .schemas import PROFILE_ID_REGEX, SpeakerProfile


PROFILE_ID_PATTERN = re.compile(PROFILE_ID_REGEX)
LEGACY_SPEAKER_EMBEDDING_FILE = "speaker_embedding.json"
LEGACY_GPT_COND_LATENT_FILE = "gpt_cond_latent.json"


class SpeakerProfileStore:
    def __init__(self, storage_dir: Path | str, logger: logging.Logger | None = None) -> None:
        self.storage_dir = Path(storage_dir)
        self.logger = logger or logging.getLogger("xtts.speaker_profiles")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[Dict[str, Any]]:
        profiles_by_id: Dict[str, Dict[str, Any]] = {}

        for profile_path in sorted(self.storage_dir.glob("*.json")):
            try:
                profile = self._read_json_profile(profile_path)
            except ValueError as exc:
                self.logger.warning("Skipping invalid speaker profile '%s': %s", profile_path.name, exc)
                continue
            profiles_by_id[profile.id] = profile.to_summary().model_dump()

        for profile_dir in sorted(path for path in self.storage_dir.iterdir() if path.is_dir()):
            try:
                profile = self._read_legacy_profile_dir(profile_dir)
            except ValueError as exc:
                self.logger.warning("Skipping invalid speaker profile '%s': %s", profile_dir.name, exc)
                continue
            if profile is None or profile.id in profiles_by_id:
                continue
            profiles_by_id[profile.id] = profile.to_summary().model_dump()

        profiles = list(profiles_by_id.values())
        profiles.sort(key=lambda item: (item["name"] or item["id"]).lower())
        return profiles

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        profile = self._load_profile(profile_id)
        if profile is None:
            return None
        return profile.model_dump()

    def has_profile(self, profile_id: str) -> bool:
        return self._load_profile(profile_id) is not None

    def save_profile(
        self,
        profile_id: str,
        *,
        speaker_embedding: Any,
        gpt_cond_latent: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        validated_id = self._validate_profile_id(profile_id)
        profile_path = self._profile_path(validated_id)
        if self.has_profile(validated_id) and not overwrite:
            raise FileExistsError(f"Speaker profile '{validated_id}' already exists")

        existing_profile = self._load_profile(validated_id)
        created_at = (
            existing_profile.created_at
            if existing_profile is not None
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        normalized_name = (name or "").strip() or validated_id
        normalized_description = self._normalize_optional_text(description)
        profile = SpeakerProfile(
            id=validated_id,
            name=normalized_name,
            description=normalized_description,
            created_at=created_at,
            speaker_embedding=self._to_list(speaker_embedding),
            gpt_cond_latent=self._to_list(gpt_cond_latent),
        )

        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf8")
        return profile.model_dump()

    def delete_profile(self, profile_id: str) -> bool:
        validated_id = self._validate_profile_id(profile_id)
        profile_path = self._profile_path(validated_id)
        if profile_path.exists():
            profile_path.unlink()
            return True

        legacy_dir = self._legacy_profile_dir(validated_id)
        if self._read_legacy_profile_dir(legacy_dir) is None:
            return False

        for legacy_path in self._legacy_profile_files(legacy_dir):
            if legacy_path.exists():
                legacy_path.unlink()
        if legacy_dir.exists() and not any(legacy_dir.iterdir()):
            legacy_dir.rmdir()
        return True

    def _profile_path(self, profile_id: str) -> Path:
        validated_id = self._validate_profile_id(profile_id)
        return self.storage_dir / f"{validated_id}.json"

    def _legacy_profile_dir(self, profile_id: str) -> Path:
        validated_id = self._validate_profile_id(profile_id)
        return self.storage_dir / validated_id

    def _legacy_profile_files(self, profile_dir: Path) -> tuple[Path, Path]:
        return (
            profile_dir / LEGACY_SPEAKER_EMBEDDING_FILE,
            profile_dir / LEGACY_GPT_COND_LATENT_FILE,
        )

    def _load_profile(self, profile_id: str) -> SpeakerProfile | None:
        profile_path = self._profile_path(profile_id)
        if profile_path.exists():
            return self._read_json_profile(profile_path)

        return self._read_legacy_profile_dir(self._legacy_profile_dir(profile_id))

    def _read_json_profile(self, profile_path: Path) -> SpeakerProfile:
        try:
            return SpeakerProfile.model_validate_json(profile_path.read_text(encoding="utf8"))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Speaker profile file '{profile_path.name}' is invalid") from exc

    def _read_legacy_profile_dir(self, profile_dir: Path) -> SpeakerProfile | None:
        if not profile_dir.exists() or not profile_dir.is_dir():
            return None

        speaker_embedding_path, gpt_cond_latent_path = self._legacy_profile_files(profile_dir)
        legacy_files_present = speaker_embedding_path.exists() or gpt_cond_latent_path.exists()
        if not legacy_files_present:
            return None
        if not speaker_embedding_path.exists() or not gpt_cond_latent_path.exists():
            raise ValueError(f"Speaker profile directory '{profile_dir.name}' is missing legacy latent files")

        validated_id = self._validate_profile_id(profile_dir.name)
        try:
            speaker_embedding = json.loads(speaker_embedding_path.read_text(encoding="utf8"))
            gpt_cond_latent = json.loads(gpt_cond_latent_path.read_text(encoding="utf8"))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Speaker profile directory '{profile_dir.name}' is invalid") from exc

        latest_mtime = max(speaker_embedding_path.stat().st_mtime, gpt_cond_latent_path.stat().st_mtime)
        created_at = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            return SpeakerProfile(
                id=validated_id,
                name=validated_id,
                description=None,
                created_at=created_at,
                speaker_embedding=speaker_embedding,
                gpt_cond_latent=gpt_cond_latent,
            )
        except ValidationError as exc:
            raise ValueError(f"Speaker profile directory '{profile_dir.name}' is invalid") from exc

    def _validate_profile_id(self, profile_id: str) -> str:
        normalized = profile_id.strip()
        if not normalized:
            raise ValueError("speaker_profile_id cannot be empty")
        if not PROFILE_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "speaker_profile_id must contain only letters, numbers, underscores, or hyphens"
            )
        return normalized

    def _normalize_optional_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def _to_list(self, value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "squeeze"):
            value = value.squeeze()
        if hasattr(value, "half"):
            value = value.half()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
