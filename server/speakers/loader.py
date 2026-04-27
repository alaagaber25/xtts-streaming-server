import json
from pathlib import Path
from typing import Dict, List

from core.config import REPO_ROOT, SPEAKER_PROFILES_PATH


def _load_json_file(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_speaker_profile_roots() -> List[Path]:
    candidate_paths = [
        SPEAKER_PROFILES_PATH,
        REPO_ROOT / "speaker_profiles",
    ]

    roots: List[Path] = []
    seen = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            roots.append(resolved)
    return roots


def _load_external_speaker_profiles() -> Dict[str, dict]:
    profiles_by_id: Dict[str, dict] = {}
    leaf_name_to_id: Dict[str, str] = {}
    duplicate_leaf_names = set()

    for root in _get_speaker_profile_roots():
        for embedding_path in sorted(root.rglob("speaker_embedding.json")):
            profile_dir = embedding_path.parent
            gpt_cond_latent_path = profile_dir / "gpt_cond_latent.json"
            if not gpt_cond_latent_path.is_file():
                print(
                    "Skipping speaker profile without gpt_cond_latent.json:",
                    str(profile_dir),
                    flush=True,
                )
                continue

            profile_id = profile_dir.relative_to(root).as_posix()
            profile = {
                "speaker_embedding": _load_json_file(embedding_path),
                "gpt_cond_latent": _load_json_file(gpt_cond_latent_path),
            }
            profiles_by_id[profile_id] = profile

            leaf_name = profile_dir.name
            if leaf_name in leaf_name_to_id and leaf_name_to_id[leaf_name] != profile_id:
                duplicate_leaf_names.add(leaf_name)
            else:
                leaf_name_to_id[leaf_name] = profile_id

    aliased_profiles = dict(profiles_by_id)
    for leaf_name, profile_id in leaf_name_to_id.items():
        if leaf_name in duplicate_leaf_names or leaf_name in aliased_profiles:
            continue
        aliased_profiles[leaf_name] = profiles_by_id[profile_id]

    if aliased_profiles:
        print(
            "Loaded external speaker profiles:",
            ", ".join(sorted(aliased_profiles.keys())),
            flush=True,
        )
    else:
        print("No external speaker profiles found.", flush=True)

    return aliased_profiles


def _get_model_speaker_profiles(model) -> Dict[str, dict]:
    if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
        return {
            speaker: {
                "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in model.speaker_manager.speakers.keys()
        }
    return {}

