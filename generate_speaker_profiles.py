from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import re
from pathlib import Path
from typing import Sequence

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from server.speaker_profiles import SpeakerProfileStore
from server.settings import DEFAULT_ENV_FILE, DEFAULT_PROJECT_ROOT


AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
GENERIC_MODEL_DIR_NAMES = {"model", "models", "tts_models", "weights"}
PATH_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PROFILE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
REQUIRED_MODEL_FILES = ("config.json", "model.pth", "vocab.json")


class GeneratorEnvDefaults(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf8",
        extra="ignore",
    )

    custom_model_path: str | None = Field(default=None, validation_alias="CUSTOM_MODEL_PATH")
    references: str | None = Field(default=None, validation_alias="REFERENCES")
    speaker_profiles_path: str | None = Field(default=None, validation_alias="SPEAKER_PROFILES_PATH")


def sanitize_path_component(value: str) -> str:
    normalized = PATH_COMPONENT_PATTERN.sub("_", value.strip()).strip("._-")
    return normalized or "weights"


def sanitize_profile_id(value: str) -> str:
    normalized = PROFILE_ID_PATTERN.sub("_", value.strip()).strip("_-")
    return normalized or "profile"


def resolve_model_dir(weights_path: Path) -> Path:
    resolved = weights_path.expanduser().resolve()
    model_dir = resolved.parent if resolved.is_file() else resolved
    if not model_dir.exists():
        raise FileNotFoundError(f"Weights path '{resolved}' was not found")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Weights path '{model_dir}' is not a directory")

    missing_files = [filename for filename in REQUIRED_MODEL_FILES if not (model_dir / filename).is_file()]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(f"Weights directory '{model_dir}' is missing required files: {missing_list}")

    return model_dir


def derive_weights_name(weights_arg: Path, model_dir: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return sanitize_path_component(explicit_name)

    candidate = model_dir.name
    if weights_arg.is_file() and weights_arg.stem.lower() not in GENERIC_MODEL_DIR_NAMES:
        candidate = weights_arg.stem
    elif candidate.lower() in GENERIC_MODEL_DIR_NAMES and model_dir.parent != model_dir:
        candidate = model_dir.parent.name or candidate

    return sanitize_path_component(candidate)


def collect_reference_files(reference_path: Path) -> list[Path]:
    resolved = reference_path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.exists():
        raise FileNotFoundError(f"Reference path '{resolved}' was not found")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Reference path '{resolved}' is not a directory")

    reference_files = sorted(
        path for path in resolved.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not reference_files:
        supported = ", ".join(sorted(AUDIO_SUFFIXES))
        raise FileNotFoundError(f"No reference audio files with supported suffixes were found in '{resolved}': {supported}")

    return reference_files


def uniquify_profile_id(profile_id: str, used_ids: set[str]) -> str:
    candidate = profile_id
    index = 2
    while candidate in used_ids:
        candidate = f"{profile_id}_{index}"
        index += 1
    used_ids.add(candidate)
    return candidate


def load_env_defaults() -> GeneratorEnvDefaults:
    return GeneratorEnvDefaults()


def resolve_env_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (DEFAULT_PROJECT_ROOT / path).resolve()


def resolve_required_input_path(
    cli_value: str | None,
    env_value: str | None,
    *,
    cli_flag: str,
    env_var: str,
) -> Path:
    if cli_value:
        return Path(cli_value)
    if env_value:
        return resolve_env_path(env_value)
    raise SystemExit(f"Provide {cli_flag} or set {env_var} in {DEFAULT_ENV_FILE}.")


def resolve_output_dir(
    *,
    weights_arg: Path,
    model_dir: Path,
    explicit_weights_name: str | None,
    explicit_output_root: str | None,
    env_speaker_profiles_path: str | None,
) -> tuple[Path, str]:
    if explicit_output_root is not None:
        weights_name = derive_weights_name(weights_arg.expanduser().resolve(), model_dir, explicit_weights_name)
        return Path(explicit_output_root).expanduser().resolve() / weights_name, weights_name

    if explicit_weights_name is None and env_speaker_profiles_path:
        env_output_dir = resolve_env_path(env_speaker_profiles_path)
        if env_output_dir.name.lower() != "speaker_profiles":
            return env_output_dir, sanitize_path_component(env_output_dir.name)

        weights_name = derive_weights_name(weights_arg.expanduser().resolve(), model_dir, explicit_name=None)
        return env_output_dir / weights_name, weights_name

    weights_name = derive_weights_name(weights_arg.expanduser().resolve(), model_dir, explicit_weights_name)
    return Path("speaker_profiles").expanduser().resolve() / weights_name, weights_name


def generate_profiles(
    *,
    model: object,
    reference_files: Sequence[Path],
    output_dir: Path,
    weights_name: str,
    overwrite: bool = False,
) -> list[Path]:
    store = SpeakerProfileStore(output_dir)
    written_paths: list[Path] = []
    batch_ids: set[str] = set()

    for reference_file in reference_files:
        profile_id = uniquify_profile_id(sanitize_profile_id(reference_file.stem), batch_ids)
        if store.has_profile(profile_id) and not overwrite:
            raise FileExistsError(
                f"Speaker profile '{profile_id}' already exists in '{output_dir}'. "
                "Re-run with --overwrite to replace existing profile JSON."
            )
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(str(reference_file))
        store.save_profile(
            profile_id,
            speaker_embedding=speaker_embedding,
            gpt_cond_latent=gpt_cond_latent,
            name=reference_file.stem,
            description=f"Generated from {reference_file.name} using weights '{weights_name}'.",
            overwrite=overwrite,
        )
        written_paths.append(output_dir / f"{profile_id}.json")

    return written_paths


def resolve_runtime_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(device_name)


def patch_xtts_audio_loader(xtts_module) -> None:
    original_load_audio = xtts_module.load_audio
    if getattr(original_load_audio, "_xtts_soundfile_fallback", False):
        return

    def _load_audio_with_fallback(audiopath, sampling_rate):
        try:
            return original_load_audio(audiopath, sampling_rate)
        except ImportError as exc:
            if "TorchCodec" not in str(exc):
                raise

        import soundfile as sf
        import torch
        import torchaudio

        audio, loaded_sample_rate = sf.read(audiopath, dtype="float32", always_2d=False)
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.transpose(0, 1)
            if audio_tensor.size(0) != 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        if loaded_sample_rate != sampling_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, loaded_sample_rate, sampling_rate)

        if torch.any(audio_tensor > 10) or not torch.any(audio_tensor < 0):
            print(f"Error with {audiopath}. Max={audio_tensor.max()} min={audio_tensor.min()}")
        audio_tensor.clip_(-1, 1)
        return audio_tensor

    _load_audio_with_fallback._xtts_soundfile_fallback = True
    xtts_module.load_audio = _load_audio_with_fallback


def load_xtts_model_from_path(model_dir: Path, device, logger):
    import torch
    from server.compat import apply_runtime_compatibility_patches, ensure_torchaudio_compatibility

    apply_runtime_compatibility_patches()
    ensure_torchaudio_compatibility()

    import TTS.tts.models.xtts as xtts_module
    import TTS.utils.io as tts_io
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.layers.xtts.stream_generator import init_stream_support as _init_stream_support  # noqa: F401
    from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

    patch_xtts_audio_loader(xtts_module)

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])

    if "weights_only" in inspect.signature(torch.load).parameters:
        original_load_fsspec = tts_io.load_fsspec

        def _load_fsspec_trusted(path, map_location=None, cache=True, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load_fsspec(path, map_location=map_location, cache=cache, **kwargs)

        tts_io.load_fsspec = _load_fsspec_trusted
        xtts_module.load_fsspec = _load_fsspec_trusted

    logger.info("Loading XTTS")
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    use_deepspeed = device.type == "cuda" and importlib.util.find_spec("deepspeed") is not None
    if device.type == "cuda" and not use_deepspeed:
        logger.warning("Deepspeed not installed; loading XTTS without Deepspeed")
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_dir),
        eval=True,
        use_deepspeed=use_deepspeed,
    )
    model.to(device)
    logger.info("XTTS Loaded.")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repo-compatible XTTS speaker profile JSON files for a specific weights directory."
    )
    parser.add_argument(
        "--weights",
        help="Path to a weights directory containing config.json/model.pth/vocab.json, or a direct path to model.pth.",
    )
    parser.add_argument(
        "--references",
        help="Path to a single reference audio file or a directory of reference audio files.",
    )
    parser.add_argument(
        "--weights-name",
        help="Optional folder name to use under speaker_profiles/. Defaults to a sanitized name derived from the weights path.",
    )
    parser.add_argument(
        "--output-root",
        help="Root directory where weight-specific speaker profile folders are created. When omitted, SPEAKER_PROFILES_PATH from .env is used if it targets a specific folder; otherwise speaker_profiles is used.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device to use when generating latents. Default: auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing speaker profile JSON files when profile IDs already exist.",
    )
    return parser.parse_args()


class _NullLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def getChild(self, _name: str):
        return self


def main() -> None:
    args = parse_args()
    env_defaults = load_env_defaults()
    weights_arg = resolve_required_input_path(
        args.weights,
        env_defaults.custom_model_path,
        cli_flag="--weights",
        env_var="CUSTOM_MODEL_PATH",
    )
    model_dir = resolve_model_dir(weights_arg)
    reference_files = collect_reference_files(
        resolve_required_input_path(
            args.references,
            env_defaults.references,
            cli_flag="--references",
            env_var="REFERENCES",
        )
    )
    output_dir, weights_name = resolve_output_dir(
        weights_arg=weights_arg,
        model_dir=model_dir,
        explicit_weights_name=args.weights_name,
        explicit_output_root=args.output_root,
        env_speaker_profiles_path=env_defaults.speaker_profiles_path,
    )

    import torch

    from server.logging_utils import get_logger

    logger = get_logger("xtts.profile_generator")
    device = resolve_runtime_device(args.device)
    logger.info("Generating speaker profiles for weights '%s' from %s", weights_name, model_dir)
    model = load_xtts_model_from_path(model_dir=model_dir, device=device, logger=logger.getChild("service"))

    with torch.inference_mode():
        written_paths = generate_profiles(
            model=model,
            reference_files=reference_files,
            output_dir=output_dir,
            weights_name=weights_name,
            overwrite=args.overwrite,
        )

    print(f"Generated {len(written_paths)} speaker profile(s) in: {output_dir}")
    print(f"Set SPEAKER_PROFILES_PATH={output_dir} if you want the server to use this weight-specific folder.")
    for profile_path in written_paths:
        print(profile_path)


if __name__ == "__main__":
    main()
