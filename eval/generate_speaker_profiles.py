import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models import xtts as xtts_module
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


# Manual settings.
OVERWRITE = True
DOWNLOAD_BASE = False
FORCE_CPU = False
USE_DEEPSPEED = False

# Set to None for all prefixes, or use values like ["sa", "base"].
SELECTED_PREFIXES = None


EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import core.patches  # noqa: E402,F401

PROMPTS_ROOT = EVAL_ROOT / "prompts"
OUTPUT_ROOT = EVAL_ROOT / "speaker_profiles"
TTS_MODELS_ROOT = REPO_ROOT / "tts_models"

BASE_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
PROMPT_DIRS_BY_PREFIX = {
    "sa": [PROMPTS_ROOT / "sa"],
    "new_sa": [PROMPTS_ROOT / "sa"],
    "eg": [PROMPTS_ROOT / "eg"],
    "base": [PROMPTS_ROOT / "sa", PROMPTS_ROOT / "eg"],
}
SHARED_PROMPTS = sorted(PROMPTS_ROOT.glob("base*.wav"))


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_with_trusted_checkpoint_default(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_with_trusted_checkpoint_default


def _load_audio_without_torchcodec(audio_path):
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T).contiguous(), sample_rate


torchaudio.load = _load_audio_without_torchcodec
xtts_module.torchaudio.load = _load_audio_without_torchcodec


def _json_ready(tensor: torch.Tensor):
    return tensor.detach().cpu().squeeze().half().tolist()


def _base_model_path() -> Path:
    try:
        data_dir = Path(get_user_data_dir("tts"))
    except FileNotFoundError:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise
        data_dir = Path(local_app_data) / "tts"

    model_path = data_dir / BASE_MODEL_NAME.replace("/", "--")
    required = [model_path / "config.json", model_path / "model.pth", model_path / "vocab.json"]
    if all(path.is_file() for path in required):
        return model_path

    if not DOWNLOAD_BASE:
        missing = "\n".join(str(path) for path in required if not path.is_file())
        raise FileNotFoundError(
            "Base XTTS model is not available locally. Missing:\n"
            f"{missing}\nSet DOWNLOAD_BASE = True to let TTS download it."
        )

    print(f"Downloading base XTTS model: {BASE_MODEL_NAME}", flush=True)
    ModelManager().download_model(BASE_MODEL_NAME)
    return model_path


def _discover_models() -> list[tuple[str, Path]]:
    models = []
    for model_dir in sorted(TTS_MODELS_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name not in PROMPT_DIRS_BY_PREFIX:
            print(f"Skipping model without eval prompt mapping: {model_dir.name}", flush=True)
            continue
        models.append((model_dir.name, model_dir))

    models.append(("base", _base_model_path()))
    return models


def _prompt_paths(prefix: str) -> list[Path]:
    prompts = []
    for prompt_dir in PROMPT_DIRS_BY_PREFIX[prefix]:
        if not prompt_dir.is_dir():
            raise FileNotFoundError(f"Missing prompt directory: {prompt_dir}")
        prompts.extend(sorted(prompt_dir.glob("*.wav")))

    prompts.extend(SHARED_PROMPTS)
    unique_prompts = sorted({prompt.resolve(): prompt for prompt in prompts}.values())

    if not unique_prompts:
        raise FileNotFoundError(f"No .wav prompts found for {prefix}")
    return unique_prompts


def _validate_model_dir(model_path: Path) -> tuple[Path, Path]:
    config_path = model_path / "config.json"
    vocab_path = model_path / "vocab.json"
    weights_path = model_path / "model.pth"
    missing = [path for path in (config_path, vocab_path, weights_path) if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Incomplete XTTS model folder: {model_path}\n{missing_text}")
    return config_path, vocab_path


def _load_model(model_path: Path, device: torch.device):
    config_path, vocab_path = _validate_model_dir(model_path)
    print(f"Loading XTTS model from {model_path}", flush=True)

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_path),
        eval=True,
        use_deepspeed=USE_DEEPSPEED and device.type == "cuda",
        vocab_path=str(vocab_path),
    )
    model.to(device)
    return model


def _write_profile(model, prompt_path: Path, output_dir: Path) -> bool:
    speaker_embedding_path = output_dir / "speaker_embedding.json"
    gpt_cond_latent_path = output_dir / "gpt_cond_latent.json"

    if not OVERWRITE and speaker_embedding_path.exists() and gpt_cond_latent_path.exists():
        print(f"Skipping existing profile: {output_dir}", flush=True)
        return False

    print(f"Computing profile from {prompt_path}", flush=True)
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(prompt_path)]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_embedding_path.write_text(
        json.dumps(_json_ready(speaker_embedding), ensure_ascii=True),
        encoding="utf-8",
    )
    gpt_cond_latent_path.write_text(
        json.dumps(_json_ready(gpt_cond_latent), ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Saved profile: {output_dir}", flush=True)
    return True


def main():
    device = torch.device("cpu" if FORCE_CPU else "cuda" if torch.cuda.is_available() else "cpu")
    selected_prefixes = set(SELECTED_PREFIXES or PROMPT_DIRS_BY_PREFIX)

    print(f"Generating eval speaker profiles on {device}", flush=True)
    generated = 0
    skipped = 0

    for prefix, model_path in _discover_models():
        if prefix not in selected_prefixes:
            continue

        model = _load_model(model_path, device)
        try:
            for prompt_path in _prompt_paths(prefix):
                output_dir = OUTPUT_ROOT / prefix / prompt_path.stem
                if _write_profile(model, prompt_path, output_dir):
                    generated += 1
                else:
                    skipped += 1
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        f"Done. Generated {generated} profile(s), skipped {skipped} existing profile(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
