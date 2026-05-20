import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models import xtts as xtts_module
from TTS.tts.models.xtts import Xtts


SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent
sys.path.insert(0, str(SERVER_ROOT))
import core.patches  # noqa: E402,F401

DEFAULT_MODEL_PATH = REPO_ROOT / "tts_models" / "eg_lora"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "speaker_profiles" / "eg_lora"
PROMPTS_ROOT = SERVER_ROOT / "prompts"

DEFAULT_PROFILES = {
    "saied": PROMPTS_ROOT / "saied.wav",
    "nadia": PROMPTS_ROOT / "nadia.wav",
    "shahad": PROMPTS_ROOT / "shahad.wav",
    "abdo": PROMPTS_ROOT / "abdo.wav",
    "fahd": PROMPTS_ROOT / "fahd.wav",
}

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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create XTTS speaker profile JSON files for one checkpoint."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="XTTS model folder containing config.json, model.pth, and vocab.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-speaker profile folders will be written.",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=list(DEFAULT_PROFILES),
        help=f"Profile names to generate. Defaults to: {', '.join(DEFAULT_PROFILES)}",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Generate on CPU even when CUDA is available.",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use DeepSpeed while loading the checkpoint when CUDA is available.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip profiles that already contain both JSON files.",
    )
    return parser.parse_args()


def _validate_model_dir(model_path: Path) -> tuple[Path, Path]:
    config_path = model_path / "config.json"
    vocab_path = model_path / "vocab.json"
    weights_path = model_path / "model.pth"
    missing = [
        path for path in (config_path, vocab_path, weights_path) if not path.is_file()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Incomplete XTTS model folder:\n{missing_text}")
    return config_path, vocab_path


def _selected_profiles(profile_names: list[str]) -> list[tuple[str, Path]]:
    unknown_names = sorted(set(profile_names) - set(DEFAULT_PROFILES))
    if unknown_names:
        known_names = ", ".join(DEFAULT_PROFILES)
        unknown_text = ", ".join(unknown_names)
        raise ValueError(f"Unknown profile(s): {unknown_text}. Known profiles: {known_names}")

    profiles = [(name, DEFAULT_PROFILES[name]) for name in profile_names]
    missing = [path for _, path in profiles if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prompt audio file(s):\n{missing_text}")
    return profiles


def _load_model(model_path: Path, device: torch.device, use_deepspeed: bool):
    config_path, vocab_path = _validate_model_dir(model_path)

    if use_deepspeed and device.type != "cuda":
        print("Ignoring --use-deepspeed because CUDA is not available.", flush=True)
        use_deepspeed = False

    print(f"Loading XTTS model from {model_path}", flush=True)
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_path),
        eval=True,
        use_deepspeed=use_deepspeed,
        vocab_path=str(vocab_path),
    )
    model.to(device)
    return model


def _write_profile(
    model,
    name: str,
    prompt_path: Path,
    output_root: Path,
    overwrite: bool,
) -> bool:
    output_dir = output_root / name
    speaker_embedding_path = output_dir / "speaker_embedding.json"
    gpt_cond_latent_path = output_dir / "gpt_cond_latent.json"

    if (
        not overwrite
        and speaker_embedding_path.is_file()
        and gpt_cond_latent_path.is_file()
    ):
        print(f"Skipping existing profile: {output_dir}", flush=True)
        return False

    print(f"Computing {name} from {prompt_path}", flush=True)
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

    print(f"Saved speaker_profile_id: {output_root.name}/{name}", flush=True)
    return True


def main():
    args = _parse_args()
    profiles = _selected_profiles(args.profiles)
    device = torch.device(
        "cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda"
    )

    print(f"Generating speaker profiles on {device}", flush=True)
    model = _load_model(args.model_path, device, args.use_deepspeed)

    generated = 0
    skipped = 0
    try:
        for name, prompt_path in profiles:
            if _write_profile(
                model,
                name,
                prompt_path,
                args.output_root,
                overwrite=not args.no_overwrite,
            ):
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
