import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models import xtts as xtts_module
from TTS.tts.models.xtts import Xtts

# Edit these values manually before running the script.
MODEL_PATH = Path(r"F:\VOOM-AI\GitHubs\xtts-server\xtts\tts_models\new_sa")
SPEAKER_AUDIO_PATH = Path(
    r"F:\VOOM-AI\GitHubs\xtts-server\xtts\src\prompts\shahad_1.wav"
)
OUTPUT_PROFILE_DIR = Path(
    r"F:\VOOM-AI\GitHubs\xtts-server\xtts\speaker_profiles\sa\nada"
)

USE_CUDA = True
USE_DEEPSPEED = False
OVERWRITE = True


def _load_audio_without_torchcodec(audio_path):
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T).contiguous(), sample_rate


torchaudio.load = _load_audio_without_torchcodec
xtts_module.torchaudio.load = _load_audio_without_torchcodec


def _json_ready(tensor: torch.Tensor):
    return tensor.detach().cpu().squeeze().half().tolist()


def _validate_paths() -> tuple[Path, Path, Path]:
    config_path = MODEL_PATH / "config.json"
    vocab_path = MODEL_PATH / "vocab.json"
    model_weights_path = MODEL_PATH / "model.pth"

    missing_paths = [
        path
        for path in (config_path, vocab_path, model_weights_path, SPEAKER_AUDIO_PATH)
        if not path.is_file()
    ]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise RuntimeError(f"Missing required file(s):\n{missing}")

    speaker_embedding_path = OUTPUT_PROFILE_DIR / "speaker_embedding.json"
    gpt_cond_latent_path = OUTPUT_PROFILE_DIR / "gpt_cond_latent.json"
    if not OVERWRITE and (
        speaker_embedding_path.exists() or gpt_cond_latent_path.exists()
    ):
        raise RuntimeError(
            f"Speaker profile already exists: {OUTPUT_PROFILE_DIR}\n"
            "Set OVERWRITE = True to replace it."
        )
    if speaker_embedding_path.exists() or gpt_cond_latent_path.exists():
        print(f"Overwriting existing speaker profile: {OUTPUT_PROFILE_DIR}")

    return config_path, vocab_path, model_weights_path


def main():
    config_path, vocab_path, _ = _validate_paths()
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

    print(f"Loading XTTS model from {MODEL_PATH}...")
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(MODEL_PATH),
        eval=True,
        use_deepspeed=USE_DEEPSPEED and device.type == "cuda",
        vocab_path=str(vocab_path),
    )
    model.to(device)

    print(f"Computing speaker latents from {SPEAKER_AUDIO_PATH}...")
    with torch.inference_mode():
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(SPEAKER_AUDIO_PATH)]
        )

    OUTPUT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PROFILE_DIR / "speaker_embedding.json").write_text(
        json.dumps(_json_ready(speaker_embedding), ensure_ascii=True),
        encoding="utf-8",
    )
    (OUTPUT_PROFILE_DIR / "gpt_cond_latent.json").write_text(
        json.dumps(_json_ready(gpt_cond_latent), ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Saved speaker profile: {OUTPUT_PROFILE_DIR}")
    print(
        "speaker_profile_id:",
        OUTPUT_PROFILE_DIR.parent.name + "/" + OUTPUT_PROFILE_DIR.name,
    )


if __name__ == "__main__":
    main()
