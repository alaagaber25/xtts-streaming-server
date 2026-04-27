import os
from pathlib import Path

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

from core.config import Settings

settings = Settings()


def _resolve_model_path():
    custom_model_path = settings.resolved_custom_model_path
    if custom_model_path is not None:
        config_path = custom_model_path / "config.json"
        model_path = custom_model_path / "model.pth"
        if not custom_model_path.is_dir():
            raise RuntimeError(
                "CUSTOM_MODEL_PATH does not exist or is not a directory: "
                f"{custom_model_path}"
            )
        if not config_path.is_file() or not model_path.is_file():
            raise RuntimeError(
                "CUSTOM_MODEL_PATH must point to an exact XTTS model directory containing "
                f"config.json and model.pth. Got: {custom_model_path}"
            )
        print("Loading custom model from", custom_model_path, flush=True)
        return custom_model_path

    print("Loading default model", flush=True)
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print("Downloading XTTS Model:", model_name, flush=True)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    print("XTTS Model downloaded", flush=True)
    return Path(model_path)


def load_model():
    model_path = _resolve_model_path()

    print("Loading XTTS", flush=True)
    xtts_config = XttsConfig()
    xtts_config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(xtts_config)

    use_deepspeed = settings.use_deepspeed
    device = settings.device
    if use_deepspeed and device.type != "cuda":
        print("Ignoring USE_DEEPSPEED=1 because CUDA is not enabled.", flush=True)
        use_deepspeed = False

    model.load_checkpoint(
        xtts_config,
        checkpoint_dir=str(model_path),
        eval=True,
        use_deepspeed=use_deepspeed,
    )
    model.to(device)
    print("XTTS Loaded.", flush=True)
    return xtts_config, model
