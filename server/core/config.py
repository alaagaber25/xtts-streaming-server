import os
from pathlib import Path

import torch
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_ROOT.parent

load_dotenv(dotenv_path=SERVER_ROOT / ".env", override=False)

USE_CPU = os.environ.get("USE_CPU", "0") == "1"
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))
USE_DEEPSPEED = os.environ.get("USE_DEEPSPEED", "0") == "1"
_custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "").strip()
CUSTOM_MODEL_PATH = Path(_custom_model_path) if _custom_model_path else None
SPEAKER_PROFILES_PATH = Path(
    os.environ.get("SPEAKER_PROFILES_PATH", "/app/speaker_profiles")
)
DEVICE = torch.device("cpu" if USE_CPU else "cuda")

if DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")
