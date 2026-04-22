import importlib.util
import inspect
import logging
import os
from typing import Any, Dict

import torch

from .audio import postprocess
from .compat import apply_runtime_compatibility_patches, ensure_torchaudio_compatibility
from .settings import Settings
from .streaming import build_isolated_xtts_stream


apply_runtime_compatibility_patches()
ensure_torchaudio_compatibility()

# XTTS imports rely on the stream generator resolving generation helpers
# from `transformers` during import time, so patch before importing them.
from TTS.tts.layers.xtts.stream_generator import init_stream_support as _init_stream_support
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
import TTS.tts.models.xtts as xtts_module
import TTS.utils.io as tts_io
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])

if "weights_only" in inspect.signature(torch.load).parameters:
    _original_load_fsspec = tts_io.load_fsspec

    def _load_fsspec_trusted(path, map_location=None, cache=True, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load_fsspec(path, map_location=map_location, cache=cache, **kwargs)

    tts_io.load_fsspec = _load_fsspec_trusted
    xtts_module.load_fsspec = _load_fsspec_trusted


class XTTSService:
    def __init__(
        self,
        model: Xtts,
        config: XttsConfig,
        device: torch.device,
        model_path: str,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.model_path = model_path
        self.logger = logger

    @classmethod
    def create(
        cls,
        settings: Settings,
        device: torch.device,
        logger: logging.Logger,
    ) -> "XTTSService":
        model_path = _resolve_model_path(settings, logger)

        logger.info("Loading XTTS")
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        model = Xtts.init_from_config(config)
        use_deepspeed = device.type == "cuda" and importlib.util.find_spec("deepspeed") is not None
        if device.type == "cuda" and not use_deepspeed:
            logger.warning("Deepspeed not installed; loading XTTS without Deepspeed")
        model.load_checkpoint(
            config,
            checkpoint_dir=model_path,
            eval=True,
            use_deepspeed=use_deepspeed,
        )
        model.to(device)
        logger.info("XTTS Loaded.")
        return cls(model=model, config=config, device=device, model_path=model_path, logger=logger)

    def serialize_chunk(self, chunk: Any) -> bytes:
        return postprocess(chunk).tobytes()

    def build_stream_iterator(self, payload: Dict[str, Any]):
        gpt_cond_latent = self._prepare_gpt_cond_latent(payload["gpt_cond_latent"])
        speaker_embedding = self._prepare_speaker_embedding(payload["speaker_embedding"])
        xtts_stream = build_isolated_xtts_stream(
            model=self.model,
            device=self.device,
            text=payload["text"],
            language=payload["language"],
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=payload["stream_chunk_size"],
            enable_text_splitting=True,
            split_sentence_fn=xtts_module.split_sentence,
        )

        def guarded_stream():
            while True:
                with torch.inference_mode():
                    try:
                        yield next(xtts_stream)
                    except StopIteration:
                        return

        return guarded_stream()

    def clone_speaker_latents(self, audio_path: str):
        with torch.inference_mode():
            return self.model.get_conditioning_latents(audio_path)

    def warmup(self) -> None:
        speaker_manager = getattr(self.model, "speaker_manager", None)
        speakers = getattr(speaker_manager, "speakers", None)
        if not speakers:
            self.logger.info("Skipping XTTS warmup because no studio speakers are available.")
            return

        speaker_name = next(iter(speakers.keys()))
        speaker_data = speakers[speaker_name]
        language = self.config.languages[0] if self.config.languages else "en"

        warmup_stream = self.model.inference_stream(
            "Warm up.",
            language,
            self._prepare_gpt_cond_latent(speaker_data["gpt_cond_latent"]),
            self._prepare_speaker_embedding(speaker_data["speaker_embedding"]),
            stream_chunk_size=20,
            enable_text_splitting=True,
        )

        with torch.inference_mode():
            next(warmup_stream, None)

        close = getattr(warmup_stream, "close", None)
        if close is not None:
            close()

        self.logger.info("XTTS warmup completed with studio speaker '%s'.", speaker_name)

    def get_studio_speakers(self) -> Dict[str, Dict[str, Any]]:
        speaker_manager = getattr(self.model, "speaker_manager", None)
        speakers = getattr(speaker_manager, "speakers", None)
        if not speakers:
            return {}

        return {
            speaker: {
                "speaker_embedding": speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                "gpt_cond_latent": speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
            }
            for speaker in speakers.keys()
        }

    def get_languages(self):
        return self.config.languages

    def get_default_speaker(self) -> Dict[str, Any] | None:
        speaker_manager = getattr(self.model, "speaker_manager", None)
        speakers = getattr(speaker_manager, "speakers", None)
        if not speakers:
            return None

        speaker_name = next(iter(speakers.keys()))
        return {
            "id": speaker_name,
            "speaker_embedding": speakers[speaker_name]["speaker_embedding"].cpu().squeeze().half().tolist(),
            "gpt_cond_latent": speakers[speaker_name]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
        }

    def _prepare_speaker_embedding(self, raw_embedding: Any) -> torch.Tensor:
        tensor = torch.as_tensor(raw_embedding, dtype=torch.float32, device=self.device)

        if tensor.ndim == 1:
            return tensor.unsqueeze(0).unsqueeze(-1)
        if tensor.ndim == 2:
            if tensor.shape[-1] == 1:
                return tensor.unsqueeze(0)
            return tensor.unsqueeze(-1)
        if tensor.ndim == 3:
            return tensor

        raise ValueError("speaker_embedding must resolve to a 1D, 2D, or 3D tensor for XTTS streaming")

    def _prepare_gpt_cond_latent(self, raw_latent: Any) -> torch.Tensor:
        tensor = torch.as_tensor(raw_latent, dtype=torch.float32, device=self.device)

        if tensor.ndim == 1:
            if tensor.numel() % 1024 != 0:
                raise ValueError("gpt_cond_latent must be divisible into 1024-wide vectors")
            return tensor.reshape((-1, 1024)).unsqueeze(0)

        if tensor.ndim == 2:
            if tensor.shape[-1] != 1024:
                if tensor.numel() % 1024 != 0:
                    raise ValueError("gpt_cond_latent must be divisible into 1024-wide vectors")
                tensor = tensor.reshape((-1, 1024))
            return tensor.unsqueeze(0)

        if tensor.ndim == 3 and tensor.shape[-1] == 1024:
            return tensor

        raise ValueError(
            "gpt_cond_latent must resolve to a tensor shaped like [steps, 1024] or [1, steps, 1024]"
        )


def _resolve_model_path(settings: Settings, logger: logging.Logger) -> str:
    custom_model_path = str(settings.custom_model_path)
    custom_config_path = os.path.join(custom_model_path, "config.json")
    if os.path.exists(custom_model_path) and os.path.isfile(custom_config_path):
        logger.info("Loading custom model from %s", custom_model_path)
        return custom_model_path

    logger.info("Loading default model")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    logger.info("Downloading XTTS Model: %s", model_name)
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    logger.info("XTTS Model downloaded")
    return model_path
