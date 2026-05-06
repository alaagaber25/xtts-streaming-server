import csv
import json
import random
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))
import core.patches  # noqa: E402,F401

TEXT = "وبعدها كيف اقدر اساعدك اليوم"
LANGUAGE = "ar"
STREAM_CHUNK_SIZE = 25
OUTPUT_ROOT = REPO_ROOT / "matrix_runs" / (
    "direct_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

MODELS = [
    {"id": "new_sa", "path": REPO_ROOT / "tts_models" / "new_sa"},
]

SPEAKERS_BY_MODEL = {
    "new_sa": ["sa/nada", "sa/shahad", "eg/saied"],
}

GENERATION_PRESETS = [
    {
        "id": "default",
        "kwargs": {
            "temperature": 0.75,
            "top_k": 50,
            "top_p": 0.85,
            "length_penalty": 1.0,
            "repetition_penalty": 10.0,
            "do_sample": True,
        },
    },
    {
        "id": "conservative",
        "kwargs": {
            "temperature": 0.55,
            "top_k": 30,
            "top_p": 0.75,
            "length_penalty": 1.2,
            "repetition_penalty": 10.0,
            "do_sample": True,
        },
    },
    {
        "id": "low_variance",
        "kwargs": {
            "temperature": 0.35,
            "top_k": 5,
            "top_p": 0.6,
            "length_penalty": 1.0,
            "repetition_penalty": 10.0,
            "do_sample": True,
        },
    },
]

SEEDS = [1, 2, 3, 4, 5, 6]


def _safe_id(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_profile(profile_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    profile_dir = REPO_ROOT / "speaker_profiles" / profile_id
    speaker_embedding_path = profile_dir / "speaker_embedding.json"
    gpt_cond_latent_path = profile_dir / "gpt_cond_latent.json"
    if not speaker_embedding_path.is_file() or not gpt_cond_latent_path.is_file():
        raise FileNotFoundError(f"Missing speaker profile files for {profile_id}")

    speaker_embedding = json.loads(speaker_embedding_path.read_text(encoding="utf-8"))
    gpt_cond_latent = json.loads(gpt_cond_latent_path.read_text(encoding="utf-8"))
    resolved_speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).unsqueeze(-1)
    resolved_gpt_cond_latent = (
        torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    )
    return resolved_speaker_embedding, resolved_gpt_cond_latent


def _load_model(model_path: Path, device: torch.device):
    config_path = model_path / "config.json"
    model_weights_path = model_path / "model.pth"
    if not config_path.is_file() or not model_weights_path.is_file():
        raise FileNotFoundError(f"Incomplete XTTS model folder: {model_path}")

    print(f"Loading model: {model_path}", flush=True)
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(model_path),
        eval=True,
        use_deepspeed=False,
        vocab_path=str(model_path / "vocab.json"),
    )
    model.to(device)
    return model


def _postprocess_chunk(chunk: torch.Tensor) -> np.ndarray:
    wav = chunk.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    return (wav * 32767).astype(np.int16)


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm.tobytes())


def _run_case(model, case_dir: Path, speaker_id: str, preset: dict, seed: int) -> dict:
    _set_seed(seed)
    speaker_embedding, gpt_cond_latent = _load_profile(speaker_id)

    started = time.perf_counter()
    chunks = []
    chunk_samples = []
    for chunk in model.inference_stream(
        TEXT,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=STREAM_CHUNK_SIZE,
        enable_text_splitting=True,
        **preset["kwargs"],
    ):
        pcm = _postprocess_chunk(chunk)
        chunks.append(pcm)
        chunk_samples.append(int(pcm.shape[-1]))

    elapsed = time.perf_counter() - started
    if chunks:
        final_pcm = np.concatenate(chunks, axis=1)
    else:
        final_pcm = np.zeros((1, 0), dtype=np.int16)

    wav_path = case_dir / "final.wav"
    _write_wav(wav_path, final_pcm)
    duration_sec = final_pcm.shape[-1] / 24000

    return {
        "speaker": speaker_id,
        "preset": preset["id"],
        "seed": seed,
        "chunk_count": len(chunks),
        "chunk_samples": chunk_samples,
        "duration_sec": round(duration_sec, 3),
        "elapsed_sec": round(elapsed, 3),
        "wav": str(wav_path.relative_to(OUTPUT_ROOT)),
        **preset["kwargs"],
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running matrix on {device}. Output: {OUTPUT_ROOT}", flush=True)

    rows = []
    for model_spec in MODELS:
        model_id = model_spec["id"]
        model = _load_model(model_spec["path"], device)
        try:
            for speaker_id in SPEAKERS_BY_MODEL[model_id]:
                for preset in GENERATION_PRESETS:
                    for seed in SEEDS:
                        case_name = "_".join(
                            [
                                model_id,
                                _safe_id(speaker_id),
                                preset["id"],
                                f"seed{seed}",
                            ]
                        )
                        case_dir = OUTPUT_ROOT / case_name
                        print(f"Running {case_name}", flush=True)
                        row = {
                            "model": model_id,
                            "text": TEXT,
                            "language": LANGUAGE,
                            "stream_chunk_size": STREAM_CHUNK_SIZE,
                        }
                        try:
                            row.update(_run_case(model, case_dir, speaker_id, preset, seed))
                            row["status"] = "complete"
                        except Exception as exc:
                            row.update(
                                {
                                    "speaker": speaker_id,
                                    "preset": preset["id"],
                                    "seed": seed,
                                    "status": "error",
                                    "error": repr(exc),
                                }
                            )
                        rows.append(row)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report_json = OUTPUT_ROOT / "report.json"
    report_csv = OUTPUT_ROOT / "report.csv"
    report_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with report_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved report: {report_json}", flush=True)
    print(f"Saved report: {report_csv}", flush=True)


if __name__ == "__main__":
    main()
