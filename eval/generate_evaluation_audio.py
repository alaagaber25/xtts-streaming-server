import csv
import json
import os
import random
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from dotenv import dotenv_values
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# Manual settings.
OVERWRITE = True
DOWNLOAD_BASE = False
INCLUDE_SHARED_PROFILES = True
MAX_CHARS_PER_CHUNK = 150

# Set to None for all prefixes, or use values like ["sa", "base"].
SELECTED_PREFIXES = None

# Edit these evaluation texts before running a final evaluation pass.
SA_TEXT = """ 
        السلام عليكم ورحمة الله وبركاته حياك الله يا استاذ معك خالد العتيبي مستشار عقاري مختص بالمشاريع السكنية الحديثة في الرياض والخبر وجدة

ابغى في البداية اعرفك بنفسي بشكل سريع وافهم احتياجك بالتفصيل عشان اقدر اوفر لك الخيار المناسب سواء للسكن العائلي او للاستثمار طويل المدى

المشروع اللي اتكلم عنه اليوم يعتبر من المشاريع المميزة جدا من ناحية الموقع وجودة البناء والخدمات المتوفرة داخله

المشروع قريب من الطرق الرئيسية والمدارس والمستشفيات والمولات وهذا الشي يعطيه قيمة عالية وراحة كبيرة للسكان

عندنا خيارات متعددة شقق تاون هاوس فلل مستقلة ومساحات مختلفة تناسب العوايل الصغيرة والكبيرة

كذلك فيه انظمة دفع مرنة جدا تقدر تبدأ بدفعة اولى مناسبة والباقي على اقساط شهرية او ربع سنوية حسب الخطة اللي تناسبك

من الاشياء الجميلة بالمشروع وجود مساحات خضراء نادي رياضي مسارات للمشي جلسات خارجية ومناطق العاب للاطفال

واذا كنت مهتم بالاستثمار فالمشروع عليه طلب ممتاز خصوصا مع التطور الكبير اللي تشهده المنطقة وارتفاع الاسعار بشكل مستمر

بعض الوحدات تتميز باطلالة مباشرة على الحديقة او المسبح وفيه وحدات جاهزة للتسليم الفوري ووحدات تحت الانشاء باسعار منافسة

اذا تسمح لي ابغى اعرف كم عدد الغرف اللي تحتاجها وهل تفضل السكن داخل مجمع هادي او قريب من المناطق الحيوية

وانا باذن الله اوضح لك كل التفاصيل بكل شفافية من المساحات الى الضمانات والخدمات عشان تكون الصورة واضحة بالكامل قبل اتخاذ القرار

وتشرفنا بخدمتك وان شاء الله نساعدك تحصل على العقار المناسب اللي يليق فيك وفي عايلتك"""
EG_TEXT = """ 
        مساء الفل يا فندم مع حضرتك احمد الجندي مستشار عقاري في شركة متخصصة في المشروعات السكنية والاستثمارية في القاهرة الجديدة والعاصمة الادارية

حبيت اعرف حضرتك بنفسي بسرعة وافهم احتياجك بشكل دقيق علشان اقدر ارشح لك انسب اختيار سواء للسكن او للاستثمار

المشروع اللي بكلمك عنه موجود في موقع مميز جدا قريب من الطرق الرئيسية والمحاور المهمة وده بيسهل الحركة والانتقال بشكل كبير

الكمباوند فيه مساحات متنوعة شقق دوبلكسات وبنتهاوس والتشطيب فيه مستوى عالي جدا وكمان فيه انظمة سداد مرنة تبدأ بمقدم بسيط وتقسيط على عدة سنين

من الحاجات اللي مميزة المشروع فعلا ان فيه مناطق خضراء واسعة جيم كلوب هاوس تراك للجري واماكن مخصصة للاطفال

ولو حضرتك بتفكر في الاستثمار فالمشروع عليه طلب عالي جدا خصوصا ان المنطقة سعر المتر فيها بيزيد بشكل ملحوظ كل فترة

على فكرة عندنا وحدات باطلالات مختلفة لاجون جاردن وكورنر وفيه كمان وحدات جاهزة للاستلام الفوري

احب اعرف من حضرتك بتدور على كام غرفة وايه الميزانية المناسبة ليك

وانا هشرح لك كل التفاصيل بكل وضوح من غير اي ضغط او لف ودوران علشان تكون واخد قرار مرتاح ومقتنع مية في المية

متشرف جدا بالتعامل مع حضرتك وان شاء الله نلاقي الوحدة المناسبة اللي تناسب ذوقك واحتياجات اسرتك
"""
LANGUAGE = "ar"


EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import core.patches  # noqa: E402,F401

ENV = dotenv_values(REPO_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = ENV.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = ENV.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = ENV.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


USE_CPU = _env_bool("USE_CPU", False)
USE_DEEPSPEED = _env_bool("USE_DEEPSPEED", False)
NUM_THREADS = _env_int("NUM_THREADS", os.cpu_count() or 1)
SEED = _env_int("TTS_SEED", None)
GENERATION_KWARGS = {
    "temperature": _env_float("TTS_TEMPERATURE", 0.35),
    "top_k": _env_int("TTS_TOP_K", 5),
    "top_p": _env_float("TTS_TOP_P", 0.6),
    "length_penalty": _env_float("TTS_LENGTH_PENALTY", 1.5),
    "repetition_penalty": _env_float("TTS_REPETITION_PENALTY", 10.0),
    "do_sample": _env_bool("TTS_DO_SAMPLE", True),
}
if NUM_THREADS is not None:
    torch.set_num_threads(NUM_THREADS)

PROMPTS_ROOT = EVAL_ROOT / "prompts"
SPEAKER_PROFILES_ROOT = EVAL_ROOT / "speaker_profiles"
OUTPUT_ROOT = EVAL_ROOT / "audio_samples"
TTS_MODELS_ROOT = REPO_ROOT / "tts_models"

BASE_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_DIALECTS = {
    "sa": ["sa"],
    "new_sa": ["sa"],
    "eg": ["eg"],
    "base": ["sa", "eg"],
}
TEXT_BY_DIALECT = {
    "sa": SA_TEXT,
    "eg": EG_TEXT,
}
PROMPT_DIR_BY_DIALECT = {
    "sa": PROMPTS_ROOT / "sa",
    "eg": PROMPTS_ROOT / "eg",
}
SHARED_PROFILE_NAMES = [path.stem for path in sorted(PROMPTS_ROOT.glob("base*.wav"))]


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_with_trusted_checkpoint_default(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_with_trusted_checkpoint_default


def _base_model_path() -> Path:
    try:
        data_dir = Path(get_user_data_dir("tts"))
    except FileNotFoundError:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise
        data_dir = Path(local_app_data) / "tts"

    model_path = data_dir / BASE_MODEL_NAME.replace("/", "--")
    required = [
        model_path / "config.json",
        model_path / "model.pth",
        model_path / "vocab.json",
    ]
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
        if model_dir.name not in MODEL_DIALECTS:
            print(
                f"Skipping model without eval dialect mapping: {model_dir.name}",
                flush=True,
            )
            continue
        models.append((model_dir.name, model_dir))

    models.append(("base", _base_model_path()))
    return models


def _dialect_profile_names(dialect: str) -> list[str]:
    prompt_dir = PROMPT_DIR_BY_DIALECT[dialect]
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Missing prompt directory: {prompt_dir}")
    names = [path.stem for path in sorted(prompt_dir.glob("*.wav"))]
    if not names:
        raise FileNotFoundError(f"No prompts found for dialect: {dialect}")
    return names


def _profile_ids_for_model(prefix: str) -> list[tuple[str, str]]:
    profile_ids = []
    for dialect in MODEL_DIALECTS[prefix]:
        for profile_name in _dialect_profile_names(dialect):
            profile_dir = SPEAKER_PROFILES_ROOT / prefix / profile_name
            if not profile_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing generated speaker profile: {profile_dir}"
                )
            profile_ids.append((dialect, profile_name))
        if INCLUDE_SHARED_PROFILES:
            for profile_name in SHARED_PROFILE_NAMES:
                profile_dir = SPEAKER_PROFILES_ROOT / prefix / profile_name
                if not profile_dir.is_dir():
                    raise FileNotFoundError(
                        f"Missing generated shared speaker profile: {profile_dir}"
                    )
                profile_ids.append((dialect, profile_name))
    return profile_ids


def _validate_model_dir(model_path: Path) -> tuple[Path, Path]:
    config_path = model_path / "config.json"
    vocab_path = model_path / "vocab.json"
    weights_path = model_path / "model.pth"
    missing = [
        path for path in (config_path, vocab_path, weights_path) if not path.is_file()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Incomplete XTTS model folder: {model_path}\n{missing_text}"
        )
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


def _load_profile(prefix: str, profile_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    profile_dir = SPEAKER_PROFILES_ROOT / prefix / profile_name
    speaker_embedding_path = profile_dir / "speaker_embedding.json"
    gpt_cond_latent_path = profile_dir / "gpt_cond_latent.json"
    if not speaker_embedding_path.is_file() or not gpt_cond_latent_path.is_file():
        raise FileNotFoundError(f"Missing speaker profile files: {profile_dir}")

    speaker_embedding = json.loads(speaker_embedding_path.read_text(encoding="utf-8"))
    gpt_cond_latent = json.loads(gpt_cond_latent_path.read_text(encoding="utf-8"))
    resolved_speaker_embedding = (
        torch.tensor(speaker_embedding).unsqueeze(0).unsqueeze(-1)
    )
    resolved_gpt_cond_latent = (
        torch.tensor(gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0)
    )
    return resolved_speaker_embedding, resolved_gpt_cond_latent


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _postprocess_wav(wav) -> np.ndarray:
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    return (wav * 32767).astype(np.int16)


def _split_text_for_xtts(text: str, max_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks = []
    current_words = []
    current_length = 0
    for word in normalized.split(" "):
        word_length = len(word)
        next_length = word_length if not current_words else current_length + 1 + word_length
        if current_words and next_length > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = word_length
        else:
            current_words.append(word)
            current_length = next_length

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm.tobytes())


def _run_case(model, prefix: str, dialect: str, profile_name: str) -> dict:
    output_path = OUTPUT_ROOT / prefix / dialect / f"{profile_name}.wav"
    if not OVERWRITE and output_path.is_file():
        print(f"Skipping existing audio: {output_path}", flush=True)
        return {
            "model": prefix,
            "dialect": dialect,
            "profile": profile_name,
            "text": TEXT_BY_DIALECT[dialect],
            "wav": str(output_path.relative_to(EVAL_ROOT)),
            "status": "skipped",
        }

    speaker_embedding, gpt_cond_latent = _load_profile(prefix, profile_name)
    _set_seed(SEED)

    print(f"Generating {prefix}/{dialect}/{profile_name}", flush=True)
    started = time.perf_counter()
    pcm_chunks = []
    text_chunks = _split_text_for_xtts(TEXT_BY_DIALECT[dialect], MAX_CHARS_PER_CHUNK)
    with torch.inference_mode():
        for chunk_index, text_chunk in enumerate(text_chunks, start=1):
            print(
                f"  chunk {chunk_index}/{len(text_chunks)} ({len(text_chunk)} chars)",
                flush=True,
            )
            out = model.inference(
                text_chunk,
                LANGUAGE,
                gpt_cond_latent,
                speaker_embedding,
                **GENERATION_KWARGS,
            )
            pcm_chunks.append(_postprocess_wav(out["wav"]))
    elapsed = time.perf_counter() - started

    if pcm_chunks:
        pcm = np.concatenate(pcm_chunks, axis=1)
    else:
        pcm = np.zeros((1, 0), dtype=np.int16)
    _write_wav(output_path, pcm)
    duration_sec = pcm.shape[-1] / 24000

    return {
        "model": prefix,
        "dialect": dialect,
        "profile": profile_name,
        "text": TEXT_BY_DIALECT[dialect],
        "language": LANGUAGE,
        "seed": SEED,
        "duration_sec": round(duration_sec, 3),
        "elapsed_sec": round(elapsed, 3),
        "text_chunks": len(text_chunks),
        "max_chars_per_chunk": MAX_CHARS_PER_CHUNK,
        "wav": str(output_path.relative_to(EVAL_ROOT)),
        "status": "complete",
        **GENERATION_KWARGS,
    }


def _write_reports(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
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


def main():
    device = torch.device(
        "cpu" if USE_CPU or not torch.cuda.is_available() else "cuda"
    )
    if USE_DEEPSPEED and device.type != "cuda":
        print("Ignoring USE_DEEPSPEED=1 because CUDA is not enabled.", flush=True)
    selected_prefixes = set(SELECTED_PREFIXES or MODEL_DIALECTS)

    print(f"Generating evaluation audio on {device}", flush=True)
    print(
        "Generation settings:",
        {"seed": SEED, "num_threads": NUM_THREADS, **GENERATION_KWARGS},
        flush=True,
    )
    rows = []
    for prefix, model_path in _discover_models():
        if prefix not in selected_prefixes:
            continue

        model = _load_model(model_path, device)
        try:
            for dialect, profile_name in _profile_ids_for_model(prefix):
                try:
                    rows.append(_run_case(model, prefix, dialect, profile_name))
                except Exception as exc:
                    rows.append(
                        {
                            "model": prefix,
                            "dialect": dialect,
                            "profile": profile_name,
                            "text": TEXT_BY_DIALECT[dialect],
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"Error in {prefix}/{dialect}/{profile_name}: {exc!r}",
                        flush=True,
                    )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_reports(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(
        f"Done. Complete: {complete}, skipped: {skipped}, errors: {errors}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
