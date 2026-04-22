import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

# ====== EDIT ONLY IF FOLDERS MOVE ======
RUN_DIR = os.path.abspath(r".\output_xtts_multispeaker\artifacts\checkpoint/tts")
ORIGINAL_HINT = os.path.abspath(
    r".\output_xtts_multispeaker\XTTS_v2.0_original_model_files"
)
REF_WAV = os.path.abspath(
    r"001_010.wav"
)  # TalksOfArabia\wavs\Abajora Podcast\001_010.wav   TalksOfArabia\wavs\Dupamicaffeine\001_010.wav
# AR_TEXT ="""أنا المساعد الذكي من شركة فوم،
# وباخذ من وقتكم دقائق بسيطة أشرح لكم كيف التقنية غيّرت شكل بيع العقار بالكامل." #" اللي قاعدين نتكلم عنها الموضوع جدا بسيط صح جدا بسيط لما تنام وش يصير جسمك يتوقف عن افراز ماده الادينوزين" #" العلاقات العاطفيه من اكثر المناطق المربكه والتجارب المنهكه بالنسبه للشخص "
# =======================================
AR_TEXT = """
أهلا بيك أنا أحمد من مستشارك العقاري
يفضل أستاذك هذا من حضرتك وممكن أعرف حضرتك اسمك عشان نسهل علينا التنسيق
وبعدها كيف أقدر أخدمك اليوم هل تبحث عن شيء معين أو تودك تشوف خريطة المشاريع والمرافق العامة أولا"""


def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def clean_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    p = re.sub(r"[\x00-\x1F]", "_", p)
    p = p.replace("/", os.sep).replace("\\", os.sep)
    p = re.sub(rf"{re.escape(os.sep)}+", lambda m: os.sep, p)
    return p.strip()


def ensure_model_pth(run_dir: str) -> str:
    """XTTS expects model.pth when model_path is a folder."""
    model_pth = os.path.join(run_dir, "model.pth")
    if os.path.isfile(model_pth):
        print(f"[INFO] Found existing model.pth ******: {model_pth}")
        return model_pth
    candidates = []
    preferred = os.path.join(run_dir, "checkpoint_275000.pth")
    if os.path.isfile(preferred):
        candidates.append(preferred)
    for f in os.listdir(run_dir):
        if f.endswith(".pth"):
            candidates.append(os.path.join(run_dir, f))
    for src in candidates:
        try:
            print(f"[INFO] Creating model.pth from: {src}")
            shutil.copyfile(src, model_pth)
            return model_pth
        except Exception as e:
            print(f"[WARN] Could not copy {src} -> model.pth: {e}")
    die("No .pth checkpoint found in run directory.")


def find_file(root: str, filename: str) -> str | None:
    if not os.path.isdir(root):
        return None
    IGNORE = {
        ".venv",
        "venv",
        "env",
        "site-packages",
        ".git",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    }
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in IGNORE]
        if filename in files:
            return os.path.abspath(os.path.join(r, filename))
    return None


def ensure_vocab_json(run_dir: str):
    """
    Some XTTS builds ignore config and load tokenizer from <run_dir>/vocab.json.
    Ensure it's present by copying from ORIGINAL_HINT (or project tree).
    """
    dst = os.path.join(run_dir, "vocab.json")
    if os.path.isfile(dst):
        return
    # preferred source: ORIGINAL_HINT
    src = None
    if os.path.isdir(ORIGINAL_HINT):
        cand = os.path.join(ORIGINAL_HINT, "vocab.json")
        if os.path.isfile(cand):
            src = cand
    if src is None:
        # search whole project (parent of run_dir's parent = project root)
        project_root = os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir))
        src = find_file(project_root, "vocab.json")
    if not src:
        die(
            "Could not find 'vocab.json'. Place it under "
            f"'{ORIGINAL_HINT}' or anywhere under the project folder."
        )
    print(f"[INFO] Copying tokenizer vocab -> {dst}")
    shutil.copyfile(src, dst)


def tensor_to_python(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [tensor_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: tensor_to_python(v) for k, v in value.items()}
    return value


def save_latent_files(out_dir: str, gpt_cond_latent, speaker_embedding):
    os.makedirs(out_dir, exist_ok=True)
    gpt_json_path = os.path.join(out_dir, "gpt.json")
    speaker_embedding_json_path = os.path.join(out_dir, "speaker_embedding.json")
    gpt_pt_path = os.path.join(out_dir, "gpt_cond_latent.pt")
    speaker_pt_path = os.path.join(out_dir, "speaker_embedding.pt")

    with open(gpt_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"gpt_cond_latent": tensor_to_python(gpt_cond_latent)},
            f,
            ensure_ascii=False,
        )

    with open(speaker_embedding_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"speaker_embedding": tensor_to_python(speaker_embedding)},
            f,
            ensure_ascii=False,
        )

    import torch

    torch.save(gpt_cond_latent, gpt_pt_path)
    torch.save(speaker_embedding, speaker_pt_path)

    print(f"[INFO] Saved gpt.json: {gpt_json_path}")
    print(f"[INFO] Saved speaker_embedding.json: {speaker_embedding_json_path}")
    print(f"[INFO] Saved gpt tensor: {gpt_pt_path}")
    print(f"[INFO] Saved speaker embedding tensor: {speaker_pt_path}")


def main():
    run_dir = os.path.abspath(RUN_DIR)
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isdir(run_dir):
        die(f"Run folder not found: {run_dir}")
    if not os.path.isfile(cfg_path):
        die(f"config.json not found: {cfg_path}")

    ensure_model_pth(run_dir)
    ensure_vocab_json(run_dir)  # <-- critical fix

    spk = os.path.abspath(clean_path(REF_WAV))
    if not os.path.isfile(spk):
        # try relative to project root
        prj = os.path.abspath(os.path.join(run_dir, os.pardir, os.pardir))
        alt = os.path.abspath(os.path.join(prj, REF_WAV))
        if os.path.isfile(alt):
            spk = alt
        else:
            die(f"Speaker wav not found:\n  {spk}\n  (also tried {alt})")

    # load model
    try:
        import torch
        import torchaudio
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except Exception as e:
        die(
            f"Missing deps. Run:\n  pip install TTS torch torchaudio tokenizers sentencepiece\nError: {e}"
        )

    checkpoint_path = ensure_model_pth(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Model folder: {run_dir}")
    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")
    print(" > Using model: xtts")

    config = XttsConfig()
    config.load_json(cfg_path)

    print("[INFO] Loading XTTS model from checkpoint...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=run_dir,
        checkpoint_path=checkpoint_path,
        vocab_path=os.path.join(run_dir, "vocab.json"),
        use_deepspeed=False,
    )
    model = model.cuda() if device == "cuda" else model.cpu()
    model.eval()

    print("[INFO] Computing conditioning latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[spk]
    )

    out_dir = os.path.join(run_dir, "generated_tests")
    os.makedirs(out_dir, exist_ok=True)
    save_latent_files(out_dir, gpt_cond_latent, speaker_embedding)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    target_sr = (cfg.get("audio") or {}).get("output_sample_rate", 24000)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = os.path.splitext(os.path.basename(spk))[0]
    out_path = os.path.join(out_dir, f"xtts_test_{tag}_{stamp}.wav")

    print(f"[INFO] Synthesizing | lang=ar")
    print(f"      speaker_wav: {spk}")
    print(f"      -> {out_path}")

    t0 = time.time()
    chunks = model.inference_stream(
        text=AR_TEXT,
        language="ar",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            print("Time to first chunk:", round(time.time() - t0, 3), "sec")
        print(f"Chunk {i}: shape = {chunk.shape}")
        wav_chunks.append(chunk.cpu())

    if not wav_chunks:
        die("No audio chunks were generated.")

    wav = torch.cat(wav_chunks, dim=0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2 and wav.shape[0] != 1:
        wav = wav
    else:
        wav = wav

    torchaudio.save(out_path, wav, target_sr)
    print(f"\n[DONE] Test audio saved:\n  {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
