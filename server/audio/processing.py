import base64
import io
import wave

import numpy as np
import torch


def postprocess(wav):
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        return base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
    return wav_buf.read()

