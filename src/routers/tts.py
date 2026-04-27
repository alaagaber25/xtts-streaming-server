import torch
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from audio.processing import encode_audio_common, postprocess
from schemas.requests import StreamingInputs, TTSInputs
from speakers.resolver import _resolve_conditioning_inputs

router = APIRouter()


def predict_streaming_generator(parsed_input: StreamingInputs, model, external_speaker_profiles):
    speaker_embedding, gpt_cond_latent = _resolve_conditioning_inputs(
        model=model,
        external_speaker_profiles=external_speaker_profiles,
        speaker_profile_id=parsed_input.speaker_profile_id,
        speaker_embedding=parsed_input.speaker_embedding,
        gpt_cond_latent=parsed_input.gpt_cond_latent,
    )
    text = parsed_input.text
    language = parsed_input.language
    stream_chunk_size = int(parsed_input.stream_chunk_size)
    add_wav_header = parsed_input.add_wav_header

    chunks = model.inference_stream(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=stream_chunk_size,
        enable_text_splitting=True,
    )

    for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        if i == 0 and add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()


@router.post("/tts_stream")
def predict_streaming_endpoint(parsed_input: StreamingInputs, request: Request):
    return StreamingResponse(
        predict_streaming_generator(
            parsed_input,
            request.app.state.model,
            request.app.state.external_speaker_profiles,
        ),
        media_type="audio/wav",
    )


@router.post("/tts")
def predict_speech(parsed_input: TTSInputs, request: Request):
    speaker_embedding, gpt_cond_latent = _resolve_conditioning_inputs(
        model=request.app.state.model,
        external_speaker_profiles=request.app.state.external_speaker_profiles,
        speaker_profile_id=parsed_input.speaker_profile_id,
        speaker_embedding=parsed_input.speaker_embedding,
        gpt_cond_latent=parsed_input.gpt_cond_latent,
    )
    out = request.app.state.model.inference(
        parsed_input.text,
        parsed_input.language,
        gpt_cond_latent,
        speaker_embedding,
    )

    wav = postprocess(torch.tensor(out["wav"]))
    return encode_audio_common(wav.tobytes())
