from datetime import datetime
from uuid import uuid4

import torch
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from audio.debug_recorder import TTSOutputRecorder
from audio.processing import encode_audio_common, postprocess
from core.config import Settings
from schemas.requests import StreamingInputs, TTSInputs
from speakers.resolver import _resolve_conditioning_inputs

router = APIRouter()


def _new_request_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _build_streaming_recorder(parsed_input: StreamingInputs, request_id: str):
    settings = Settings()
    if not settings.save_tts_outputs:
        return None

    return TTSOutputRecorder(
        output_root=settings.resolved_tts_outputs_path,
        request_id=request_id,
        metadata={
            "endpoint": "/tts_stream",
            "text": parsed_input.text,
            "language": parsed_input.language,
            "speaker_profile_id": parsed_input.speaker_profile_id,
            "stream_chunk_size": parsed_input.stream_chunk_size,
            "add_wav_header": parsed_input.add_wav_header,
        },
    )


def predict_streaming_generator(
    parsed_input: StreamingInputs,
    model,
    external_speaker_profiles,
    request_id: str,
):
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

    recorder = _build_streaming_recorder(parsed_input, request_id)

    status = "interrupted"
    try:
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
            pcm_bytes = chunk.tobytes()
            if recorder is not None:
                recorder.write_chunk(pcm_bytes)

            if i == 0 and add_wav_header:
                yield encode_audio_common(b"", encode_base64=False)
            yield pcm_bytes
        status = "complete"
    finally:
        if recorder is not None:
            recorder.close(status=status)


@router.post("/tts_stream")
def predict_streaming_endpoint(parsed_input: StreamingInputs, request: Request):
    settings = Settings()
    request_id = _new_request_id()
    headers = {}
    if settings.save_tts_outputs:
        headers["X-TTS-Output-ID"] = request_id

    return StreamingResponse(
        predict_streaming_generator(
            parsed_input,
            request.app.state.model,
            request.app.state.external_speaker_profiles,
            request_id,
        ),
        media_type="audio/wav",
        headers=headers,
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
