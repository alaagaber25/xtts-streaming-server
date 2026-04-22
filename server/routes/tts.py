import asyncio
import tempfile
from pathlib import Path
from typing import Tuple

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..audio import WAV_HEADER_BYTES, encode_audio_common
from ..runtime import collect_session_bytes, iterate_session_chunks
from ..schemas import StreamingInputs, TTSInputs
from ..state import get_app_state


router = APIRouter()


@router.post("/clone_speaker")
async def predict_speaker(
    wav_file: UploadFile,
    request: Request,
    speaker_profile_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    description: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
):
    state = get_app_state(request)
    temp_audio_path: Path | None = None
    try:
        if speaker_profile_id is None and (name is not None or description is not None):
            raise HTTPException(
                status_code=422,
                detail="speaker_profile_id is required when saving profile metadata",
            )

        audio_bytes = await wav_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = Path(temp_audio.name)
            temp_audio.write(audio_bytes)

        gpt_cond_latent, speaker_embedding = await state.scheduler.run_sync(
            state.service.clone_speaker_latents,
            str(temp_audio_path),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_audio_path is not None:
            temp_audio_path.unlink(missing_ok=True)

    response = {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }
    if speaker_profile_id is not None:
        try:
            profile = state.speaker_profiles.save_profile(
                speaker_profile_id,
                speaker_embedding=speaker_embedding,
                gpt_cond_latent=gpt_cond_latent,
                name=name,
                description=description,
                overwrite=overwrite,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        response["speaker_profile"] = {
            "id": profile["id"],
            "name": profile.get("name"),
            "description": profile.get("description"),
            "created_at": profile.get("created_at"),
        }

    return response


@router.post("/tts_stream")
async def predict_streaming_endpoint(parsed_input: StreamingInputs, request: Request):
    state = get_app_state(request)
    speaker_embedding, gpt_cond_latent = _resolve_speaker_data(parsed_input, request)
    payload = {
        "text": parsed_input.text,
        "language": parsed_input.language,
        "speaker_embedding": speaker_embedding,
        "gpt_cond_latent": gpt_cond_latent,
        "stream_chunk_size": parsed_input.stream_chunk_size,
        "add_wav_header": parsed_input.add_wav_header,
    }
    request_id, session = await state.scheduler.submit(payload)
    state.logger.info(
        "[%s] queued stream request (queued=%s)",
        request_id,
        state.scheduler.request_queue.qsize(),
    )

    async def stream_audio():
        header_sent = False
        pcm_chunks_sent = 0
        pcm_bytes_sent = 0
        try:
            async for chunk in iterate_session_chunks(session):
                if payload["add_wav_header"] and not header_sent:
                    header_sent = True
                    yield WAV_HEADER_BYTES

                pcm_chunks_sent += 1
                pcm_bytes_sent += len(chunk)
                yield chunk

            if session.error is not None and pcm_chunks_sent == 0:
                raise session.error

            if session.error is not None:
                state.logger.warning(
                    "[%s] stream ended with error after %s chunks (%s bytes): %s",
                    request_id,
                    pcm_chunks_sent,
                    pcm_bytes_sent,
                    session.error,
                )
        except asyncio.CancelledError:
            session.cancel()
            state.logger.info("[%s] client disconnected", request_id)
            raise
        finally:
            state.scheduler.remove_session(request_id)

    return StreamingResponse(stream_audio(), media_type="audio/wav")


@router.post("/tts")
async def predict_speech(parsed_input: TTSInputs, request: Request):
    state = get_app_state(request)
    speaker_embedding, gpt_cond_latent = _resolve_speaker_data(parsed_input, request)
    payload = {
        "text": parsed_input.text,
        "language": parsed_input.language,
        "speaker_embedding": speaker_embedding,
        "gpt_cond_latent": gpt_cond_latent,
        "stream_chunk_size": 20,
        "add_wav_header": False,
    }
    request_id, session = await state.scheduler.submit(payload)
    state.logger.info(
        "[%s] queued full synthesis request (queued=%s)",
        request_id,
        state.scheduler.request_queue.qsize(),
    )

    try:
        pcm_audio = await collect_session_bytes(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        state.scheduler.remove_session(request_id)

    return encode_audio_common(pcm_audio)


def _resolve_speaker_data(parsed_input: StreamingInputs | TTSInputs, request: Request) -> Tuple[list, list]:
    state = get_app_state(request)

    if parsed_input.speaker_embedding is not None and parsed_input.gpt_cond_latent is not None:
        return parsed_input.speaker_embedding, parsed_input.gpt_cond_latent

    if parsed_input.speaker_profile_id:
        return _load_profile_or_404(state, parsed_input.speaker_profile_id)

    default_profile_id = state.settings.default_speaker_profile_id
    if default_profile_id:
        return _load_profile_or_404(state, default_profile_id)

    if state.speaker_profiles.has_profile("default"):
        return _load_profile_or_404(state, "default")

    default_speaker = state.service.get_default_speaker()
    if default_speaker is not None:
        return default_speaker["speaker_embedding"], default_speaker["gpt_cond_latent"]

    raise HTTPException(
        status_code=422,
        detail=(
            "No speaker source was provided. Send speaker_embedding and gpt_cond_latent, "
            "or speaker_profile_id, or configure a server default speaker profile."
        ),
    )


def _load_profile_or_404(state, profile_id: str) -> Tuple[list, list]:
    try:
        profile = state.speaker_profiles.get_profile(profile_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker profile '{profile_id}' was not found",
        )
    return profile["speaker_embedding"], profile["gpt_cond_latent"]
