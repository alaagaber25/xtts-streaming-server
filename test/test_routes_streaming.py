import logging
import pathlib
import sys
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from starlette.requests import Request


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.audio import WAV_HEADER_BYTES
from server.routes.tts import predict_streaming_endpoint
from server.runtime.scheduler import StreamSession
from server.schemas import StreamingInputs
from server.state import AppState


async def _empty_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


class _FakeQueue:
    def qsize(self) -> int:
        return 0


class _FakeScheduler:
    def __init__(self, session: StreamSession) -> None:
        self.request_queue = _FakeQueue()
        self._session = session
        self.removed_request_ids: list[str] = []

    async def submit(self, payload):
        return "req-1", self._session

    def remove_session(self, request_id: str) -> None:
        self.removed_request_ids.append(request_id)


def _build_request(session: StreamSession) -> Request:
    app = FastAPI()
    app.state.xtts_state = AppState(
        settings=SimpleNamespace(default_speaker_profile_id=None),
        service=SimpleNamespace(),
        scheduler=_FakeScheduler(session),
        speaker_profiles=SimpleNamespace(),
        logger=logging.getLogger("test.tts_route"),
    )
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/tts_stream",
            "headers": [],
            "app": app,
        },
        _empty_receive,
    )


class StreamingRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_header_is_deferred_until_first_pcm_chunk(self):
        session = StreamSession()
        session.put_nowait(b"pcm")
        session.finish()
        request = _build_request(session)

        response = await predict_streaming_endpoint(
            StreamingInputs(
                text="hello",
                language="en",
                speaker_embedding=[0.1, 0.2],
                gpt_cond_latent=[[1.0] * 1024],
            ),
            request,
        )

        self.assertEqual(await anext(response.body_iterator), WAV_HEADER_BYTES)
        self.assertEqual(await anext(response.body_iterator), b"pcm")
        with self.assertRaises(StopAsyncIteration):
            await anext(response.body_iterator)

        scheduler = request.app.state.xtts_state.scheduler
        self.assertEqual(scheduler.removed_request_ids, ["req-1"])

    async def test_failed_stream_without_pcm_raises_before_sending_header(self):
        session = StreamSession()
        session.fail(RuntimeError("boom"))
        request = _build_request(session)

        response = await predict_streaming_endpoint(
            StreamingInputs(
                text="hello",
                language="en",
                speaker_embedding=[0.1, 0.2],
                gpt_cond_latent=[[1.0] * 1024],
            ),
            request,
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await anext(response.body_iterator)

        scheduler = request.app.state.xtts_state.scheduler
        self.assertEqual(scheduler.removed_request_ids, ["req-1"])


if __name__ == "__main__":
    unittest.main()
