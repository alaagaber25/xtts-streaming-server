import asyncio
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.runtime.scheduler import InferenceScheduler, collect_session_bytes


class InferenceSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_requests_respects_max_batch(self):
        scheduler = InferenceScheduler(
            build_stream=lambda payload: iter(payload["chunks"]),
            serialize_chunk=lambda chunk: chunk,
            collection_window=0.01,
            max_batch_size=4,
            poll_interval=0.001,
        )

        try:
            for index in range(3):
                await scheduler.submit(
                    {"chunks": [f"chunk-{index}".encode("utf-8")]},
                    request_id=str(index),
                )

            batch = await scheduler.collect_requests(
                timeout=0.001,
                max_batch=2,
                wait_for_first=False,
            )

            self.assertEqual([item.request_id for item in batch], ["0", "1"])
            self.assertEqual(scheduler.request_queue.qsize(), 1)
        finally:
            scheduler.remove_session("0")
            scheduler.remove_session("1")
            scheduler.remove_session("2")

    async def test_worker_round_robins_active_streams(self):
        call_order = []

        def build_stream(payload):
            def iterator():
                for chunk in payload["chunks"]:
                    call_order.append(f"{payload['id']}:{chunk.decode('utf-8')}")
                    yield chunk

            return iterator()

        scheduler = InferenceScheduler(
            build_stream=build_stream,
            serialize_chunk=lambda chunk: chunk,
            collection_window=0.02,
            max_batch_size=2,
            poll_interval=0.001,
        )
        await scheduler.start()

        try:
            request_a, session_a = await scheduler.submit(
                {"id": "A", "chunks": [b"a1", b"a2"]},
                request_id="A",
            )
            request_b, session_b = await scheduler.submit(
                {"id": "B", "chunks": [b"b1", b"b2"]},
                request_id="B",
            )

            audio_a, audio_b = await asyncio.gather(
                asyncio.wait_for(collect_session_bytes(session_a), timeout=1.0),
                asyncio.wait_for(collect_session_bytes(session_b), timeout=1.0),
            )

            self.assertEqual(audio_a, b"a1a2")
            self.assertEqual(audio_b, b"b1b2")
            self.assertEqual(call_order[:4], ["A:a1", "B:b1", "A:a2", "B:b2"])
        finally:
            scheduler.remove_session(request_a)
            scheduler.remove_session(request_b)
            await scheduler.stop()


if __name__ == "__main__":
    unittest.main()
