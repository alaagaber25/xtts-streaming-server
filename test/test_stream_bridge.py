import asyncio
import logging
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.runtime.scheduler import StreamSession
from server.stream_bridge import RequestStreamBridge


class RequestStreamBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_preserves_chunk_order(self):
        session = StreamSession()
        session.put_nowait(b"chunk-1")
        session.put_nowait(b"chunk-2")
        session.finish()

        bridge = RequestStreamBridge(
            request_id="req-order",
            session=session,
            logger=logging.getLogger("test.stream_bridge.order"),
            queue_maxsize=2,
        )

        try:
            received = []
            async for chunk in bridge.iter_chunks():
                received.append(chunk)
        finally:
            await bridge.close()

        self.assertEqual(received, [b"chunk-1", b"chunk-2"])

    async def test_bridge_uses_bounded_queue_for_backpressure(self):
        session = StreamSession()
        session.put_nowait(b"chunk-1")
        session.put_nowait(b"chunk-2")
        session.finish()

        bridge = RequestStreamBridge(
            request_id="req-backpressure",
            session=session,
            logger=logging.getLogger("test.stream_bridge.backpressure"),
            queue_maxsize=1,
        )
        bridge.start()

        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(bridge.queue.qsize(), 1)

        consumer = bridge.iter_chunks()
        try:
            first = await anext(consumer)
            second = await anext(consumer)
            self.assertEqual([first, second], [b"chunk-1", b"chunk-2"])
            with self.assertRaises(StopAsyncIteration):
                await anext(consumer)
        finally:
            await bridge.close()


if __name__ == "__main__":
    unittest.main()
