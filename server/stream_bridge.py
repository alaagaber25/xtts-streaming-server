import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from .runtime import StreamSession, iterate_session_chunks


REQUEST_STREAM_QUEUE_MAXSIZE = 8
_BRIDGE_FINISHED = object()


@dataclass
class BridgedChunk:
    sequence: int
    payload: bytes
    enqueued_at_ns: int


class RequestStreamBridge:
    def __init__(
        self,
        *,
        request_id: str,
        session: StreamSession,
        logger: logging.Logger,
        queue_maxsize: int = REQUEST_STREAM_QUEUE_MAXSIZE,
    ) -> None:
        self.request_id = request_id
        self.session = session
        self.logger = logger
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=max(1, queue_maxsize))
        self._producer_task: asyncio.Task | None = None
        self._producer_error: BaseException | None = None
        self._enqueued_chunks = 0
        self._dequeued_chunks = 0

    def start(self) -> None:
        if self._producer_task is not None:
            return

        self.logger.info(
            "[%s] bridge start queue_maxsize=%s started_at_ns=%s",
            self.request_id,
            self.queue.maxsize,
            time.monotonic_ns(),
        )
        self._producer_task = asyncio.create_task(
            self._run_producer(),
            name=f"stream-bridge-{self.request_id}",
        )

    async def _run_producer(self) -> None:
        try:
            async for chunk in iterate_session_chunks(self.session):
                sequence = self._enqueued_chunks + 1
                qsize_before = self.queue.qsize()
                started_at_ns = time.monotonic_ns()
                await self.queue.put(
                    BridgedChunk(
                        sequence=sequence,
                        payload=chunk,
                        enqueued_at_ns=started_at_ns,
                    )
                )
                finished_at_ns = time.monotonic_ns()
                wait_ms = (finished_at_ns - started_at_ns) / 1_000_000
                self._enqueued_chunks = sequence

                self.logger.info(
                    "[%s] bridge enqueue seq=%s bytes=%s qsize_before=%s qsize_after=%s wait_ms=%.3f ts_ns=%s",
                    self.request_id,
                    sequence,
                    len(chunk),
                    qsize_before,
                    self.queue.qsize(),
                    wait_ms,
                    finished_at_ns,
                )
                if qsize_before >= self.queue.maxsize or wait_ms >= 1.0:
                    self.logger.warning(
                        "[%s] bridge backpressure seq=%s qsize_before=%s qsize_after=%s wait_ms=%.3f",
                        self.request_id,
                        sequence,
                        qsize_before,
                        self.queue.qsize(),
                        wait_ms,
                    )

            await self.queue.put(_BRIDGE_FINISHED)
            self.logger.info(
                "[%s] bridge producer finished enqueued=%s session_error=%s qsize=%s finished_at_ns=%s",
                self.request_id,
                self._enqueued_chunks,
                self.session.error,
                self.queue.qsize(),
                time.monotonic_ns(),
            )
        except asyncio.CancelledError:
            self.logger.info("[%s] bridge producer cancelled", self.request_id)
            raise
        except Exception as exc:
            self._producer_error = exc
            self.logger.exception("[%s] bridge producer failed: %s", self.request_id, exc)
            await self.queue.put(_BRIDGE_FINISHED)

    async def iter_chunks(self) -> AsyncGenerator[bytes, None]:
        self.start()

        while True:
            started_at_ns = time.monotonic_ns()
            item = await self.queue.get()
            finished_at_ns = time.monotonic_ns()
            wait_ms = (finished_at_ns - started_at_ns) / 1_000_000

            if item is _BRIDGE_FINISHED:
                self.logger.info(
                    "[%s] bridge dequeue finished dequeued=%s qsize=%s wait_ms=%.3f ts_ns=%s",
                    self.request_id,
                    self._dequeued_chunks,
                    self.queue.qsize(),
                    wait_ms,
                    finished_at_ns,
                )
                break

            assert isinstance(item, BridgedChunk)
            self._dequeued_chunks = item.sequence
            queue_age_ms = (finished_at_ns - item.enqueued_at_ns) / 1_000_000
            self.logger.info(
                "[%s] bridge dequeue seq=%s bytes=%s qsize_after=%s queue_age_ms=%.3f wait_ms=%.3f ts_ns=%s",
                self.request_id,
                item.sequence,
                len(item.payload),
                self.queue.qsize(),
                queue_age_ms,
                wait_ms,
                finished_at_ns,
            )
            yield item.payload

        if self._producer_error is not None:
            raise self._producer_error

    async def close(self) -> None:
        if self._producer_task is None:
            return

        if not self._producer_task.done():
            self._producer_task.cancel()

        try:
            await self._producer_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._producer_error is None:
                self._producer_error = exc
