import asyncio
import contextlib
import functools
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Iterator, List, Optional, Tuple


_ITERATION_FINISHED = object()
_SESSION_FINISHED = object()


class StreamSession:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.finished = False
        self.cancelled = False
        self.error: Optional[BaseException] = None
        self.created_at = time.monotonic()

    async def put(self, chunk: bytes) -> None:
        if self.finished or self.cancelled:
            return
        await self.queue.put(chunk)

    def put_nowait(self, chunk: bytes) -> None:
        if self.finished or self.cancelled:
            return
        self.queue.put_nowait(chunk)

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.queue.put_nowait(_SESSION_FINISHED)

    def fail(self, exc: BaseException) -> None:
        self.error = exc
        self.finish()

    def cancel(self) -> None:
        self.cancelled = True
        self.finish()


@dataclass
class ScheduledRequest:
    request_id: str
    payload: Any
    session: StreamSession


@dataclass
class ActiveRequest:
    request: ScheduledRequest
    iterator: Iterator[Any]


def _next_or_sentinel(iterator: Iterator[Any]) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return _ITERATION_FINISHED


class InferenceScheduler:
    def __init__(
        self,
        build_stream: Callable[[Any], Iterator[Any]],
        serialize_chunk: Callable[[Any], bytes],
        collection_window: float = 0.03,
        max_batch_size: int = 4,
        poll_interval: float = 0.005,
        worker_count: int = 1,
        warmup: Optional[Callable[[], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.build_stream = build_stream
        self.serialize_chunk = serialize_chunk
        self.collection_window = collection_window
        self.max_batch_size = max_batch_size
        self.poll_interval = poll_interval
        self.worker_count = max(1, worker_count)
        self.warmup = warmup
        self.logger = logger or logging.getLogger("xtts.scheduler")

        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.sessions: Dict[str, StreamSession] = {}

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        self._loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="xtts-gpu-worker",
        )

        if self.warmup is not None:
            try:
                await self.run_sync(self.warmup)
            except Exception as exc:
                self.logger.warning("XTTS warmup skipped after error: %s", exc)

        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(worker_index), name=f"xtts-worker-{worker_index}")
            for worker_index in range(self.worker_count)
        ]
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return

        for task in self._worker_tasks:
            task.cancel()

        for task in self._worker_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._worker_tasks.clear()

        for request_id, session in list(self.sessions.items()):
            session.fail(RuntimeError("Inference scheduler stopped"))
            self.sessions.pop(request_id, None)

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

        self._loop = None
        self._started = False

    async def run_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._loop is None or self._executor is None:
            raise RuntimeError("Inference scheduler has not been started")

        call = functools.partial(func, *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, call)

    async def submit(
        self,
        payload: Any,
        request_id: Optional[str] = None,
    ) -> Tuple[str, StreamSession]:
        resolved_request_id = request_id or uuid.uuid4().hex
        session = StreamSession()
        scheduled_request = ScheduledRequest(
            request_id=resolved_request_id,
            payload=payload,
            session=session,
        )
        self.sessions[resolved_request_id] = session
        await self.request_queue.put(scheduled_request)
        return resolved_request_id, session

    def remove_session(self, request_id: str) -> None:
        self.sessions.pop(request_id, None)

    async def collect_requests(
        self,
        timeout: Optional[float] = None,
        max_batch: Optional[int] = None,
        wait_for_first: bool = False,
    ) -> List[ScheduledRequest]:
        batch: List[ScheduledRequest] = []
        window = self.collection_window if timeout is None else max(timeout, 0.0)
        batch_limit = self.max_batch_size if max_batch is None else max(0, max_batch)

        if batch_limit == 0:
            return batch

        start = time.monotonic()

        if wait_for_first:
            try:
                item = await asyncio.wait_for(self.request_queue.get(), timeout=window)
            except asyncio.TimeoutError:
                return batch
            batch.append(item)

        while len(batch) < batch_limit:
            try:
                batch.append(self.request_queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            remaining = window - (time.monotonic() - start)
            if remaining <= 0:
                break

            await asyncio.sleep(min(self.poll_interval, remaining))

        return batch

    async def _worker_loop(self, worker_index: int) -> None:
        active_requests: List[ActiveRequest] = []
        self.logger.info("Worker %s started", worker_index)

        while True:
            if len(active_requests) < self.max_batch_size:
                new_requests = await self.collect_requests(
                    timeout=self.collection_window if not active_requests else 0.0,
                    max_batch=self.max_batch_size - len(active_requests),
                    wait_for_first=not active_requests,
                )
                active_requests.extend(await self._activate_requests(new_requests))

            if not active_requests:
                continue

            next_round: List[ActiveRequest] = []
            for active_request in active_requests:
                if active_request.request.session.cancelled:
                    await self._close_iterator(active_request.iterator)
                    continue

                chunk = await self._advance_iterator(active_request)
                if chunk is _ITERATION_FINISHED:
                    active_request.request.session.finish()
                    await self._close_iterator(active_request.iterator)
                    continue

                if active_request.request.session.cancelled:
                    await self._close_iterator(active_request.iterator)
                    continue

                next_round.append(active_request)

            active_requests = next_round

    async def _activate_requests(
        self,
        requests: List[ScheduledRequest],
    ) -> List[ActiveRequest]:
        active_requests: List[ActiveRequest] = []
        for request in requests:
            if request.session.cancelled:
                continue

            try:
                iterator = await self.run_sync(self.build_stream, request.payload)
            except Exception as exc:
                request.session.fail(exc)
                continue

            active_requests.append(ActiveRequest(request=request, iterator=iterator))
            self.logger.info(
                "[%s] activated (active=%s, queued=%s)",
                request.request_id,
                len(active_requests),
                self.request_queue.qsize(),
            )

        return active_requests

    async def _advance_iterator(self, active_request: ActiveRequest) -> Any:
        try:
            chunk = await self.run_sync(_next_or_sentinel, active_request.iterator)
        except Exception as exc:
            active_request.request.session.fail(exc)
            await self._close_iterator(active_request.iterator)
            return _ITERATION_FINISHED

        if chunk is _ITERATION_FINISHED:
            return chunk

        try:
            serialized_chunk = self.serialize_chunk(chunk)
        except Exception as exc:
            active_request.request.session.fail(exc)
            await self._close_iterator(active_request.iterator)
            return _ITERATION_FINISHED

        await active_request.request.session.put(serialized_chunk)
        return serialized_chunk

    async def _close_iterator(self, iterator: Iterator[Any]) -> None:
        close = getattr(iterator, "close", None)
        if close is None:
            return

        with contextlib.suppress(Exception):
            await self.run_sync(close)


async def iterate_session_chunks(session: StreamSession) -> AsyncGenerator[bytes, None]:
    while True:
        chunk = await session.queue.get()
        if chunk is _SESSION_FINISHED:
            break
        yield chunk


async def collect_session_bytes(session: StreamSession) -> bytes:
    audio = bytearray()
    async for chunk in iterate_session_chunks(session):
        audio.extend(chunk)

    if session.error is not None:
        raise session.error

    return bytes(audio)
