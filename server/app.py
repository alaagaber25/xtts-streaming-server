from contextlib import asynccontextmanager

from fastapi import FastAPI

from .logging_utils import get_logger
from .routes import meta_router, profiles_router, tts_router
from .runtime import InferenceScheduler
from .speaker_profiles import SpeakerProfileStore
from .settings import load_settings, resolve_device
from .state import AppState
from .xtts_service import XTTSService


def create_app() -> FastAPI:
    logger = get_logger("xtts")
    settings = load_settings(logger=logger)
    device = resolve_device(settings, logger=logger)
    service = XTTSService.create(settings=settings, device=device, logger=logger.getChild("service"))
    speaker_profiles = SpeakerProfileStore(
        settings.speaker_profiles_path,
        logger=logger.getChild("speaker_profiles"),
    )
    scheduler = InferenceScheduler(
        build_stream=service.build_stream_iterator,
        serialize_chunk=service.serialize_chunk,
        collection_window=settings.batch_collection_window,
        max_batch_size=settings.max_batch_size,
        poll_interval=settings.queue_poll_interval,
        worker_count=settings.gpu_worker_count,
        warmup=service.warmup,
        logger=logger.getChild("scheduler"),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await scheduler.start()
        logger.info(
            "Streaming scheduler ready "
            f"(batch_window={settings.batch_collection_window:.3f}s, max_batch_size={settings.max_batch_size})."
        )
        try:
            yield
        finally:
            await scheduler.stop()

    app = FastAPI(
        title="XTTS Streaming server",
        description="XTTS Streaming server",
        version="0.1.0",
        docs_url="/",
        lifespan=lifespan,
    )
    app.state.xtts_state = AppState(
        settings=settings,
        service=service,
        scheduler=scheduler,
        speaker_profiles=speaker_profiles,
        logger=logger,
    )

    app.include_router(tts_router)
    app.include_router(profiles_router)
    app.include_router(meta_router)

    logger.info("Running XTTS Server ...")
    return app
