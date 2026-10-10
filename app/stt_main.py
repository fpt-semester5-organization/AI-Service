"""stt-service (Cloud Run, D119): chỉ /health và /stt/*. Chạy: uvicorn app.stt_main:app"""
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from app.api.routers import stt, stt_health
from app.core.config import Settings, get_settings
from app.core.errors import register_handlers
from app.core.gemini_client import GeminiClient, LLMClient
from app.services.stt.pipeline import SttPipeline
from app.services.stt.whisper import WhisperEngine

log = logging.getLogger("veggielife.stt")


@dataclass
class SttContainer:
    settings: Settings
    llm: LLMClient
    engine: WhisperEngine
    pipeline: SttPipeline


def create_stt_app(settings: Settings | None = None, llm: LLMClient | None = None,
                   engine: WhisperEngine | None = None, *, preload: bool | None = None,
                   http_client: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    llm = llm or GeminiClient(settings)
    engine = engine or WhisperEngine(settings)
    container = SttContainer(settings, llm, engine, SttPipeline(settings, engine, llm, http_client))
    do_preload = preload if preload is not None else True

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if do_preload:
            engine.load()  # nạp model khi khởi động để yêu cầu đầu tiên không phải chờ
        yield

    app = FastAPI(title="VeggieLife STT Service", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs" if settings.enable_docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if settings.enable_docs else None)
    app.state.settings = settings
    app.state.stt = container
    register_handlers(app)
    app.include_router(stt_health.router)
    app.include_router(stt.router)
    return app


app = create_stt_app()
