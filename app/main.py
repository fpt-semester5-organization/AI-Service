"""Khởi tạo FastAPI app của ai-service (VPS). stt-service (Cloud Run) do Huỳnh Việt Phát bổ sung router /stt/* (D119)."""
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from app.api.routers import chat, health, rag
from app.core.config import Settings, get_settings
from app.core.errors import AiServiceError, register_handlers
from app.core.gemini_client import GeminiClient, LLMClient
from app.services.chat.answerer import ChatAnswerer
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.indexer import Indexer
from app.services.rag.qdrant_store import QdrantStore
from app.services.rag.retriever import Retriever

log = logging.getLogger("veggielife")


@dataclass
class Container:
    settings: Settings
    llm: LLMClient
    store: QdrantStore
    indexer: Indexer
    answerer: ChatAnswerer


def build_container(settings: Settings, llm: LLMClient | None = None, store: QdrantStore | None = None) -> Container:
    llm = llm or GeminiClient(settings)
    store = store or QdrantStore(settings)
    embedder = Embedder(llm, concurrency=settings.embedding_concurrency)
    chunker = Chunker(llm.count_tokens, chunk_tokens=settings.rag_chunk_tokens,
                      overlap_tokens=settings.rag_chunk_overlap, concurrency=settings.embedding_concurrency)
    indexer = Indexer(chunker, embedder, store)
    retriever = Retriever(embedder, store, top_k=settings.rag_top_k, min_score=settings.rag_min_score)
    return Container(settings, llm, store, indexer, ChatAnswerer(settings, llm, retriever))


def create_app(settings: Settings | None = None, llm: LLMClient | None = None,
               store: QdrantStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    container = build_container(settings, llm, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:  # Qdrant chưa sẵn sàng không làm sập service; collection được tạo lại khi dùng lần đầu
            await container.store.ensure_collection()
        except AiServiceError:
            log.warning("Qdrant chưa sẵn sàng khi khởi động; sẽ tạo collection khi dùng lần đầu.")
        yield
        await container.store.close()

    app = FastAPI(title="VeggieLife AI Service", version="0.1.0", lifespan=lifespan,
                  docs_url="/docs" if settings.enable_docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if settings.enable_docs else None)
    app.state.settings = settings
    app.state.container = container
    register_handlers(app)
    app.include_router(health.router)
    app.include_router(rag.router)
    app.include_router(chat.router)
    return app


app = create_app()
