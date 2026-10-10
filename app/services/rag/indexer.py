"""Nạp một nguồn vào Qdrant: kiểm hợp lệ → tách đoạn → embedding → upsert (id UUIDv5 tất định)."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.errors import AiServiceError, invalid_source_type, validation_error
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.qdrant_store import QdrantStore
from app.services.rag.types import SOURCE_ID_RE, SOURCE_TYPES

log = logging.getLogger("veggielife.indexer")


@dataclass
class IndexState:
    last_indexed_at: datetime | None = None
    failed: int = 0


@dataclass
class UpsertResult:
    chunks: int
    embedding_tokens: int


class Indexer:
    def __init__(self, chunker: Chunker, embedder: Embedder, store: QdrantStore):
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self.state = IndexState()

    @staticmethod
    def validate(source_type: str, source_id: str, title: str, text: str) -> None:
        if source_type not in SOURCE_TYPES:  # chặn dữ liệu cá nhân: chỉ 8 loại công khai
            raise invalid_source_type(source_type)
        if not SOURCE_ID_RE.match(source_id or ""):
            raise validation_error("sourceId phải là chuỗi chữ số.")
        if not (title or "").strip():
            raise validation_error("title không được rỗng.")
        if not (text or "").strip():
            raise validation_error("text không được rỗng.")

    async def upsert(self, source_type: str, source_id: str, title: str, text: str,
                     updated_at: datetime | None = None) -> UpsertResult:
        self.validate(source_type, source_id, title, text)
        try:
            chunks = await self._chunker.chunk(title, text)
            vectors = await self._embedder.embed_documents([c.text for c in chunks])
            await self._store.upsert_source(source_type, source_id, title.strip(), chunks, vectors,
                                            updated_at or datetime.now(timezone.utc))
        except AiServiceError:
            self.state.failed += 1
            raise
        self.state.last_indexed_at = datetime.now(timezone.utc)
        return UpsertResult(len(chunks), sum(c.tokens for c in chunks))

    async def delete(self, source_type: str, source_id: str) -> int:
        if source_type not in SOURCE_TYPES:
            raise invalid_source_type(source_type)
        if not SOURCE_ID_RE.match(source_id or ""):
            raise validation_error("sourceId phải là chuỗi chữ số.")
        return await self._store.delete_source(source_type, source_id)
