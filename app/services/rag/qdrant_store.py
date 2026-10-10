"""Kho vector Qdrant: collection `veggielife_knowledge` (Cosine), upsert/xóa/tìm. Mọi lỗi → 503 QDRANT_UNAVAILABLE."""
import asyncio
import logging
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings
from app.core.errors import AiServiceError, qdrant_unavailable
from app.services.rag.types import SOURCE_TYPES, Chunk, Hit, point_id

log = logging.getLogger("veggielife.qdrant")


def _source_filter(source_type: str, source_id: str | None = None, min_chunk_index: int | None = None) -> models.Filter:
    must: list[models.Condition] = [models.FieldCondition(key="sourceType", match=models.MatchValue(value=source_type))]
    if source_id is not None:
        must.append(models.FieldCondition(key="sourceId", match=models.MatchValue(value=source_id)))
    if min_chunk_index is not None:
        must.append(models.FieldCondition(key="chunkIndex", range=models.Range(gte=min_chunk_index)))
    return models.Filter(must=must)


class QdrantStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None):
        self._s = settings
        self._collection = settings.qdrant_collection
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
            timeout=int(max(settings.rag_timeout_seconds, 5)))
        self._ready = False
        self._lock = asyncio.Lock()

    async def _guard(self, coro, timeout: float | None = None):
        try:
            return await asyncio.wait_for(coro, timeout=timeout or max(self._s.rag_timeout_seconds, 5))
        except AiServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("Qdrant lỗi: %r", exc)
            raise qdrant_unavailable() from exc

    async def ensure_collection(self) -> None:
        """Tạo collection + chỉ mục payload nếu chưa có (gọi khi khởi động và khi dùng lần đầu)."""
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            exists = await self._guard(self._client.collection_exists(self._collection))
            if not exists:
                await self._guard(self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(size=self._s.embedding_dimensions, distance=models.Distance.COSINE)))
            for field in ("sourceType", "sourceId"):
                await self._guard(self._client.create_payload_index(
                    self._collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD))
            self._ready = True

    async def is_up(self) -> bool:
        try:
            await self._guard(self._client.get_collections(), timeout=2)
            return True
        except AiServiceError:
            return False

    async def upsert_source(self, source_type: str, source_id: str, title: str, chunks: list[Chunk],
                            vectors: list[list[float]], updated_at: datetime) -> None:
        await self.ensure_collection()
        stamp = updated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        points = [
            models.PointStruct(
                id=point_id(source_type, source_id, c.index), vector=v,
                payload={"sourceType": source_type, "sourceId": source_id, "title": title,
                         "text": c.text, "chunkIndex": c.index, "updatedAt": stamp})
            for c, v in zip(chunks, vectors)
        ]
        await self._guard(self._client.upsert(self._collection, points=points, wait=True))
        # xóa đoạn thừa của lần nạp trước (nội dung ngắn đi)
        await self._guard(self._client.delete(
            self._collection, points_selector=models.FilterSelector(
                filter=_source_filter(source_type, source_id, min_chunk_index=len(chunks))), wait=True))

    async def delete_source(self, source_type: str, source_id: str) -> int:
        await self.ensure_collection()
        flt = _source_filter(source_type, source_id)
        n = (await self._guard(self._client.count(self._collection, count_filter=flt, exact=True))).count
        if n:
            await self._guard(self._client.delete(
                self._collection, points_selector=models.FilterSelector(filter=flt), wait=True))
        return int(n)

    async def search(self, vector: list[float], top_k: int, min_score: float) -> list[Hit]:
        await self.ensure_collection()
        res = await self._guard(
            self._client.search(self._collection, query_vector=vector, limit=top_k,
                                score_threshold=min_score, with_payload=True),
            timeout=self._s.rag_timeout_seconds)
        hits: list[Hit] = []
        for r in res:
            p = r.payload or {}
            hits.append(Hit(p.get("sourceType", ""), str(p.get("sourceId", "")), p.get("title", ""),
                            p.get("text", ""), float(r.score), int(p.get("chunkIndex", 0))))
        return hits

    async def list_source_ids(self, source_type: str) -> set[str]:
        await self.ensure_collection()
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = await self._guard(self._client.scroll(
                self._collection, scroll_filter=_source_filter(source_type), limit=256,
                offset=offset, with_payload=["sourceId"], with_vectors=False))
            ids.update(str((p.payload or {}).get("sourceId")) for p in points)
            if offset is None:
                break
        return ids

    async def stats(self) -> tuple[int, dict[str, int]]:
        await self.ensure_collection()
        by = {}
        for t in SOURCE_TYPES:
            by[t] = int((await self._guard(self._client.count(
                self._collection, count_filter=_source_filter(t), exact=True))).count)
        return sum(by.values()), by

    async def close(self) -> None:
        await self._client.close()
