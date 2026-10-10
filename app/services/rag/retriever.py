"""Tìm kiếm: embed câu hỏi (RETRIEVAL_QUERY) → top-K → lọc theo ngưỡng điểm."""
from app.services.rag.embedder import Embedder
from app.services.rag.qdrant_store import QdrantStore
from app.services.rag.types import Hit


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, *, top_k: int, min_score: float):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    async def search(self, query: str) -> list[Hit]:
        vector = await self._embedder.embed_query(query)
        hits = await self._store.search(vector, self._top_k, self._min_score)
        return [h for h in hits if h.score >= self._min_score]
