"""Tạo embedding: `RETRIEVAL_DOCUMENT` khi nạp, `RETRIEVAL_QUERY` khi hỏi.

gemini-embedding-001 trên Vertex chỉ nhận 1 văn bản mỗi lời gọi, nên gửi từng đoạn với số lời gọi song song có giới hạn.
"""
import asyncio

from app.core.gemini_client import LLMClient

DOCUMENT = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


class Embedder:
    def __init__(self, llm: LLMClient, *, concurrency: int = 4):
        self._llm = llm
        self._sem = asyncio.Semaphore(concurrency)

    async def _one(self, text: str, task: str) -> list[float]:
        async with self._sem:
            return await self._llm.embed(text, task=task)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(await asyncio.gather(*(self._one(t, DOCUMENT) for t in texts)))

    async def embed_query(self, text: str) -> list[float]:
        return await self._one(text, QUERY)
