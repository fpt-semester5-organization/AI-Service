"""Bản giả để test không cần mạng: LLM băm từ vào vector (độ giống theo từ chung) và kho vector trong bộ nhớ."""
import hashlib
import math
import re
from datetime import datetime

from app.core.config import Settings
from app.core.errors import qdrant_unavailable
from app.core.gemini_client import Generation, Message
from app.services.rag.types import SOURCE_TYPES, Chunk, Hit

DIM = 768
_WORD = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def hash_vector(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    for w in words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeLLM:
    def __init__(self, reply: str | None = None, *, fail: Exception | None = None):
        self.reply = reply
        self.fail = fail
        self.embed_calls: list[tuple[str, str]] = []
        self.count_calls = 0
        self.generate_calls: list[tuple[str, list[Message]]] = []
        self.generate_kwargs: list[dict] = []   # temperature, max_output_tokens, json_schema của từng lần gọi

    def configured(self) -> bool:
        return True

    async def embed(self, text: str, *, task: str) -> list[float]:
        if self.fail:
            raise self.fail
        self.embed_calls.append((task, text))
        return hash_vector(text)

    async def count_tokens(self, text: str) -> int:
        self.count_calls += 1
        return len(words(text))

    async def generate(self, system, messages, *, temperature, max_output_tokens, json_schema=None) -> Generation:
        if self.fail:
            raise self.fail
        self.generate_calls.append((system, messages))
        self.generate_kwargs.append({"temperature": temperature, "max_output_tokens": max_output_tokens,
                                     "json_schema": json_schema})
        text = self.reply if self.reply is not None else '{"answer": "Trả lời mẫu [S1].", "recommendedRecipeIds": []}'
        return Generation(text=text, model="fake-model", prompt_tokens=10, completion_tokens=5,
                          latency_ms=3, temperature=temperature)


class InMemoryStore:
    """Giao diện giống QdrantStore, dùng cho test API không cần Docker."""

    def __init__(self, settings: Settings | None = None, *, down: bool = False):
        self.points: dict[tuple[str, str, int], tuple[list[float], dict]] = {}
        self.down = down

    def _check(self):
        if self.down:
            raise qdrant_unavailable()

    async def ensure_collection(self): self._check()
    async def is_up(self): return not self.down
    async def close(self): ...

    async def upsert_source(self, source_type, source_id, title, chunks: list[Chunk], vectors, updated_at: datetime):
        self._check()
        for k in [k for k in self.points if k[:2] == (source_type, source_id) and k[2] >= len(chunks)]:
            del self.points[k]
        for c, v in zip(chunks, vectors):
            self.points[(source_type, source_id, c.index)] = (v, {
                "title": title, "text": c.text})

    async def delete_source(self, source_type, source_id) -> int:
        self._check()
        ks = [k for k in self.points if k[:2] == (source_type, source_id)]
        for k in ks:
            del self.points[k]
        return len(ks)

    async def search(self, vector, top_k, min_score) -> list[Hit]:
        self._check()
        scored = []
        for (st, sid, idx), (v, p) in self.points.items():
            s = sum(a * b for a, b in zip(vector, v))
            if s >= min_score:
                scored.append(Hit(st, sid, p["title"], p["text"], s, idx))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]

    async def list_source_ids(self, source_type) -> set[str]:
        self._check()
        return {k[1] for k in self.points if k[0] == source_type}

    async def stats(self):
        self._check()
        by = {t: sum(1 for k in self.points if k[0] == t) for t in SOURCE_TYPES}
        return sum(by.values()), by
