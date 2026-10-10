"""Tách văn bản thành đoạn ≈ 500 token, chồng lấn 50 (D99), đếm token bằng `count_tokens` thật (D120).

Cách làm: tách thành câu; đếm token từng câu (có cache, chạy song song có giới hạn); cộng dồn để ghép đoạn.
Tổng cộng dồn xấp xỉ số token thật của cả đoạn (sai lệch nhỏ ở ranh giới câu). Nguồn ngắn (≤ ngưỡng) là 1 đoạn và
được đếm đúng bằng 1 lời gọi. Tiêu đề được đưa vào đầu mỗi đoạn.
"""
import asyncio
import re
from typing import Awaitable, Callable

from app.services.rag.types import Chunk

TokenCounter = Callable[[str], Awaitable[int]]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def split_units(text: str) -> list[str]:
    units: list[str] = []
    for part in _SENTENCE_SPLIT.split(text.strip()):
        part = part.strip()
        if part:
            units.append(part)
    return units


class Chunker:
    def __init__(self, count_tokens: TokenCounter, *, chunk_tokens: int = 500, overlap_tokens: int = 50,
                 concurrency: int = 4):
        if overlap_tokens >= chunk_tokens:
            raise ValueError("overlap phải nhỏ hơn kích thước đoạn")
        self._count = count_tokens
        self._size = chunk_tokens
        self._overlap = overlap_tokens
        self._sem = asyncio.Semaphore(concurrency)
        self._cache: dict[str, int] = {}

    async def _tokens(self, text: str) -> int:
        if text in self._cache:
            return self._cache[text]
        async with self._sem:
            n = await self._count(text)
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[text] = n
        return n

    async def chunk(self, title: str, text: str) -> list[Chunk]:
        title = title.strip()
        body = text.strip()
        if not body:
            return []
        head = f"{title}\n\n" if title else ""

        whole = head + body
        total = await self._tokens(whole)
        if total <= self._size:
            return [Chunk(0, whole, total)]

        head_tokens = await self._tokens(title) if title else 0
        budget = max(self._size - head_tokens - 2, self._overlap + 1)
        units = await self._shrink_units(split_units(body), budget)
        counts = await asyncio.gather(*(self._tokens(u) for u in units))

        chunks: list[Chunk] = []
        i = 0
        while i < len(units):
            j, used = i, 0
            while j < len(units) and (used + counts[j] <= budget or j == i):
                used += counts[j]
                j += 1
            chunks.append(Chunk(len(chunks), head + " ".join(units[i:j]), used + head_tokens))
            if j >= len(units):
                break
            # chồng lấn: lùi lại các câu cuối tới tối đa `overlap` token (luôn tiến ít nhất 1 câu)
            back, acc = j, 0
            while back - 1 > i and acc + counts[back - 1] <= self._overlap:
                acc += counts[back - 1]
                back -= 1
            i = max(back, i + 1)
        return chunks

    async def _shrink_units(self, units: list[str], budget: int) -> list[str]:
        """Câu dài hơn ngân sách thì cắt theo từ."""
        out: list[str] = []
        counts = await asyncio.gather(*(self._tokens(u) for u in units))
        for u, n in zip(units, counts):
            if n <= budget:
                out.append(u)
                continue
            words = u.split()
            parts = -(-n // budget)
            size = max(len(words) // parts, 1)
            out.extend(" ".join(words[k:k + size]) for k in range(0, len(words), size))
        return out
