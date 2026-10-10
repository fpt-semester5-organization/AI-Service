"""Nạp NGUYÊN VĂN các sách trong knowledge/nutrition/0N_*/ vào Qdrant (NGOẠI LỆ so với D120, do người dùng quyết định).

Quy ước để không lẫn với tài liệu nhóm biên soạn:
- sourceType = NUTRITION_DOC, sourceId = book_no * 1000 + số thứ tự phần (>= 1001). Tài liệu nhóm dùng id < 1000.
- `POST /rag/reindex/nutrition` không xóa các điểm có id >= 1000 (xem nutrition_loader).
- Gỡ toàn bộ:  python -m scripts.ingest_books --delete

Chạy:
  python -m scripts.ingest_books --dry-run     # chỉ đếm phần/đoạn dự kiến, không gọi Vertex AI
  python -m scripts.ingest_books               # nạp (bỏ qua phần đã nạp; chạy lại an toàn)
  python -m scripts.ingest_books --only 1 3    # chỉ một số sách

Lưu ý: dùng bộ đếm token ƯỚC LƯỢNG (ký tự / 3,8) thay cho count_tokens thật để không gọi API hàng chục nghìn lần.
"""
import argparse
import asyncio
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.core.errors import AiServiceError
from app.core.gemini_client import GeminiClient
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.indexer import Indexer
from app.services.rag.qdrant_store import QdrantStore

log = logging.getLogger("ingest_books")
ROOT = Path("knowledge/nutrition")
BOOKS = {
    1: ("Becoming Vegan", "01_Becoming_Vegan_Comprehensive_Edition/Becoming_Vegan_Comprehensive_Edition.md"),
    2: ("How Not To Die (bản tiếng Đức)", "02_How_Not_To_Die/How_Not_To_Die.md"),
    3: ("The China Study", "03_The_China_Study/The_China_Study_Revised_and_Expanded_Edition.md"),
}
MIN_WORDS, MAX_WORDS = 400, 3000
CHARS_PER_TOKEN = 3.8


def clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)           # ảnh
    text = re.sub(r"\[\]\{#[^}]*\}", "", text)                   # neo html
    text = re.sub(r"\{[.#][^}]*\}", "", text)                    # thuộc tính pandoc
    text = re.sub(r"<[^>]+>", " ", text)                         # thẻ html
    text = re.sub(r"^\|.*\|\s*$", " ", text, flags=re.M)         # bảng markdown
    text = re.sub(r"[*_]{1,3}", "", text)                        # nhấn mạnh
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class Section:
    book_no: int
    index: int
    title: str
    text: str

    @property
    def source_id(self) -> str:
        return str(self.book_no * 1000 + self.index)


def _words(s: str) -> int:
    return len(s.split())


def split_sections(book_no: int, name: str, raw: str) -> list[Section]:
    # 1) tách theo tiêu đề cấp 1–3 nếu sách có; 2) gộp phần nhỏ; 3) cắt phần quá dài theo đoạn văn.
    parts: list[tuple[str, list[str]]] = [("", [])]
    for line in raw.split("\n"):
        m = re.match(r"^#{1,3}\s+(.*)$", line)
        if m and clean(m.group(1)).strip():
            parts.append((clean(m.group(1)).strip()[:80], []))
        else:
            parts[-1][1].append(line)
    blocks = [(h, clean("\n".join(ls))) for h, ls in parts]
    blocks = [(h, t) for h, t in blocks if _words(t) >= 30]

    merged: list[tuple[str, str]] = []
    for h, t in blocks:
        if merged and _words(merged[-1][1]) < MIN_WORDS:
            ph, pt = merged[-1]
            merged[-1] = (ph or h, pt + "\n\n" + (f"{h}\n" if h else "") + t)
        else:
            merged.append((h, t))

    out: list[Section] = []
    for h, t in merged:
        paras, buf, n = t.split("\n\n"), [], 0
        pieces: list[str] = []
        for p in paras:
            if n + _words(p) > MAX_WORDS and buf:
                pieces.append("\n\n".join(buf))
                buf, n = [], 0
            buf.append(p)
            n += _words(p)
        if buf:
            pieces.append("\n\n".join(buf))
        for k, piece in enumerate(pieces, start=1):
            idx = len(out) + 1
            label = h or f"phần {idx}"
            suffix = f" (tiếp {k})" if len(pieces) > 1 and k > 1 else ""
            out.append(Section(book_no, idx, f"{name} — {label}{suffix}", piece))
    return out


async def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def load_sections(only: list[int] | None) -> list[Section]:
    sections: list[Section] = []
    for no, (name, rel) in BOOKS.items():
        if only and no not in only:
            continue
        sections += split_sections(no, name, (ROOT / rel).read_text(encoding="utf-8"))
    return sections


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--limit", type=int, help="chỉ nạp N phần đầu (thử nghiệm)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    s = Settings()
    if a.dry_run:
        secs = load_sections(a.only)
        for no, (name, _) in BOOKS.items():
            mine = [x for x in secs if x.book_no == no]
            if mine:
                chunks = sum(math.ceil(len(x.text) / CHARS_PER_TOKEN / s.rag_chunk_tokens) for x in mine)
                print(f"{name}: {len(mine)} phần, ~{sum(_words(x.text) for x in mine)} từ, ~{chunks} đoạn")
        return 0

    llm = GeminiClient(s)
    store = QdrantStore(s)
    try:
        await store.ensure_collection()
        existing = await store.list_source_ids("NUTRITION_DOC")
        if a.delete:
            gone = [i for i in existing if i.isdigit() and int(i) >= 1000]
            for i in gone:
                await store.delete_source("NUTRITION_DOC", i)
            print(f"Đã xóa {len(gone)} phần sách khỏi Qdrant.")
            return 0
        embedder = Embedder(llm, concurrency=8)
        chunker = Chunker(approx_tokens, chunk_tokens=s.rag_chunk_tokens, overlap_tokens=s.rag_chunk_overlap,
                          concurrency=8)
        indexer = Indexer(chunker, embedder, store)
        secs = [x for x in load_sections(a.only) if x.source_id not in existing]
        if a.limit:
            secs = secs[: a.limit]
        print(f"Cần nạp {len(secs)} phần (đã bỏ qua phần có sẵn).")
        t0, chunks, tokens, failed = time.time(), 0, 0, []
        for n, sec in enumerate(secs, start=1):
            try:
                r = await indexer.upsert("NUTRITION_DOC", sec.source_id, sec.title, sec.text,
                                         datetime.now(timezone.utc))
                chunks += r.chunks
                tokens += r.embedding_tokens
            except AiServiceError as e:
                failed.append((sec.source_id, e.error_code))
                log.warning("Lỗi phần %s: %s", sec.source_id, e.error_code)
            if n % 10 == 0 or n == len(secs):
                print(f"{n}/{len(secs)} phần, {chunks} đoạn, ~{tokens} token, {int(time.time() - t0)} giây", flush=True)
        print(f"Xong: {chunks} đoạn, lỗi {len(failed)}: {failed[:10]}")
        return 1 if failed else 0
    finally:
        await store.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
