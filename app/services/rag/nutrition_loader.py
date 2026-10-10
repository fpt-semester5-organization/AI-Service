"""Đọc `knowledge/nutrition/*.md` (front-matter YAML) và nạp lại toàn bộ (FN-CHAT-07, scope NUTRITION_DOC)."""
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter

from app.services.rag.indexer import Indexer
from app.services.rag.qdrant_store import QdrantStore

log = logging.getLogger("veggielife.nutrition")
_NAME_ID = re.compile(r"^(\d+)-")
BOOK_ID_FLOOR = 1000  # tài liệu nhóm biên soạn dùng id < 1000


@dataclass
class NutritionDoc:
    file: str
    id: int
    title: str
    text: str
    updated_at: datetime


@dataclass
class LoadResult:
    docs: list[NutritionDoc] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)


def _parse_updated(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def load_directory(directory: str | Path) -> LoadResult:
    result = LoadResult()
    seen: dict[int, str] = {}
    for path in sorted(Path(directory).glob("*.md")):
        if path.name.lower() == "readme.md" or path.name.startswith("_"):
            continue
        try:
            post = frontmatter.load(path)
            meta = post.metadata
            m = _NAME_ID.match(path.name)
            if not m:
                raise ValueError("tên tệp phải bắt đầu bằng '<id>-'")
            if not isinstance(meta.get("id"), int) or isinstance(meta.get("id"), bool):
                raise ValueError("front-matter thiếu id (số nguyên)")
            if meta["id"] != int(m.group(1)):
                raise ValueError(f"id {meta['id']} không trùng số đầu tên tệp {m.group(1)}")
            if meta["id"] in seen:
                raise ValueError(f"id {meta['id']} trùng với {seen[meta['id']]}")
            if not str(meta.get("title", "")).strip():
                raise ValueError("thiếu title")
            if not meta.get("sources"):
                raise ValueError("thiếu sources (cần ít nhất 1 nguồn)")
            body = post.content.strip()
            if not body:
                raise ValueError("nội dung rỗng")
            seen[meta["id"]] = path.name
            result.docs.append(NutritionDoc(path.name, meta["id"], str(meta["title"]).strip(), body,
                                            _parse_updated(meta.get("updated", date.today()))))
        except Exception as exc:  # noqa: BLE001 — báo từng tệp lỗi, vẫn nạp các tệp tốt
            result.failed.append({"file": path.name, "reason": str(exc)})
    return result


@dataclass
class ReindexResult:
    files: int
    chunks: int
    embedding_tokens: int
    failed: list[dict[str, str]]


async def reindex_nutrition(directory: str | Path, indexer: Indexer, store: QdrantStore) -> ReindexResult:
    loaded = load_directory(directory)
    chunks = tokens = 0
    for doc in loaded.docs:
        try:
            r = await indexer.upsert("NUTRITION_DOC", str(doc.id), doc.title, doc.text, doc.updated_at)
            chunks += r.chunks
            tokens += r.embedding_tokens
        except Exception as exc:  # noqa: BLE001
            if hasattr(exc, "status_code") and exc.status_code == 503:
                raise  # Qdrant sập: dừng, báo lỗi rõ ràng
            loaded.failed.append({"file": doc.file, "reason": getattr(exc, "message", str(exc))})
    # tệp bị xóa khỏi thư mục thì điểm tương ứng cũng phải biến mất; bỏ qua nếu có tệp lỗi (chưa biết id)
    if not loaded.failed:
        keep = {str(d.id) for d in loaded.docs}
        for stale in await store.list_source_ids("NUTRITION_DOC") - keep:
            if stale.isdigit() and int(stale) >= BOOK_ID_FLOOR:
                continue  # id >= 1000 do scripts/ingest_books.py nạp (sách nguyên văn), không phải tệp trong thư mục
            await store.delete_source("NUTRITION_DOC", stale)
    return ReindexResult(len(loaded.docs), chunks, tokens, loaded.failed)
