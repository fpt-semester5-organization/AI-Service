"""Schema JSON camelCase của /rag/* (Sprint Plan Phụ lục B.3)."""
from datetime import datetime

from pydantic import Field

from app.schemas.chat import CamelModel


class UpsertRequest(CamelModel):
    source_type: str
    source_id: str
    title: str = Field(max_length=300)
    text: str = Field(max_length=200_000)
    updated_at: datetime | None = None


class UpsertResponse(CamelModel):
    chunks: int
    embedding_tokens: int


class DeleteRequest(CamelModel):
    source_type: str
    source_id: str


class DeleteResponse(CamelModel):
    deleted: int


class VisibilityRequest(CamelModel):
    """Bật/ẩn nguồn tri thức: theo danh sách sourceId, hoặc toàn bộ phần của một sách (sourceId = bookNo*1000 + k)."""
    source_type: str
    source_ids: list[str] = Field(default_factory=list, max_length=1000)
    book_no: int | None = Field(default=None, ge=1, le=999)
    # Tài liệu Admin tải lên được chia phần: sourceId = <docId>, <docId>_P2, <docId>_P3...
    source_id_prefix: str | None = Field(default=None, min_length=8, max_length=100)
    active: bool


class VisibilityResponse(CamelModel):
    updated: int


class OcrUsage(CamelModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    temperature: float


class OcrPdfResponse(CamelModel):
    """Văn bản nhận dạng từ PDF scan; usage cộng dồn mọi lô trang để Backend ghi chi phí LLMOps."""
    text: str
    pages: int
    usage: OcrUsage


class FailedFile(CamelModel):
    file: str
    reason: str


class ReindexNutritionResponse(CamelModel):
    files: int
    chunks: int
    embedding_tokens: int
    failed: list[FailedFile] = Field(default_factory=list)


class StatusResponse(CamelModel):
    collection: str
    total_points: int
    by_source: dict[str, int]
    inactive_points: int = 0
    last_indexed_at: datetime | None
    failed: int


class HealthResponse(CamelModel):
    status: str
    qdrant: str
    gemini: str
