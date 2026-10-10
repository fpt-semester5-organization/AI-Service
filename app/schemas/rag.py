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
    last_indexed_at: datetime | None
    failed: int


class HealthResponse(CamelModel):
    status: str
    qdrant: str
    gemini: str
