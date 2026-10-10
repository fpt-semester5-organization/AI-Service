"""Kiểu dùng chung của RAG."""
import re
import uuid
from dataclasses import dataclass

SOURCE_TYPES = (
    "RECIPE", "POST", "VIDEO_TRANSCRIPT", "VIDEO_RECIPE",
    "NUTRITION_DOC", "E_NUMBER", "ALLERGEN", "INGREDIENT",
)
SOURCE_ID_RE = re.compile(r"^\d{1,18}$")  # sourceId luôn là chuỗi chữ số (D119)
# Namespace cố định của UUIDv5 (Sprint Plan Phụ lục B.2)
POINT_NAMESPACE = uuid.UUID("6f1c0f62-4f0b-4c7e-9a52-1d6d9a0c2b11")


def point_id(source_type: str, source_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source_type}:{source_id}:{chunk_index}"))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    tokens: int


@dataclass(frozen=True)
class Hit:
    source_type: str
    source_id: str
    title: str
    text: str
    score: float
    chunk_index: int
