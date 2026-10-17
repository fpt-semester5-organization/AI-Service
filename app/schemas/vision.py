"""Schema JSON camelCase của /vision/detect-ingredients (Sprint Plan Phụ lục B.3, FN-VIS-01)."""
from typing import Literal

from app.schemas.chat import Audit, CamelModel, Usage


class DetectedIngredient(CamelModel):
    name: str
    confidence: float
    freshness_status: Literal["FRESH", "NON-FRESH"]
    food_group: str | None = None
    is_animal_product: bool = False
    is_red_meat: bool = False


class DetectIngredientsResponse(CamelModel):
    items: list[DetectedIngredient]
    # B.3 chỉ liệt kê items; usage/audit là phần bổ sung để Backend ghi AIQuotaLogs/AuditLogs như các endpoint AI khác.
    usage: Usage
    audit: Audit
