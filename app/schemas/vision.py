"""Schema JSON camelCase của /vision/detect-ingredients (Sprint Plan Phụ lục B.3, FN-VIS-01)."""
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.chat import Audit, CamelModel, Usage


class DetectedIngredient(CamelModel):
    name: str
    confidence: float
    freshness_status: Literal["FRESH", "NON-FRESH"]
    food_group: str | None = None
    is_animal_product: bool = False
    is_red_meat: bool = False
    box2d: list[int] | None = Field(default=None, alias="box2d")

    @model_validator(mode="before")
    @classmethod
    def normalize_box_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            if "box_2d" in data and "box2d" not in data:
                data["box2d"] = data["box_2d"]
            elif "box2D" in data and "box2d" not in data:
                data["box2d"] = data["box2D"]
        return data

    @property
    def box_2d(self) -> list[int] | None:
        return self.box2d


class DetectIngredientsResponse(CamelModel):
    items: list[DetectedIngredient]
    # B.3 chỉ liệt kê items; usage/audit là phần bổ sung để Backend ghi AIQuotaLogs/AuditLogs như các endpoint AI khác.
    usage: Usage
    audit: Audit
