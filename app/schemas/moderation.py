"""Schema JSON camelCase của /moderation/check (Sprint Plan Phụ lục B.3/B.4, FN-POST-03/04/07)."""
from typing import Literal

from pydantic import Field

from app.schemas.chat import Audit, CamelModel, Usage

ViolationType = Literal["NONE", "VIO_MEAT", "VIO_LABEL", "VIO_HATE", "VIO_SPAM"]


class ModerationCheckRequest(CamelModel):
    title: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=10_000)
    steps: list[str] = Field(default_factory=list, max_length=100)
    ingredient_names: list[str] = Field(default_factory=list, max_length=200)
    dietary_school_code: str | None = Field(default=None, max_length=50)
    image_urls: list[str] = Field(default_factory=list, max_length=20)


class ModerationCheckResponse(CamelModel):
    # Chỉ trả điểm và lý do; ngưỡng 85% và Red Alert do Backend quyết (README mục 5.5).
    violation_score: float = Field(ge=0, le=1)
    violation_type: ViolationType
    reasons: list[str]
    usage: Usage
    audit: Audit
