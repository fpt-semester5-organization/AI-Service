"""Schema JSON camelCase cho endpoint /barcode/classify (FN-BAR-01)."""
from typing import Literal

from pydantic import Field

from app.schemas.chat import CamelModel


class BarcodeProduct(CamelModel):
    product_name: str = ""
    brand: str | None = None
    ingredients_text: str | None = None
    additives_tags: list[str] = Field(default_factory=list)
    allergens_tags: list[str] = Field(default_factory=list)
    traces_tags: list[str] = Field(default_factory=list)
    labels_tags: list[str] = Field(default_factory=list)
    ingredients_analysis_tags: list[str] = Field(default_factory=list)


class KnownAdditive(CamelModel):
    code: str
    name: str
    vegan_status: str


class BarcodeClassifyRequest(CamelModel):
    barcode: str = Field(min_length=8, max_length=14)
    product: BarcodeProduct
    known_additives: list[KnownAdditive] = Field(default_factory=list)


class DetectedAdditive(CamelModel):
    code: str
    name: str
    vegan_status: Literal["VEGAN", "NON_VEGAN", "DOUBTFUL"]
    reason: str = ""
    source: Literal["DICTIONARY", "MODEL"]


class BarcodeAllergen(CamelModel):
    name: str
    presence_type: Literal["CONTAINS", "MAY_CONTAIN"]


class BarcodeUsage(CamelModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    temperature: float
    latency_ms: int


class BarcodeAudit(CamelModel):
    prompt: str
    raw_result: str


class BarcodeClassifyResponse(CamelModel):
    vegan_verdict: Literal["VEGAN", "NON_VEGAN", "DOUBTFUL"]
    verdict_text: str
    detected_additives: list[DetectedAdditive] = Field(default_factory=list)
    allergens: list[BarcodeAllergen] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    usage: BarcodeUsage
    audit: BarcodeAudit
