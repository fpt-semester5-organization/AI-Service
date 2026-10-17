"""Schema JSON camelCase của /stt/* (Sprint Plan Phụ lục B.3)."""
from pydantic import Field

from app.schemas.chat import Audit, CamelModel, Usage

__all__ = ["Audit", "ExtractedIngredient", "ExtractedRecipe", "SttHealthResponse", "TranscribeResponse",
           "Usage", "VideoToRecipeRequest", "VideoToRecipeResponse"]


class VideoToRecipeRequest(CamelModel):
    video_url: str = Field(min_length=8, max_length=2048)


class TranscribeResponse(CamelModel):
    duration_seconds: int
    language: str
    stt_model_version: str
    raw_transcript: str


class ExtractedIngredient(CamelModel):
    name: str
    quantity: float | None = None
    unit: str | None = None


class ExtractedRecipe(CamelModel):
    title: str
    servings: int | None = None
    ingredients: list[ExtractedIngredient]
    steps: list[str]


class VideoToRecipeResponse(CamelModel):
    duration_seconds: int
    stt_model_version: str
    raw_transcript: str
    recipe: ExtractedRecipe
    usage: Usage
    audit: Audit


class SttHealthResponse(CamelModel):
    status: str
    whisper: str
    gemini: str
