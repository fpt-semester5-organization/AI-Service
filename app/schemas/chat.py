"""Schema JSON camelCase của /chat/answer (Sprint Plan Phụ lục B.3)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class ChatTurn(CamelModel):
    role: Literal["user", "model"]
    text: str = Field(min_length=1, max_length=4000)


class ChatProfile(CamelModel):
    dietary_school_code: str | None = None
    allergen_groups: list[str] = Field(default_factory=list)
    forbidden_food_groups: list[str] = Field(default_factory=list)


class CandidateRecipe(CamelModel):
    recipe_id: int
    title: str
    cook_time_minutes: int | float = 0
    prep_time_minutes: int | float = 0
    calories_kcal: float | int | None = None
    favorite_count: int | None = 0
    vote_count: int | None = 0
    view_count: int | None = 0
    difficulty_level: str | None = None
    category_name: str | None = None
    uses_fridge_item: bool = False


class ChatAnswerRequest(CamelModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)
    profile: ChatProfile | None = None
    candidate_recipes: list[CandidateRecipe] = Field(default_factory=list, max_length=200)
    user_id: int | None = None
    guest_id: str | None = None


class ChatSource(CamelModel):
    source_type: str
    source_id: str
    title: str
    score: float


class Usage(CamelModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    temperature: float


class Audit(CamelModel):
    prompt: str
    raw_response: str


class ChatAnswerResponse(CamelModel):
    answer: str
    sources: list[ChatSource]
    rag_used: bool
    recommended_recipe_ids: list[int]
    usage: Usage
    audit: Audit
