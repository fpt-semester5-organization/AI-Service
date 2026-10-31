"""Schema JSON camelCase của /meal/generate (Sprint Plan Phụ lục B.3, FN-MEAL-01)."""
from typing import Literal

from pydantic import Field

from app.schemas.chat import Audit, CamelModel, CandidateRecipe, Usage

DayOfWeek = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MealType = Literal["BREAKFAST", "LUNCH", "DINNER"]


class MealGenerateRequest(CamelModel):
    target_daily_calories: int = Field(gt=0, le=10000)
    # Backend đã lọc trước theo trường phái + dị ứng; AI service chỉ được chọn trong danh sách này.
    candidate_recipes: list[CandidateRecipe] = Field(min_length=1, max_length=500)
    fridge_ingredients: list[str] = Field(default_factory=list, max_length=100)
    custom_instructions: str | None = Field(default=None, max_length=200)   # BR-USR-MEAL-CI-01
    attempt: int = Field(default=1, ge=1, le=3)


class MealSlot(CamelModel):
    day_of_week: DayOfWeek
    meal_type: MealType
    recipe_id: int


class MealGenerateResponse(CamelModel):
    slots: list[MealSlot]
    usage: Usage
    audit: Audit
