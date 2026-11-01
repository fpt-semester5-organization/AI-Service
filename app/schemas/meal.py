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


class AiRecipeIngredient(CamelModel):
    ingredient_id: int = 0
    ingredient_name: str = ""
    quantity_needed: float = 0.0
    unit: str | None = None
    preparation_note: str | None = None
    is_optional: bool = False


class SwapCandidateRecipe(CamelModel):
    recipe_id: int
    title: str
    description: str | None = None
    instructions: str | None = None
    prep_time_minutes: int = 0
    cook_time_minutes: int = 0
    servings: int = 1
    difficulty_level: str | None = None
    calories_kcal: float = 0.0
    category_name: str | None = None
    uses_fridge_item: bool = False
    meal_type_preference: str | None = None
    ingredients: list[AiRecipeIngredient] = Field(default_factory=list)


class MealSwapByNoteRequest(CamelModel):
    current_recipe_id: int = 0
    swap_note: str = Field(min_length=1, max_length=500)
    dietary_school_code: str = "VEGAN"
    meal_type: str = "LUNCH"
    target_calories: float = 0.0
    excluded_recipe_ids: list[int] = Field(default_factory=list)
    candidate_recipes: list[SwapCandidateRecipe] = Field(min_length=1, max_length=200)


class MealSwapByNoteResponse(CamelModel):
    success: bool = True
    selected_recipe_id: int
    reason: str
    usage: Usage | None = None
