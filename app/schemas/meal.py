"""Schema JSON camelCase của /meal/generate (Sprint Plan Phụ lục B.3, FN-MEAL-01)."""
from typing import Literal

from pydantic import Field

from app.schemas.chat import Audit, CamelModel, CandidateRecipe, Usage

DayOfWeek = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MealType = Literal["BREAKFAST", "LUNCH", "DINNER"]


class MealGenerateRequest(CamelModel):
    target_daily_calories: float | int = Field(gt=0, le=10000)
    # Backend đã lọc trước theo trường phái + dị ứng; AI service chỉ được chọn trong danh sách này.
    candidate_recipes: list[CandidateRecipe] = Field(min_length=1, max_length=500)
    fridge_ingredients: list[str] = Field(default_factory=list, max_length=100)
    custom_instructions: str | None = Field(default=None, max_length=500)   # BR-USR-MEAL-CI-01
    duration_days: int = Field(default=7, ge=1, le=30)
    start_date: str | None = None
    attempt: int = Field(default=1, ge=1, le=3)
    # Tiêu chí cá nhân hóa dinh dưỡng từ Backend
    health_goal: str | None = None
    region_preference: str | None = None
    budget_tier: str | None = None
    dietary_school_code: str | None = None
    dietary_school_name: str | None = None
    requires_allium_free: bool | None = None
    target_weight_kg: float | None = None


class MealSlot(CamelModel):
    day_of_week: str
    meal_type: MealType
    recipe_id: int
    calories_kcal: float | None = None
    cook_time_minutes: int | None = None
    favorite_count: int | None = None
    match_reason: str | None = None


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


class SwapCandidateRecipe(CandidateRecipe):
    description: str | None = None
    instructions: str | None = None
    servings: int = 1
    meal_type_preference: str | None = None
    ingredients: list[AiRecipeIngredient] = Field(default_factory=list)


class SwapMealByNoteRequest(CamelModel):
    current_recipe_id: int = 0
    swap_note: str = Field(min_length=1, max_length=1000)
    dietary_school_code: str = "VEGAN"
    dietary_school_name: str | None = None
    meal_type: str = "LUNCH"
    target_calories: float | int = Field(default=0.0, ge=0, le=10000)
    excluded_recipe_ids: list[int] = Field(default_factory=list)
    candidate_recipes: list[SwapCandidateRecipe] = Field(min_length=1, max_length=500)
    health_goal: str | None = None
    region_preference: str | None = None
    budget_tier: str | None = None
    requires_allium_free: bool | None = None


class SwapMealByNoteResponse(CamelModel):
    success: bool = True
    selected_recipe_id: int
    reason: str
    calories_kcal: float | None = None
    cook_time_minutes: int | None = None
    favorite_count: int | None = None
    usage: Usage | None = None
    audit: Audit | None = None


# Aliases for backward compatibility
MealSwapByNoteRequest = SwapMealByNoteRequest
MealSwapByNoteResponse = SwapMealByNoteResponse
