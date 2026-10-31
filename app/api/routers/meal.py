from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.meal import MealGenerateRequest, MealGenerateResponse

router = APIRouter(prefix="/meal", dependencies=[Depends(require_internal_key)])


@router.post("/generate", response_model=MealGenerateResponse, response_model_by_alias=True)
async def generate(body: MealGenerateRequest, request: Request) -> MealGenerateResponse:
    return await request.app.state.container.meal_generator.generate(body)
