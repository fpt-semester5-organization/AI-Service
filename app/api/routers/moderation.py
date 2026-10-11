from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.moderation import ModerationCheckRequest, ModerationCheckResponse

router = APIRouter(prefix="/moderation", dependencies=[Depends(require_internal_key)])


@router.post("/check", response_model=ModerationCheckResponse, response_model_by_alias=True)
async def check(body: ModerationCheckRequest, request: Request) -> ModerationCheckResponse:
    return await request.app.state.container.moderator.check(body)
