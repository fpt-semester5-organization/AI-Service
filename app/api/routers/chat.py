from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.chat import ChatAnswerRequest, ChatAnswerResponse

router = APIRouter(prefix="/chat", dependencies=[Depends(require_internal_key)])


@router.post("/answer", response_model=ChatAnswerResponse, response_model_by_alias=True)
async def answer(body: ChatAnswerRequest, request: Request) -> ChatAnswerResponse:
    return await request.app.state.container.answerer.answer(body)
