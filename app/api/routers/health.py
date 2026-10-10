from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.rag import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, response_model_by_alias=True, dependencies=[Depends(require_internal_key)])
async def health(request: Request) -> HealthResponse:
    """200 khi tiến trình sống; trạng thái Qdrant/Gemini chỉ báo trong thân phản hồi (AVL-2)."""
    c = request.app.state.container
    return HealthResponse(
        status="ok",
        qdrant="up" if await c.store.is_up() else "down",
        gemini="configured" if c.llm.configured() else "missing")
