from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.stt import SttHealthResponse

router = APIRouter()


@router.get("/health", response_model=SttHealthResponse, response_model_by_alias=True,
            dependencies=[Depends(require_internal_key)])
async def health(request: Request) -> SttHealthResponse:
    """200 khi tiến trình sống; trạng thái Whisper/Gemini chỉ báo trong thân phản hồi."""
    c = request.app.state.stt
    return SttHealthResponse(status="ok", whisper="loaded" if c.engine.loaded else "not_loaded",
                             gemini="configured" if c.llm.configured() else "missing")
