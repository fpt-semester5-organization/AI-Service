from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.security import require_internal_key
from app.schemas.stt import TranscribeResponse, VideoToRecipeRequest, VideoToRecipeResponse
from app.services.stt.pipeline import SttPipeline

router = APIRouter(prefix="/stt", dependencies=[Depends(require_internal_key)])


def _pipeline(request: Request) -> SttPipeline:
    return request.app.state.stt.pipeline


@router.post("/transcribe", response_model=TranscribeResponse, response_model_by_alias=True)
async def transcribe(request: Request, file: UploadFile = File(...)) -> TranscribeResponse:
    """Phiên âm tệp MP4/MOV tải thẳng lên (chỉ để thử/chẩn đoán; Cloud Run giới hạn thân yêu cầu ~32 MiB)."""
    return await _pipeline(request).transcribe_upload(file)


@router.post("/video-to-recipe", response_model=VideoToRecipeResponse, response_model_by_alias=True)
async def video_to_recipe(request: Request, body: VideoToRecipeRequest) -> VideoToRecipeResponse:
    return await _pipeline(request).video_to_recipe(body.video_url)
