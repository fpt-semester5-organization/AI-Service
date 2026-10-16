from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.errors import validation_error
from app.core.security import require_internal_key
from app.schemas.vision import DetectIngredientsResponse

router = APIRouter(tags=["vision"], dependencies=[Depends(require_internal_key)])


@router.post("/vision/detect-ingredients", response_model=DetectIngredientsResponse, response_model_by_alias=True)
@router.post("/fridge/detect", response_model=DetectIngredientsResponse, response_model_by_alias=True, include_in_schema=False)
async def detect_ingredients(
    request: Request,
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
) -> DetectIngredientsResponse:
    """Nhận diện nguyên liệu từ ảnh (FN-VIS-01).

    Hỗ trợ multipart qua trường `file` (chuẩn D118/D119) hoặc `image` (client của Backend).
    """
    upload = file or image
    if upload is None:
        raise validation_error("Yêu cầu tải lên tệp ảnh qua trường 'file' hoặc 'image'.")

    c = request.app.state.container
    # Đọc tối đa MAX_UPLOAD_MB + 1 byte: đủ để biết là quá lớn mà không nạp cả tệp khổng lồ vào bộ nhớ.
    data = await upload.read(c.settings.max_upload_mb * 1024 * 1024 + 1)
    return await c.ingredient_detector.detect(data)
