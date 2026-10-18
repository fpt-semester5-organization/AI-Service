"""Router xử lý phân loại thuần chay qua mã vạch (FN-BAR-01)."""
from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.barcode import BarcodeClassifyRequest, BarcodeClassifyResponse

router = APIRouter(prefix="/barcode", tags=["barcode"], dependencies=[Depends(require_internal_key)])


@router.post("/classify", response_model=BarcodeClassifyResponse, response_model_by_alias=True)
async def classify_barcode(body: BarcodeClassifyRequest, request: Request) -> BarcodeClassifyResponse:
    """Phân loại thuần chay cho sản phẩm mã vạch đóng gói bằng Gemini (FN-BAR-01).

    Chỉ chấp nhận gọi nội bộ từ Backend qua header X-Internal-Api-Key.
    """
    classifier = request.app.state.container.barcode_classifier
    return await classifier.classify(body)
