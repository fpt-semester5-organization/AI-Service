from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.errors import validation_error
from app.core.security import require_internal_key
from app.schemas.rag import (
    DeleteRequest,
    DeleteResponse,
    FailedFile,
    OcrPdfResponse,
    OcrUsage,
    ReindexNutritionResponse,
    StatusResponse,
    UpsertRequest,
    UpsertResponse,
    VisibilityRequest,
    VisibilityResponse,
)
from app.services.rag.nutrition_loader import BOOK_ID_FLOOR, reindex_nutrition

router = APIRouter(prefix="/rag", dependencies=[Depends(require_internal_key)])


@router.post("/upsert", response_model=UpsertResponse, response_model_by_alias=True)
async def upsert(body: UpsertRequest, request: Request) -> UpsertResponse:
    c = request.app.state.container
    r = await c.indexer.upsert(body.source_type, body.source_id, body.title, body.text, body.updated_at)
    return UpsertResponse(chunks=r.chunks, embedding_tokens=r.embedding_tokens)


@router.post("/delete", response_model=DeleteResponse, response_model_by_alias=True)
async def delete(body: DeleteRequest, request: Request) -> DeleteResponse:
    c = request.app.state.container
    return DeleteResponse(deleted=await c.indexer.delete(body.source_type, body.source_id))


@router.post("/visibility", response_model=VisibilityResponse, response_model_by_alias=True)
async def visibility(body: VisibilityRequest, request: Request) -> VisibilityResponse:
    """Admin bật/ẩn tài liệu: đổi payload `active`, đoạn bị ẩn không còn được tìm thấy khi chat."""
    c = request.app.state.container
    ids = list(dict.fromkeys(body.source_ids))
    if body.book_no is not None or body.source_id_prefix:
        existing = await c.store.list_source_ids(body.source_type)
        if body.book_no is not None:
            ids += [i for i in existing
                    if i.isdigit() and int(i) // BOOK_ID_FLOOR == body.book_no and int(i) % BOOK_ID_FLOOR > 0]
        if body.source_id_prefix:
            p = body.source_id_prefix
            ids += [i for i in existing if i == p or i.startswith(f"{p}_P")]
    return VisibilityResponse(updated=await c.store.set_active(body.source_type, ids, body.active))


@router.post("/ocr-pdf", response_model=OcrPdfResponse, response_model_by_alias=True)
async def ocr_pdf(request: Request, file: UploadFile | None = File(None)) -> OcrPdfResponse:
    """Nhận dạng chữ (OCR) PDF scan bằng Gemini để Admin nạp tài liệu dinh dưỡng vào kho tri thức."""
    if file is None:
        raise validation_error("Yêu cầu tải lên tệp PDF qua trường 'file'.")
    c = request.app.state.container
    data = await file.read(c.settings.ocr_max_upload_mb * 1024 * 1024 + 1)
    r = await c.pdf_ocr.ocr(data)
    return OcrPdfResponse(text=r.text, pages=r.pages, usage=OcrUsage(
        model=r.model, prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
        latency_ms=r.latency_ms, temperature=r.temperature))


@router.post("/reindex/nutrition", response_model=ReindexNutritionResponse, response_model_by_alias=True)
async def reindex(request: Request) -> ReindexNutritionResponse:
    c = request.app.state.container
    r = await reindex_nutrition(c.settings.knowledge_dir, c.indexer, c.store)
    return ReindexNutritionResponse(
        files=r.files, chunks=r.chunks, embedding_tokens=r.embedding_tokens,
        failed=[FailedFile(**f) for f in r.failed])


@router.get("/status", response_model=StatusResponse, response_model_by_alias=True)
async def status(request: Request) -> StatusResponse:
    c = request.app.state.container
    total, by_source = await c.store.stats()
    st = c.indexer.state
    return StatusResponse(collection=c.settings.qdrant_collection, total_points=total, by_source=by_source,
                          inactive_points=await c.store.inactive_count(),
                          last_indexed_at=st.last_indexed_at, failed=st.failed)
