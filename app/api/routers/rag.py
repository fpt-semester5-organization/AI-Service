from fastapi import APIRouter, Depends, Request

from app.core.security import require_internal_key
from app.schemas.rag import (
    DeleteRequest,
    DeleteResponse,
    FailedFile,
    ReindexNutritionResponse,
    StatusResponse,
    UpsertRequest,
    UpsertResponse,
)
from app.services.rag.nutrition_loader import reindex_nutrition

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
                          last_indexed_at=st.last_indexed_at, failed=st.failed)
