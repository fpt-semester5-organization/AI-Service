"""Test tích hợp với Qdrant thật. Đặt QDRANT_TEST_URL (mặc định http://localhost:6333); bỏ qua nếu không kết nối được.
Dùng collection ngẫu nhiên và xóa sau khi chạy, không đụng dữ liệu khác trên cùng Qdrant."""
import os
import time
import uuid
from datetime import datetime, timezone

import pytest
from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.core.errors import AiServiceError
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.indexer import Indexer
from app.services.rag.qdrant_store import QdrantStore
from app.services.rag.retriever import Retriever
from app.services.rag.types import Chunk, point_id
from tests.fakes import FakeLLM, hash_vector

pytestmark = pytest.mark.qdrant
URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")
NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


async def _reachable() -> bool:
    c = AsyncQdrantClient(url=URL, timeout=2)
    try:
        await c.get_collections()
        return True
    except Exception:
        return False
    finally:
        await c.close()


@pytest.fixture
async def store():
    if not await _reachable():
        pytest.skip(f"Không kết nối được Qdrant ở {URL}")
    s = Settings(qdrant_url=URL, qdrant_collection=f"vl_test_{uuid.uuid4().hex[:8]}", rag_timeout_seconds=3, _env_file=None)
    st = QdrantStore(s)
    yield st
    cleaner = AsyncQdrantClient(url=URL, timeout=5)  # st có thể đã bị đóng bởi lifespan của app trong test HTTP
    try:
        await cleaner.delete_collection(s.qdrant_collection)
    finally:
        await cleaner.close()


def chunks(*texts):
    return [Chunk(i, t, len(t.split())) for i, t in enumerate(texts)]


async def test_creates_collection_cosine_768_with_payload_indexes(store):
    await store.ensure_collection()
    info = await store._client.get_collection(store._collection)
    assert info.config.params.vectors.size == 768 and info.config.params.vectors.distance.value == "Cosine"
    assert {"sourceType", "sourceId"} <= set(info.payload_schema)
    await store.ensure_collection()  # gọi lại không lỗi


async def test_upsert_three_points_then_search_returns_expected_order(store):
    docs = {"1": "ăn chay đạm đậu phụ đậu lăng", "2": "ăn chay đậu nành vitamin b12", "3": "ăn chay canxi hạnh nhân cải xoăn"}
    for sid, text in docs.items():
        await store.upsert_source("NUTRITION_DOC", sid, f"Tài liệu {sid}", chunks(text), [hash_vector(text)], NOW)
    hits = await store.search(hash_vector("ăn chay đạm đậu phụ"), top_k=5, min_score=0.01)
    assert [h.source_id for h in hits] == ["1", "2", "3"]
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
    assert hits[0].source_type == "NUTRITION_DOC" and hits[0].title == "Tài liệu 1"


async def test_min_score_filters_unrelated(store):
    await store.upsert_source("RECIPE", "5", "Canh", chunks("canh nấm đậu phụ"), [hash_vector("canh nấm đậu phụ")], NOW)
    assert await store.search(hash_vector("giá vàng hôm nay bao nhiêu"), 5, 0.55) == []


async def test_point_ids_are_deterministic_and_rerun_does_not_duplicate(store):
    vec = hash_vector("a b c")
    for _ in range(2):
        await store.upsert_source("POST", "9", "Bài", chunks("a b c", "d e f"), [vec, vec], NOW)
    total, by = await store.stats()
    assert total == 2 and by["POST"] == 2
    got = await store._client.retrieve(store._collection, ids=[point_id("POST", "9", 0), point_id("POST", "9", 1)])
    assert len(got) == 2 and got[0].payload["sourceId"] == "9"
    assert set(got[0].payload) == {"sourceType", "sourceId", "title", "text", "chunkIndex", "updatedAt"}
    assert got[0].payload["updatedAt"].endswith("Z")


async def test_shorter_content_drops_stale_chunks(store):
    vec = hash_vector("x")
    await store.upsert_source("RECIPE", "7", "R", chunks("a", "b", "c"), [vec] * 3, NOW)
    await store.upsert_source("RECIPE", "7", "R", chunks("a"), [vec], NOW)
    assert (await store.stats())[0] == 1


async def test_delete_removes_every_chunk_of_source_only(store):
    vec = hash_vector("x")
    await store.upsert_source("RECIPE", "7", "R", chunks("a", "b"), [vec] * 2, NOW)
    await store.upsert_source("RECIPE", "8", "R", chunks("a"), [vec], NOW)
    assert await store.delete_source("RECIPE", "7") == 2
    assert await store.list_source_ids("RECIPE") == {"8"}
    assert await store.delete_source("RECIPE", "7") == 0


async def test_same_id_different_type_are_separate_sources(store):
    vec = hash_vector("x")
    await store.upsert_source("RECIPE", "1", "R", chunks("a"), [vec], NOW)
    await store.upsert_source("POST", "1", "P", chunks("a"), [vec], NOW)
    await store.delete_source("RECIPE", "1")
    assert (await store.stats())[1] == {**{t: 0 for t in (await store.stats())[1]}, "POST": 1}


async def test_qdrant_down_gives_503_quickly():
    s = Settings(qdrant_url="http://127.0.0.1:1", qdrant_collection="x", rag_timeout_seconds=3, _env_file=None)
    st = QdrantStore(s)
    started = time.perf_counter()
    with pytest.raises(AiServiceError) as e:
        await st.search(hash_vector("x"), 5, 0.5)
    assert e.value.status_code == 503 and e.value.error_code == "QDRANT_UNAVAILABLE"
    assert time.perf_counter() - started < 6
    assert await st.is_up() is False


async def test_end_to_end_index_and_retrieve_with_real_chunker(store):
    llm = FakeLLM()
    ix = Indexer(Chunker(llm.count_tokens, chunk_tokens=30, overlap_tokens=5), Embedder(llm), store)
    long_text = " ".join(f"Câu số {i} nói về đậu phụ và đạm thực vật." for i in range(40))
    r = await ix.upsert("NUTRITION_DOC", "1", "Đạm thực vật", long_text, NOW)
    assert r.chunks > 1
    hits = await Retriever(Embedder(llm), store, top_k=5, min_score=0.1).search("đậu phụ đạm thực vật")
    assert hits and all(h.source_id == "1" for h in hits) and len(hits) <= 5


async def test_full_pipeline_through_http_with_real_qdrant(store):
    """HTTP → /rag/upsert → /chat/answer → /rag/delete với Qdrant thật (Gemini giả)."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import HEADERS

    s = Settings(internal_api_key="test-key", google_cloud_project="t", qdrant_url=URL,
                 qdrant_collection=store._collection, rag_min_score=0.2, _env_file=None)
    llm = FakeLLM('{"answer": "Khoảng 0,8 g/kg [S1].", "recommendedRecipeIds": []}')
    with TestClient(create_app(s, llm=llm, store=store)) as c:
        body = {"sourceType": "NUTRITION_DOC", "sourceId": "1", "title": "Đạm thực vật",
                "text": "Người ăn chay cần khoảng 0,8 gam đạm mỗi kg cân nặng mỗi ngày."}
        assert c.post("/rag/upsert", json=body, headers=HEADERS).status_code == 200
        r = c.post("/chat/answer", json={"question": "Người ăn chay cần bao nhiêu gam đạm mỗi kg cân nặng"}, headers=HEADERS).json()
        assert r["ragUsed"] is True and r["sources"][0]["sourceId"] == "1"
        assert c.get("/rag/status", headers=HEADERS).json()["bySource"]["NUTRITION_DOC"] == 1
        assert c.post("/rag/delete", json={"sourceType": "NUTRITION_DOC", "sourceId": "1"}, headers=HEADERS).json() == {"deleted": 1}
        r = c.post("/chat/answer", json={"question": "Người ăn chay cần bao nhiêu gam đạm mỗi kg cân nặng"}, headers=HEADERS).json()
        assert r["ragUsed"] is False and r["sources"] == []


async def test_chat_returns_503_within_budget_when_qdrant_is_down():
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import HEADERS

    s = Settings(internal_api_key="test-key", google_cloud_project="t", qdrant_url="http://127.0.0.1:1",
                 rag_timeout_seconds=3, _env_file=None)
    with TestClient(create_app(s, llm=FakeLLM())) as c:
        t = time.perf_counter()
        r = c.post("/chat/answer", json={"question": "đạm thực vật"}, headers=HEADERS)
        assert r.status_code == 503 and r.json()["errorCode"] == "QDRANT_UNAVAILABLE"
        assert time.perf_counter() - t < 4
