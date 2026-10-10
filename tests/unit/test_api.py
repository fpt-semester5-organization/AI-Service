import json

from tests.conftest import HEADERS
from tests.fakes import FakeLLM

UP = {"sourceType": "RECIPE", "sourceId": "512", "title": "Canh nấm đậu phụ",
      "text": "Canh nấm đậu phụ giàu đạm thực vật. Nấm rơm, đậu phụ non, hành lá.", "updatedAt": "2026-10-10T09:00:00Z"}


def test_health_requires_key(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-Internal-Api-Key": "sai"}).status_code == 401
    r = client.get("/health", headers=HEADERS)
    assert r.status_code == 200 and r.json() == {"status": "ok", "qdrant": "up", "gemini": "configured"}


def test_unauthorized_body_has_error_code(client):
    r = client.post("/rag/upsert", json=UP)
    assert r.status_code == 401 and r.json()["errorCode"] == "UNAUTHORIZED"


def test_empty_server_key_rejects_everything(settings, llm, store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    settings.internal_api_key = ""
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        assert c.get("/health", headers={"X-Internal-Api-Key": ""}).status_code == 401


def test_upsert_status_delete_roundtrip(client):
    r = client.post("/rag/upsert", json=UP, headers=HEADERS)
    assert r.status_code == 200 and r.json() == {"chunks": 1, "embeddingTokens": r.json()["embeddingTokens"]}
    assert r.json()["embeddingTokens"] > 0
    s = client.get("/rag/status", headers=HEADERS).json()
    assert s["collection"] == "veggielife_knowledge" and s["totalPoints"] == 1 and s["bySource"]["RECIPE"] == 1
    assert s["lastIndexedAt"] is not None and s["failed"] == 0
    assert client.post("/rag/delete", json={"sourceType": "RECIPE", "sourceId": "512"}, headers=HEADERS).json() == {"deleted": 1}
    assert client.get("/rag/status", headers=HEADERS).json()["totalPoints"] == 0


def test_upsert_twice_does_not_duplicate(client):
    for _ in range(2):
        client.post("/rag/upsert", json=UP, headers=HEADERS)
    assert client.get("/rag/status", headers=HEADERS).json()["totalPoints"] == 1


def test_rejects_personal_data_source_types(client):
    for t in ("HEALTH_PROFILE", "CHAT_MESSAGE", "USER_ALLERGY", "recipe", ""):
        r = client.post("/rag/upsert", json={**UP, "sourceType": t}, headers=HEADERS)
        assert r.status_code == 400 and r.json()["errorCode"] == "INVALID_SOURCE_TYPE", t
    assert client.get("/rag/status", headers=HEADERS).json()["totalPoints"] == 0


def test_validation_errors(client):
    for bad in ({**UP, "sourceId": "01-dam.md"}, {**UP, "title": "  "}, {**UP, "text": ""}):
        r = client.post("/rag/upsert", json=bad, headers=HEADERS)
        assert r.status_code == 400 and r.json()["errorCode"] == "VALIDATION_ERROR"
    r = client.post("/rag/upsert", json={"sourceType": "RECIPE"}, headers=HEADERS)
    assert r.status_code == 400 and r.json()["errorCode"] == "VALIDATION_ERROR"


def test_qdrant_down_returns_503(settings, llm):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.fakes import InMemoryStore
    with TestClient(create_app(settings, llm=llm, store=InMemoryStore(down=True))) as c:
        r = c.post("/rag/upsert", json=UP, headers=HEADERS)
        assert r.status_code == 503 and r.json()["errorCode"] == "QDRANT_UNAVAILABLE"
        r = c.post("/chat/answer", json={"question": "đạm"}, headers=HEADERS)
        assert r.status_code == 503 and r.json()["errorCode"] == "QDRANT_UNAVAILABLE"
        assert c.get("/health", headers=HEADERS).json()["qdrant"] == "down"


def test_gemini_failure_maps_to_502(settings, store):
    from fastapi.testclient import TestClient

    from app.core.errors import gemini_error
    from app.main import create_app
    with TestClient(create_app(settings, llm=FakeLLM(fail=gemini_error()), store=store)) as c:
        r = c.post("/rag/upsert", json=UP, headers=HEADERS)
        assert r.status_code == 502 and r.json()["errorCode"] == "GEMINI_ERROR"
        assert c.get("/rag/status", headers=HEADERS).json()["failed"] == 1


def seed(client):
    client.post("/rag/upsert", json={"sourceType": "NUTRITION_DOC", "sourceId": "1", "title": "Đạm thực vật",
                "text": "Người ăn chay cần khoảng 0,8 gam đạm mỗi kg cân nặng. Đậu phụ đậu lăng tempeh là nguồn đạm thực vật tốt."},
                headers=HEADERS)
    client.post("/rag/upsert", json={"sourceType": "RECIPE", "sourceId": "512", "title": "Canh nấm đậu phụ",
                "text": "Canh nấm đậu phụ nấu nhanh giàu đạm."}, headers=HEADERS)


def test_chat_answer_shape_and_sources(settings, store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    llm = FakeLLM(json.dumps({"answer": "Khoảng 0,8 g/kg [S1].", "recommendedRecipeIds": [512, 999]}))
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        seed(c)
        r = c.post("/chat/answer", headers=HEADERS, json={
            "question": "Người ăn chay cần bao nhiêu gam đạm mỗi kg cân nặng?",
            "history": [{"role": "user", "text": "chào"}, {"role": "model", "text": "chào bạn"}],
            "profile": {"dietarySchoolCode": "VEGAN", "allergenGroups": ["PEANUT"], "forbiddenFoodGroups": ["EGG"]},
            "candidateRecipes": [{"recipeId": 512, "title": "Canh nấm đậu phụ", "cookTimeMinutes": 25}],
            "userId": 5})
        body = r.json()
        assert r.status_code == 200
        assert set(body) == {"answer", "sources", "ragUsed", "recommendedRecipeIds", "usage", "audit"}
        assert body["ragUsed"] is True and body["answer"].startswith("Khoảng")
        assert body["recommendedRecipeIds"] == [512]  # 999 không thuộc ứng viên → bị loại
        assert body["sources"][0] == {"sourceType": "NUTRITION_DOC", "sourceId": "1", "title": "Đạm thực vật",
                                      "score": body["sources"][0]["score"]}
        assert set(body["usage"]) == {"model", "promptTokens", "completionTokens", "latencyMs", "temperature"}
        assert "VL-SYS" in body["audit"]["prompt"] and body["audit"]["rawResponse"]
        # sources khớp các đoạn đã đưa vào lời nhắc, không lấy từ văn bản AI
        sent = llm.generate_calls[-1][1][-1].text
        for s in body["sources"]:
            assert s["title"] in sent


def test_chat_out_of_scope_has_no_sources(settings, store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    llm = FakeLLM('{"answer": "Nội dung này không có trong kho tri thức của VeggieLife.", "recommendedRecipeIds": []}')
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        seed(c)
        r = c.post("/chat/answer", headers=HEADERS, json={"question": "Giá vàng hôm nay là bao nhiêu"}).json()
        assert r["ragUsed"] is False and r["sources"] == []
        assert "Không có đoạn tri thức nào" in llm.generate_calls[-1][1][-1].text


def test_chat_unsafe_question_blocked_before_any_ai_call(client, llm):
    r = client.post("/chat/answer", headers=HEADERS, json={"question": "tôi nên bỏ thuốc tiểu đường"})
    assert r.status_code == 400 and r.json()["errorCode"] == "UNSAFE_CONTENT"
    assert llm.generate_calls == [] and llm.embed_calls == []


def test_chat_plain_text_reply_is_tolerated(settings, store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    with TestClient(create_app(settings, llm=FakeLLM("Trả lời không phải JSON."), store=store)) as c:
        r = c.post("/chat/answer", headers=HEADERS, json={"question": "đạm là gì"}).json()
        assert r["answer"].endswith("Trả lời không phải JSON.") and r["recommendedRecipeIds"] == []
        assert r["answer"].startswith("Nội dung này không có trong kho tri thức của VeggieLife")  # kho trống: luôn nói rõ


def test_chat_validation(client):
    assert client.post("/chat/answer", headers=HEADERS, json={"question": ""}).status_code == 400
    assert client.post("/chat/answer", headers=HEADERS, json={"question": "x" * 1001}).status_code == 400
    assert client.post("/chat/answer", headers=HEADERS, json={"question": "a", "history": [{"role": "bot", "text": "x"}]}).status_code == 400


def test_reindex_nutrition_endpoint(client, settings):
    from pathlib import Path
    d = Path(settings.knowledge_dir)
    d.mkdir(parents=True)
    (d / "01-dam.md").write_text('---\nid: 1\ntitle: "Đạm"\nsources:\n  - name: WHO\nupdated: "2026-10-10"\n---\nĐạm thực vật từ đậu.', encoding="utf-8")
    (d / "02-hong.md").write_text("không có front-matter", encoding="utf-8")
    r = client.post("/rag/reindex/nutrition", headers=HEADERS).json()
    assert r["files"] == 1 and r["chunks"] == 1 and r["failed"][0]["file"] == "02-hong.md"
