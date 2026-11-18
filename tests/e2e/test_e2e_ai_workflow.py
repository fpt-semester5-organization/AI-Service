import pytest
from tests.conftest import HEADERS


def test_e2e_full_ai_assistant_workflow(client, llm):
    """
    Kiểm thử luồng nghiệp vụ hoàn chỉnh (E2E Workflow):
    1. Kiểm tra sức khỏe dịch vụ AI (/health).
    2. Nạp tài liệu tri thức dinh dưỡng vào kho RAG (/rag/upsert).
    3. Kiểm duyệt nội dung bài đăng món ăn (/moderation/check).
    4. Trò chuyện chuyên sâu với chuyên gia dinh dưỡng RAG (/chat/answer).
    """
    # 1. Health check
    health_res = client.get("/health", headers=HEADERS)
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "ok"

    # 2. Nạp tri thức RAG (chuẩn sourceType NUTRITION_DOC)
    up_res = client.post("/rag/upsert", headers=HEADERS, json={
        "sourceType": "NUTRITION_DOC",
        "sourceId": "doc-512",
        "title": "Tài liệu dinh dưỡng đạm thực vật",
        "text": "Đậu hũ và nấm rơm là nguồn protein thực vật tuyệt vời, giàu axit amin và khoáng chất cho người ăn chay tập gym.",
        "updatedAt": "2026-10-10T09:00:00Z"
    })
    assert up_res.status_code == 200
    assert up_res.json()["chunks"] >= 1

    # 3. Moderation check
    llm.reply = '{"violationScore": 0.05, "violationType": "NONE", "reasons": []}'
    mod_res = client.post("/moderation/check", headers=HEADERS, json={
        "title": "Cách nấu canh nấm chay thanh đạm",
        "description": "Món ăn bổ dưỡng với nấm rơm và đậu hũ non",
        "ingredient_names": ["Nấm rơm tươi", "Đậu hũ non", "Nước tương"]
    })
    assert mod_res.status_code == 200
    mod_data = mod_res.json()
    assert mod_data["violationType"] == "NONE"
    assert mod_data["violationScore"] < 0.85

    # 4. Nutrition Chat Consultation
    llm.reply = "Đậu hũ và nấm là nguồn protein thực vật tuyệt vời, cung cấp đầy đủ axit amin thiết yếu cho người ăn thuần chay."
    chat_res = client.post("/chat/answer", headers=HEADERS, json={
        "question": "Đậu hũ kho nấm có protein cho người tập thể hình không?",
        "history": [
            {"role": "user", "text": "Tôi vừa nấu đậu hũ kho nấm"},
            {"role": "model", "text": "Rất ngon miệng và bổ dưỡng!"}
        ]
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "protein" in chat_data["answer"].lower()
