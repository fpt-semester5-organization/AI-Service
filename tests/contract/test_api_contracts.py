import pytest
from tests.conftest import HEADERS


def test_contract_chat_response_schema(client, llm):
    """
    Kiểm thử tương thích Schema / Contract:
    Đảm bảo endpoint /chat/answer trả đúng cấu trúc JSON camelCase mà Backend DTO yêu cầu.
    """
    llm.reply = "Câu trả lời dinh dưỡng chuẩn."
    res = client.post("/chat/answer", headers=HEADERS, json={"question": "Kiểm tra schema"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "usage" in data
    assert "audit" in data


def test_contract_moderation_response_schema(client, llm):
    """
    Kiểm thử tương thích Schema / Contract:
    Đảm bảo endpoint /moderation/check trả đúng schema vi phạm camelCase.
    """
    llm.reply = '{"violationScore": 0.1, "violationType": "NONE", "reasons": []}'
    res = client.post("/moderation/check", headers=HEADERS, json={
        "title": "Bún chả chay",
        "description": "Làm từ nấm và sườn non đậu nành"
    })
    assert res.status_code == 200
    data = res.json()
    assert "violationScore" in data
    assert "violationType" in data
    assert "reasons" in data
    assert isinstance(data["reasons"], list)


def test_contract_health_schema(client):
    """
    Kiểm thử tương thích Contract:
    Endpoint /health phải trả về status: ok.
    """
    res = client.get("/health", headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
