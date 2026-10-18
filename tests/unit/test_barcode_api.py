import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import HEADERS, KEY
from tests.fakes import FakeLLM

SAMPLE_REQ = {
    "barcode": "8934563138165",
    "product": {
        "productName": "Mì chay rau nấm",
        "brand": "VeggieFood",
        "ingredientsText": "Vắt mì (bột mì, dầu cọ, chất điều chỉnh độ acid (E330), phẩm màu tự nhiên (E100i)), gói gia vị (muối, đường, chất điều vị (E621), bột nấm rơm).",
        "additivesTags": ["en:e330", "en:e100i", "en:e621"],
        "allergensTags": ["en:gluten"],
        "tracesTags": ["en:soybeans"],
        "labelsTags": ["en:vegetarian"],
        "ingredientsAnalysisTags": ["en:vegan"],
    },
    "knownAdditives": [
        {"code": "E330", "name": "Acid citric", "veganStatus": "VEGAN"},
        {"code": "E621", "name": "Bột ngọt (Mononatri glutamat)", "veganStatus": "VEGAN"},
    ],
}

OK_AI_REPLY = {
    "veganVerdict": "VEGAN",
    "verdictText": "Sản phẩm thuần chay, các chất phụ gia và nguyên liệu đều có nguồn gốc thực vật.",
    "detectedAdditives": [
        {"code": "E330", "name": "Acid citric", "veganStatus": "VEGAN", "reason": "Có sẵn trong từ điển", "source": "DICTIONARY"},
        {"code": "E100i", "name": "Curcumin", "veganStatus": "VEGAN", "reason": "Chiết xuất nghệ tự nhiên", "source": "MODEL"},
        {"code": "E621", "name": "Bột ngọt", "veganStatus": "VEGAN", "reason": "Lên men từ mật mía/tinh bột", "source": "DICTIONARY"},
    ],
    "allergens": [
        {"name": "Gluten", "presenceType": "CONTAINS"},
        {"name": "Đậu nành", "presenceType": "MAY_CONTAIN"},
    ],
    "confidence": 0.95,
}


def make(settings, store, reply=None, llm=None):
    llm = llm or FakeLLM(json.dumps(OK_AI_REPLY, ensure_ascii=False) if reply is None else reply)
    return TestClient(create_app(settings, llm=llm, store=store)), llm


def test_barcode_classify_requires_internal_key(client):
    res = client.post("/barcode/classify", json=SAMPLE_REQ)
    assert res.status_code == 401
    assert res.json()["errorCode"] == "UNAUTHORIZED"


def test_barcode_classify_invalid_key(client):
    res = client.post("/barcode/classify", json=SAMPLE_REQ, headers={"X-Internal-Api-Key": "wrong-key"})
    assert res.status_code == 401
    assert res.json()["errorCode"] == "UNAUTHORIZED"


def test_barcode_classify_validation_error_barcode_length(settings, store):
    c, _ = make(settings, store)
    with c:
        res = c.post("/barcode/classify", json={**SAMPLE_REQ, "barcode": "123"}, headers=HEADERS)
    assert res.status_code == 400
    assert res.json()["errorCode"] == "VALIDATION_ERROR"


def test_barcode_classify_success(settings, store):
    c, llm = make(settings, store)
    with c:
        res = c.post("/barcode/classify", json=SAMPLE_REQ, headers=HEADERS)

    assert res.status_code == 200
    data = res.json()
    assert data["veganVerdict"] == "VEGAN"
    assert "thuần chay" in data["verdictText"].lower()
    assert len(data["detectedAdditives"]) == 3
    assert len(data["allergens"]) == 2
    assert data["confidence"] == 0.95

    # Check usage & audit
    assert "usage" in data
    assert data["usage"]["promptTokens"] > 0
    assert data["usage"]["totalTokens"] >= data["usage"]["promptTokens"]
    assert "audit" in data
    assert "8934563138165" in data["audit"]["prompt"]
    assert "rawResult" in data["audit"]

    # Verify LLM call
    assert len(llm.generate_calls) == 1
    sys_prompt, msgs = llm.generate_calls[0]
    assert "VeggieLife" in sys_prompt
    assert "knownAdditives" in sys_prompt
    assert "8934563138165" in msgs[0].text


def test_barcode_classify_prioritizes_dictionary_source(settings, store):
    # Model mistakenly returns MODEL and DOUBTFUL for E330, but knownAdditives has VEGAN
    reply = {
        "veganVerdict": "DOUBTFUL",
        "verdictText": "Nghi vấn",
        "detectedAdditives": [
            {"code": "E330", "name": "Acid citric", "veganStatus": "DOUBTFUL", "reason": "Chưa rõ", "source": "MODEL"}
        ],
        "allergens": [],
        "confidence": 0.8,
    }
    c, _ = make(settings, store, json.dumps(reply))
    with c:
        res = c.post("/barcode/classify", json=SAMPLE_REQ, headers=HEADERS)

    assert res.status_code == 200
    data = res.json()
    # E330 must be overridden to VEGAN and DICTIONARY because it exists in knownAdditives
    e330 = next(a for a in data["detectedAdditives"] if a["code"] == "E330")
    assert e330["veganStatus"] == "VEGAN"
    assert e330["source"] == "DICTIONARY"


def test_barcode_classify_handles_json_parse_error(settings, store):
    # LLM returns invalid non-JSON string
    c, _ = make(settings, store, "Không phải JSON hợp lệ")
    with c:
        res = c.post("/barcode/classify", json=SAMPLE_REQ, headers=HEADERS)

    assert res.status_code == 502
    assert res.json()["errorCode"] == "GEMINI_ERROR"
