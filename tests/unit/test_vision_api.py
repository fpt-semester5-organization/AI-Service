import json

import pytest
from fastapi.testclient import TestClient
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app
from app.services.vision.ingredient_detector import RESPONSE_SCHEMA, normalize
from tests.conftest import HEADERS, KEY
from tests.fakes import FakeLLM
from tests.image_helpers import jpeg, png, webp

GOOD = {"items": [
    {"name": "Cà chua", "confidence": 0.93, "freshnessStatus": "FRESH", "isAnimalProduct": False},
    {"name": "Trứng gà", "confidence": 0.8, "freshnessStatus": "FRESH", "isAnimalProduct": True},
    {"name": "Rau muống", "confidence": 0.7, "freshnessStatus": "NON-FRESH", "isAnimalProduct": False}]}
URL = "/vision/detect-ingredients"


def post(client, data: bytes, name="a.jpg", ctype="image/jpeg", headers=HEADERS, field="file"):
    return client.post(URL, files={field: (name, data, ctype)}, headers=headers)


def test_requires_key(client):
    assert post(client, jpeg(640, 480), headers={}).status_code == 401


def test_detect_ok(settings, store):
    llm = FakeLLM(json.dumps(GOOD, ensure_ascii=False))
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        r = post(c, jpeg(1280, 720))
    assert r.status_code == 200
    body = r.json()
    assert [i["name"] for i in body["items"]] == ["Cà chua", "Trứng gà", "Rau muống"]
    assert body["items"][1] == {"name": "Trứng gà", "confidence": 0.8, "freshnessStatus": "FRESH", "isAnimalProduct": True}
    assert body["usage"]["model"] == "fake-model"
    # ảnh thật sự được chuyển cho Gemini, MIME lấy từ chữ ký tệp; nhiệt độ thấp, có JSON schema
    system, msgs = llm.generate_calls[0]
    assert len(msgs[0].images) == 1 and msgs[0].images[0].mime_type == "image/jpeg"
    assert msgs[0].images[0].data == jpeg(1280, 720)
    kw = llm.generate_kwargs[0]
    assert kw["temperature"] <= 0.4 and kw["json_schema"] is RESPONSE_SCHEMA
    assert "DỮ LIỆU" in system


def test_detect_with_image_field_name(settings, store):
    llm = FakeLLM(json.dumps(GOOD, ensure_ascii=False))
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        r = post(c, jpeg(1280, 720), field="image")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 3


@pytest.mark.parametrize("data,mime", [(png(640, 480), "image/png"), (webp(800, 600), "image/webp")])
def test_png_and_webp_accepted(client, llm, data, mime):
    llm.reply = json.dumps({"items": []})
    r = post(client, data, "x", "application/octet-stream")   # Content-Type client khai không quan trọng
    assert r.status_code == 200 and r.json()["items"] == []
    assert llm.generate_calls[0][1][0].images[0].mime_type == mime


def test_portrait_photo_ok(client, llm):
    llm.reply = json.dumps({"items": []})
    assert post(client, png(480, 640)).status_code == 200


@pytest.mark.parametrize("data", [png(639, 480), png(640, 479), jpeg(320, 240), png(100, 100)])
def test_too_small_is_image_invalid(client, llm, data):
    r = post(client, data)
    assert r.status_code == 400 and r.json()["errorCode"] == "IMAGE_INVALID"
    assert llm.generate_calls == []        # không gọi Gemini khi ảnh không đạt


@pytest.mark.parametrize("data", [b"GIF89a" + b"\x00" * 100, b"%PDF-1.7\n" + b"x" * 100, b"not an image", b""])
def test_wrong_format_is_image_invalid(client, llm, data):
    r = post(client, data, "a.jpg", "image/jpeg")        # đuôi và Content-Type nói dối
    assert r.status_code == 400 and r.json()["errorCode"] == "IMAGE_INVALID"
    assert llm.generate_calls == []


def test_missing_file_is_400(client):
    r = client.post(URL, headers=HEADERS)
    assert r.status_code == 400 and r.json()["errorCode"] == "VALIDATION_ERROR"


def test_size_limit_413(llm, store):
    s = Settings(internal_api_key=KEY, google_cloud_project="t", max_upload_mb=1, _env_file=None)
    big = png(640, 480) + b"\x00" * (1024 * 1024 + 1)
    with TestClient(create_app(s, llm=llm, store=store)) as c:
        r = post(c, big, "a.png", "image/png")
        assert r.status_code == 413 and r.json()["errorCode"] == "FILE_TOO_LARGE"
        ok = png(640, 480)       # đúng giới hạn thì qua kiểm kích thước
        assert len(ok) < 1024 * 1024
        llm.reply = json.dumps({"items": []})
        assert post(c, ok, "a.png", "image/png").status_code == 200
    assert len(llm.generate_calls) == 1


def test_size_exactly_at_limit_ok(llm, store):
    s = Settings(internal_api_key=KEY, google_cloud_project="t", max_upload_mb=1, _env_file=None)
    llm.reply = json.dumps({"items": []})
    exact = png(640, 480)
    exact += b"\x00" * (1024 * 1024 - len(exact))
    with TestClient(create_app(s, llm=llm, store=store)) as c:
        assert len(exact) == 1024 * 1024 and post(c, exact, "a.png", "image/png").status_code == 200


def test_bad_json_is_gemini_error(settings, store):
    with TestClient(create_app(settings, llm=FakeLLM("xin chào"), store=store)) as c:
        r = post(c, png(640, 480))
    assert r.status_code == 502 and r.json()["errorCode"] == "GEMINI_ERROR"


def test_gemini_failure_maps_to_502_and_timeout_to_504(settings, store):
    from app.core.errors import ai_timeout, gemini_error
    for exc, code, status in ((gemini_error(), "GEMINI_ERROR", 502), (ai_timeout(), "AI_TIMEOUT", 504)):
        with TestClient(create_app(settings, llm=FakeLLM(fail=exc), store=store)) as c:
            r = post(c, png(640, 480))
        assert r.status_code == status and r.json()["errorCode"] == code


def test_normalize_drops_bad_items_clamps_and_dedupes():
    out = normalize({"items": [
        {"name": " Cà chua ", "confidence": 1.7, "freshnessStatus": "FRESH", "isAnimalProduct": False},
        {"name": "cà chua", "confidence": 0.2, "freshnessStatus": "NON-FRESH", "isAnimalProduct": False},
        {"name": "Hành", "confidence": -1, "freshnessStatus": "FRESH", "isAnimalProduct": False},
        {"name": "Ớt", "confidence": "0.9", "freshnessStatus": "FRESH", "isAnimalProduct": False},      # sai kiểu
        {"name": "Tỏi", "confidence": 0.9, "freshnessStatus": "ROTTEN", "isAnimalProduct": False},      # sai enum
        {"name": "Gừng", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": "no"},       # sai kiểu
        {"name": "", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False}, "oops"]})
    assert [(i.name, i.confidence) for i in out] == [("Cà chua", 1.0), ("Hành", 0.0)]


def test_schema_is_valid_for_google_sdk():
    # types.Schema kiểm từng khóa (khóa sai chính tả sẽ báo lỗi); GenerateContentConfig nhận dict mà không kiểm.
    types.Schema.model_validate(RESPONSE_SCHEMA)
    with pytest.raises(ValidationError):
        types.Schema.model_validate({**RESPONSE_SCHEMA, "requird": ["x"]})
