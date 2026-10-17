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
    {"name": "Cà chua", "confidence": 0.93, "freshnessStatus": "FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False},
    {"name": "Trứng gà", "confidence": 0.8, "freshnessStatus": "FRESH", "foodGroup": "Trứng", "isAnimalProduct": True, "isRedMeat": False},
    {"name": "Thịt bò", "confidence": 0.88, "freshnessStatus": "FRESH", "foodGroup": "Thịt đỏ", "isAnimalProduct": True, "isRedMeat": True},
    {"name": "Rau muống", "confidence": 0.7, "freshnessStatus": "NON-FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False}]}
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
    assert [i["name"] for i in body["items"]] == ["Cà chua", "Trứng gà", "Thịt bò", "Rau muống"]
    assert body["items"][1] == {
        "name": "Trứng gà",
        "confidence": 0.8,
        "freshnessStatus": "FRESH",
        "foodGroup": "Trứng",
        "isAnimalProduct": True,
        "isRedMeat": False,
    }
    assert body["items"][2] == {
        "name": "Thịt bò",
        "confidence": 0.88,
        "freshnessStatus": "FRESH",
        "foodGroup": "Thịt đỏ",
        "isAnimalProduct": True,
        "isRedMeat": True,
    }
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
    assert len(body["items"]) == len(GOOD["items"])


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
        {"name": " Cà chua ", "confidence": 1.7, "freshnessStatus": "FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False},
        {"name": "cà chua", "confidence": 0.2, "freshnessStatus": "NON-FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False},
        {"name": "Hành", "confidence": -1, "freshnessStatus": "FRESH", "foodGroup": "Gia vị", "isAnimalProduct": False, "isRedMeat": False},
        {"name": "Thịt bò", "confidence": 0.95, "freshnessStatus": "FRESH", "foodGroup": "Thịt đỏ", "isAnimalProduct": True, "isRedMeat": True},
        {"name": "Ớt", "confidence": "0.9", "freshnessStatus": "FRESH", "isAnimalProduct": False},      # sai kiểu
        {"name": "Tỏi", "confidence": 0.9, "freshnessStatus": "ROTTEN", "isAnimalProduct": False},      # sai enum
        {"name": "Gừng", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": "no"},       # sai kiểu
        {"name": "", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False}, "oops"]})
    assert [(i.name, i.confidence, i.food_group, i.is_red_meat) for i in out] == [
        ("Cà chua", 1.0, "Rau củ", False),
        ("Hành", 0.0, "Gia vị", False),
        ("Thịt bò", 0.95, "Thịt đỏ", True),
    ]


def test_schema_is_valid_for_google_sdk():
    # types.Schema kiểm từng khóa (khóa sai chính tả sẽ báo lỗi); GenerateContentConfig nhận dict mà không kiểm.
    types.Schema.model_validate(RESPONSE_SCHEMA)
    with pytest.raises(ValidationError):
        types.Schema.model_validate({**RESPONSE_SCHEMA, "requird": ["x"]})


def test_five_sample_scenarios(settings, store):
    """Kiểm thử 5 kịch bản ảnh mẫu theo tiêu chuẩn nghiệm thu:
    1. Ảnh 1: Tủ lạnh rau củ tươi đạt chuẩn (JPEG ≥ 640×480) -> Nhận diện nguyên liệu tươi, nhóm thực phẩm
    2. Ảnh 2: Thực phẩm rau héo (PNG ≥ 640×480) -> Nhận diện NON-FRESH để cảnh báo/ưu tiên dùng sớm
    3. Ảnh 3: Tủ lạnh có thịt đỏ (WEBP ≥ 640×480) -> Nhận diện và đánh dấu isRedMeat=True, foodGroup='Thịt đỏ'
    4. Ảnh 4: Ảnh quá nhỏ (< 640×480) -> Bị từ chối với 400 IMAGE_INVALID
    5. Ảnh 5: Tệp giả mạo (không phải ảnh thật, sai magic bytes) -> Bị từ chối với 400 IMAGE_INVALID
    """
    # Sample 1: JPEG tươi
    llm1 = FakeLLM(json.dumps({
        "items": [{"name": "Cà chua", "confidence": 0.95, "freshnessStatus": "FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False}]
    }))
    with TestClient(create_app(settings, llm=llm1, store=store)) as c:
        r1 = post(c, jpeg(800, 600))
        assert r1.status_code == 200
        data1 = r1.json()["items"]
        assert len(data1) == 1
        assert data1[0]["freshnessStatus"] == "FRESH"
        assert data1[0]["foodGroup"] == "Rau củ"
        assert data1[0]["isRedMeat"] is False

    # Sample 2: PNG rau héo (NON-FRESH)
    llm2 = FakeLLM(json.dumps({
        "items": [{"name": "Rau muống", "confidence": 0.85, "freshnessStatus": "NON-FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False}]
    }))
    with TestClient(create_app(settings, llm=llm2, store=store)) as c:
        r2 = post(c, png(640, 480))
        assert r2.status_code == 200
        data2 = r2.json()["items"]
        assert len(data2) == 1
        assert data2[0]["freshnessStatus"] == "NON-FRESH"

    # Sample 3: WEBP có thịt đỏ
    llm3 = FakeLLM(json.dumps({
        "items": [{"name": "Thịt bò", "confidence": 0.92, "freshnessStatus": "FRESH", "foodGroup": "Thịt đỏ", "isAnimalProduct": True, "isRedMeat": True}]
    }))
    with TestClient(create_app(settings, llm=llm3, store=store)) as c:
        r3 = post(c, webp(1024, 768))
        assert r3.status_code == 200
        data3 = r3.json()["items"]
        assert len(data3) == 1
        assert data3[0]["isRedMeat"] is True
        assert data3[0]["isAnimalProduct"] is True
        assert data3[0]["foodGroup"] == "Thịt đỏ"

    # Sample 4: Ảnh quá nhỏ (320x240 px < 640x480) -> Từ chối 400 IMAGE_INVALID
    with TestClient(create_app(settings, llm=FakeLLM("{}"), store=store)) as c:
        r4 = post(c, jpeg(320, 240))
        assert r4.status_code == 400
        assert r4.json()["errorCode"] == "IMAGE_INVALID"

    # Sample 5: Tệp giả mạo (file PDF/text đổi đuôi .jpg) -> Từ chối 400 IMAGE_INVALID
    fake_data = b"%PDF-1.5 Fake PDF pretending to be jpeg image" + b"\x00" * 500
    with TestClient(create_app(settings, llm=FakeLLM("{}"), store=store)) as c:
        r5 = post(c, fake_data, name="fake.jpg", ctype="image/jpeg")
        assert r5.status_code == 400
        assert r5.json()["errorCode"] == "IMAGE_INVALID"
