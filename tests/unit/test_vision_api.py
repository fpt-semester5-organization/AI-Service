import json

import pytest
from fastapi.testclient import TestClient
from google.genai import types
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app
from app.services.vision.ingredient_detector import RESPONSE_SCHEMA, SYSTEM_INSTRUCTION, USER_TEXT, is_non_food, normalize
from tests.conftest import HEADERS, KEY
from tests.fakes import FakeLLM
from tests.image_helpers import jpeg, png, webp

GOOD = {"items": [
    {"name": "Cà chua", "confidence": 0.93, "freshnessStatus": "FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False, "quantity": "3 quả (~300g)", "amount": 3, "unit": "quả"},
    {"name": "Trứng gà", "confidence": 0.8, "freshnessStatus": "FRESH", "foodGroup": "Trứng", "isAnimalProduct": True, "isRedMeat": False, "quantity": "4 quả", "amount": 4, "unit": "quả"},
    {"name": "Thịt bò", "confidence": 0.88, "freshnessStatus": "FRESH", "foodGroup": "Thịt đỏ", "isAnimalProduct": True, "isRedMeat": True, "quantity": "300g", "amount": 300, "unit": "g"},
    {"name": "Rau muống", "confidence": 0.7, "freshnessStatus": "NON-FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False, "quantity": "1 bó (~400g)", "amount": 1, "unit": "bó"}]}
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
        "quantity": "4 quả",
        "amount": 4.0,
        "unit": "quả",
        "box2d": None,
    }
    assert body["items"][2] == {
        "name": "Thịt bò",
        "confidence": 0.88,
        "freshnessStatus": "FRESH",
        "foodGroup": "Thịt đỏ",
        "isAnimalProduct": True,
        "isRedMeat": True,
        "quantity": "300g",
        "amount": 300.0,
        "unit": "g",
        "box2d": None,
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


def test_detect_with_bounding_box(settings, store):
    """Kiểm tra Gemini trả về toạ độ box2d và endpoint API bảo toàn toạ độ trong JSON response."""
    data_with_box = {
        "items": [
            {
                "name": "Nấm rơm tươi",
                "confidence": 0.95,
                "freshnessStatus": "FRESH",
                "foodGroup": "Nấm",
                "isAnimalProduct": False,
                "isRedMeat": False,
                "box2d": [420, 115, 680, 395],
            },
            {
                "name": "Rau muống héo",
                "confidence": 0.82,
                "freshnessStatus": "NON-FRESH",
                "foodGroup": "Rau củ",
                "isAnimalProduct": False,
                "isRedMeat": False,
                "box_2d": [180, 520, 490, 840],  # Hỗ trợ alias box_2d
            },
        ]
    }
    llm = FakeLLM(json.dumps(data_with_box, ensure_ascii=False))
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        r = post(c, jpeg(1280, 720))
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    assert items[0]["name"] == "Nấm rơm tươi"
    assert items[0]["box2d"] == [420, 115, 680, 395]
    assert items[1]["name"] == "Rau muống héo"
    assert items[1]["box2d"] == [180, 520, 490, 840]


def test_normalize_box2d_coordinate_validation():
    """Kiểm tra hàm normalize() xử lý các trường hợp toạ độ hợp lệ và bất thường."""
    # Toạ độ hợp lệ chuẩn hóa 0..1000
    valid_data = {
        "items": [
            {"name": "Cà chua", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [100, 200, 300, 400]},
            {"name": "Dưa chuột", "confidence": 0.85, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box_2d": [0, 0, 1000, 1000]},
            {"name": "Cà rốt", "confidence": 0.8, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [12.4, 34.6, 120.1, 250.8]},
        ]
    }
    res = normalize(valid_data)
    assert res[0].box2d == [100, 200, 300, 400]
    assert res[1].box2d == [0, 0, 1000, 1000]
    assert res[2].box2d == [12, 35, 120, 251]

    # Toạ độ bất thường -> được làm sạch về None
    invalid_data = {
        "items": [
            {"name": "A", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [300, 200, 100, 400]},  # ymin > ymax
            {"name": "B", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [100, 400, 300, 200]},  # xmin > xmax
            {"name": "C", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [-10, 50, 300, 400]},   # ymin < 0
            {"name": "D", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [100, 50, 1050, 400]},  # ymax > 1000
            {"name": "E", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": [100, 200, 300]},        # Thiếu toạ độ
            {"name": "F", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False, "box2d": "invalid"},               # Sai kiểu dữ liệu
        ]
    }
    res_inv = normalize(invalid_data)
    for item in res_inv:
        assert item.box2d is None



def test_is_non_food_filter():
    """Kiểm tra hàm is_non_food lọc chính xác đồ điện tử, đồ gia dụng, trong khi giữ lại thực phẩm."""
    assert is_non_food("Điện thoại") is True
    assert is_non_food("Điện thoại iPhone 15 Pro") is True
    assert is_non_food("Laptop") is True
    assert is_non_food("Chuột máy tính") is True
    assert is_non_food("Bàn phím cơ") is True
    assert is_non_food("Tai nghe bluetooth") is True
    assert is_non_food("Củ sạc nhanh") is True
    assert is_non_food("Cáp sạc type-C") is True
    assert is_non_food("Sạc dự phòng") is True
    assert is_non_food("Tivi Samsung") is True
    assert is_non_food("Đồng hồ thông minh") is True
    assert is_non_food("Thiết bị điện tử") is True

    assert is_non_food("Vật thể lạ", food_group="Thiết bị điện tử") is True
    assert is_non_food("Đồ gia dụng", food_group="Gia dụng") is True

    assert is_non_food("Cà chua") is False
    assert is_non_food("Cà rốt tươi") is False
    assert is_non_food("Củ cải trắng") is False
    assert is_non_food("Củ gừng") is False
    assert is_non_food("Nấm rơm") is False
    assert is_non_food("Rau muống") is False
    assert is_non_food("Thịt bò") is False
    assert is_non_food("Đậu phụ non") is False
    assert is_non_food("Trứng gà") is False


def test_normalize_drops_electronic_and_non_food_items():
    """Kiểm tra normalize() tự động loại bỏ các đồ điện tử và chỉ giữ lại nguyên liệu thực phẩm."""
    raw = {
        "items": [
            {"name": "Điện thoại iPhone", "confidence": 0.95, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False},
            {"name": "Chuột máy tính", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False},
            {"name": "Cà chua", "confidence": 0.88, "freshnessStatus": "FRESH", "foodGroup": "Rau củ", "isAnimalProduct": False, "isRedMeat": False},
            {"name": "Laptop Dell", "confidence": 0.92, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False},
            {"name": "Củ sạc", "confidence": 0.85, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False},
            {"name": "Nấm đùi gà", "confidence": 0.91, "freshnessStatus": "FRESH", "foodGroup": "Nấm", "isAnimalProduct": False, "isRedMeat": False},
        ]
    }
    cleaned = normalize(raw)
    assert len(cleaned) == 2
    names = [c.name for c in cleaned]
    assert "Cà chua" in names
    assert "Nấm đùi gà" in names
    assert "Điện thoại iPhone" not in names
    assert "Chuột máy tính" not in names
    assert "Laptop Dell" not in names
    assert "Củ sạc" not in names


def test_prompt_strictness_for_food_ingredients_and_electronics():
    """Kiểm tra Prompt có các quy tắc nghiêm ngặt về đồ điện tử và chỉ nhận diện nguyên liệu thực phẩm."""
    assert "NGUYÊN LIỆU THỰC PHẨM" in SYSTEM_INSTRUCTION
    assert "ĐỒ ĐIỆN TỬ" in SYSTEM_INSTRUCTION
    assert "điện thoại" in SYSTEM_INSTRUCTION.lower()
    assert "chuột máy tính" in SYSTEM_INSTRUCTION.lower()
    assert "items: []" in SYSTEM_INSTRUCTION
    assert "đồ điện tử" in USER_TEXT.lower()


def test_prompt_has_bell_pepper_vs_tomato_morphology_rule():
    from app.services.vision.ingredient_detector import SYSTEM_INSTRUCTION, USER_TEXT
    text = SYSTEM_INSTRUCTION.lower()
    assert "ớt chuông" in text and "cà chua" in text
    assert "múi dọc" in text and "hình cầu" in text.replace("hình cầu hoặc", "hình cầu")
    assert "không chỉ dựa vào màu" in text
    assert "ớt chuông" in USER_TEXT.lower()


def test_detect_falls_back_to_bare_json_array_reply():
    """Gemini trả mảng JSON trần thay vì object → vẫn đọc được (trước đây thiếu `import json` nên nhánh này hỏng âm thầm)."""
    import asyncio

    from app.services.vision.ingredient_detector import IngredientDetector
    from tests.fakes import FakeLLM
    from tests.image_helpers import png

    reply = '[{"name": "Ớt chuông", "confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": false, "isRedMeat": false, "quantity": "2 quả"}]'
    det = IngredientDetector(Settings(gemini_api_key="k", qdrant_url="http://x"), FakeLLM(reply))
    res = asyncio.run(det.detect(png(800, 600)))
    assert [i.name for i in res.items] == ["Ớt chuông"]


def test_schema_forces_visual_cues_before_name():
    """thinking_budget=0 → mô hình phải ghi quan sát hình thái (visualCues) trước khi chốt tên."""
    item = RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert item["propertyOrdering"][:2] == ["visualCues", "name"]
    assert "visualCues" in item["required"]
    assert set(item["propertyOrdering"]) == set(item["properties"])


def test_prompt_covers_colored_bell_peppers_and_red_cabbage():
    text = SYSTEM_INSTRUCTION.lower()
    assert "vàng" in text and "cam" in text and "bắp cải tím" in text
    assert "mỗi box2d chỉ chứa một loại" in text


def test_normalize_keeps_same_name_in_different_boxes():
    base = {"confidence": 0.9, "freshnessStatus": "FRESH", "isAnimalProduct": False, "isRedMeat": False}
    out = normalize({"items": [
        {**base, "name": "Cà chua", "box2d": [100, 100, 200, 200]},
        {**base, "name": "Cà chua", "box2d": [500, 500, 600, 600]},
        {**base, "name": "Cà chua", "box2d": [100, 100, 200, 200], "confidence": 0.5},  # trùng hộp → gộp
        {**base, "name": "Cà chua"},  # dạng Unicode NFD vẫn được chuẩn hóa về NFC
    ]})
    assert [(i.name, i.box2d) for i in out] == [
        ("Cà chua", [100, 100, 200, 200]), ("Cà chua", [500, 500, 600, 600]), ("Cà chua", None)]


def test_detect_uses_dedicated_vision_timeout():
    """Ảnh nhiều món mất 20–50s → nhận diện dùng VISION_TIMEOUT_SECONDS, không dùng timeout 25s chung của chat."""
    import asyncio

    from app.core.gemini_client import Generation
    from app.services.vision.ingredient_detector import IngredientDetector

    seen = {}

    class CaptureLLM:
        async def generate(self, system, messages, **kwargs):
            seen.update(kwargs)
            return Generation(text='{"items": []}', model="m", prompt_tokens=1, completion_tokens=1,
                              latency_ms=1, temperature=0.2)

    s = Settings(vision_timeout_seconds=55, gemini_timeout_seconds=25)
    asyncio.run(IngredientDetector(s, CaptureLLM()).detect(jpeg(800, 600)))
    assert seen["timeout"] == 55


def test_normalize_merges_overlapping_boxes_of_same_name():
    """Lỗi thật 13/11: cùng cuộn phô mai dê được khoanh 3 lần với khung lệch nhau → chỉ giữ 1."""
    base = {"freshnessStatus": "FRESH", "isAnimalProduct": True, "isRedMeat": False}
    out = normalize({"items": [
        {**base, "name": "Phô mai", "confidence": 0.9, "box2d": [800, 845, 910, 930]},
        {**base, "name": "Phô mai", "confidence": 0.92, "box2d": [805, 850, 905, 935]},   # IoU cao → trùng, tin cậy hơn
        {**base, "name": "Phô mai", "confidence": 0.9, "box2d": [820, 860, 900, 920]},    # nằm trọn trong khung trên → trùng
        {**base, "name": "Phô mai", "confidence": 0.9, "box2d": [750, 660, 860, 760]},    # miếng phô mai khác chỗ → giữ
        {**base, "name": "Xúc xích", "confidence": 0.9, "box2d": [805, 850, 905, 935]},   # khác tên, cùng chỗ → giữ
    ]})
    assert [(i.name, i.confidence, i.box2d) for i in out] == [
        ("Phô mai", 0.92, [805, 850, 905, 935]),
        ("Phô mai", 0.9, [750, 660, 860, 760]),
        ("Xúc xích", 0.9, [805, 850, 905, 935]),
    ]


def test_box_key_option_renames_schema_field_but_contract_stays_box2d(settings, store):
    """Bước 4: vision_box_key="box_2d" đổi tên trường trong schema/prompt; đầu ra Backend vẫn là box2d."""
    import asyncio

    from app.core.gemini_client import Generation
    from app.services.vision.ingredient_detector import IngredientDetector, build_response_schema

    seen = {}

    class CaptureLLM:
        async def generate(self, system, messages, **kwargs):
            seen.update(kwargs, system=system)
            return Generation(text='{"items": [{"name": "Cà chua", "confidence": 0.9, "freshnessStatus": "FRESH", '
                                   '"isAnimalProduct": false, "isRedMeat": false, "quantity": "1 quả", "box_2d": [1, 2, 3, 4]}]}',
                              model="m", prompt_tokens=1, completion_tokens=1, latency_ms=1, temperature=0.2)

    s = Settings(vision_box_key="box_2d", vision_thinking_budget=1024)
    res = asyncio.run(IngredientDetector(s, CaptureLLM()).detect(jpeg(800, 600)))
    item = seen["json_schema"]["properties"]["items"]["items"]
    assert "box_2d" in item["properties"] and "box2d" not in item["properties"]
    assert item["propertyOrdering"][-1] == "box_2d"
    assert "box_2d" in seen["system"] and "box2d" not in seen["system"]
    assert seen["thinking_budget"] == 1024
    assert res.items[0].box2d == [1, 2, 3, 4]
    assert build_response_schema() == RESPONSE_SCHEMA
