"""Bộ kiểm thử mở rộng bắt lỗi khắt khe (stress & edge cases) cho /meal/generate (FN-MEAL-01)."""
import asyncio

from fastapi.testclient import TestClient

from app.core.errors import gemini_error
from app.main import create_app
from tests.conftest import HEADERS
from tests.fakes import FakeLLM
from tests.unit.test_meal_api import REQ, slots_json


def call_api(settings, store, reply, body=REQ, headers=HEADERS):
    llm = FakeLLM(reply)
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        return c.post("/meal/generate", json=body, headers=headers), llm


# --- 1. KIỂM TRA BẢO MẬT & XÁC THỰC (AUTHENTICATION TESTS) ---

def test_auth_wrong_key(settings, store):
    """Gửi sai key phải bị chặn 401."""
    r, _ = call_api(settings, store, slots_json(), headers={"X-Internal-Api-Key": "wrong-secret-key"})
    assert r.status_code == 401
    assert r.json()["errorCode"] == "UNAUTHORIZED"


def test_auth_key_with_whitespace(settings, store):
    """Key có khoảng trắng thừa phải bị chặn 401."""
    key = HEADERS["X-Internal-Api-Key"]
    r, _ = call_api(settings, store, slots_json(), headers={"X-Internal-Api-Key": f" {key} "})
    assert r.status_code == 401


def test_auth_empty_key(settings, store):
    """Key rỗng phải bị chặn 401."""
    r, _ = call_api(settings, store, slots_json(), headers={"X-Internal-Api-Key": ""})
    assert r.status_code == 401


# --- 2. KIỂM TRA BIÊN DỮ LIỆU ĐẦU VÀO (INPUT BOUNDARY TESTS) ---

def test_calories_boundary_values(settings, store):
    """Calo biên 1 và 10000 hợp lệ; calo 10001 bị lỗi 400."""
    # Biên 1
    r, _ = call_api(settings, store, slots_json(), {**REQ, "targetDailyCalories": 1})
    assert r.status_code == 200
    # Biên 10000
    r, _ = call_api(settings, store, slots_json(), {**REQ, "targetDailyCalories": 10000})
    assert r.status_code == 200
    # Vượt biên 10001
    r, _ = call_api(settings, store, slots_json(), {**REQ, "targetDailyCalories": 10001})
    assert r.status_code == 400
    assert r.json()["errorCode"] == "VALIDATION_ERROR"


def test_candidate_recipes_boundary_500(settings, store):
    """Tối đa 500 công thức; 501 công thức phải báo lỗi 400."""
    recipes_500 = [{"recipeId": i, "title": f"Món {i}", "caloriesKcal": 300} for i in range(1, 501)]
    r, _ = call_api(settings, store, slots_json(ids=[1]), {**REQ, "candidateRecipes": recipes_500})
    assert r.status_code == 200

    recipes_501 = [{"recipeId": i, "title": f"Món {i}", "caloriesKcal": 300} for i in range(1, 502)]
    r, _ = call_api(settings, store, slots_json(), {**REQ, "candidateRecipes": recipes_501})
    assert r.status_code == 400
    assert r.json()["errorCode"] == "VALIDATION_ERROR"


def test_fridge_ingredients_boundary_100(settings, store):
    """Tối đa 100 nguyên liệu tủ lạnh; 101 nguyên liệu phải báo lỗi 400."""
    fridge_100 = [f"nguyên liệu {i}" for i in range(100)]
    r, _ = call_api(settings, store, slots_json(), {**REQ, "fridgeIngredients": fridge_100})
    assert r.status_code == 200

    fridge_101 = [f"nguyên liệu {i}" for i in range(101)]
    r, _ = call_api(settings, store, slots_json(), {**REQ, "fridgeIngredients": fridge_101})
    assert r.status_code == 400
    assert r.json()["errorCode"] == "VALIDATION_ERROR"


def test_special_characters_and_injections_in_input(settings, store):
    """Ký tự đặc biệt, XSS, Unicode tiếng Việt và emoji không làm sập service."""
    payload = {
        **REQ,
        "customInstructions": "Ăn chay dưỡng sinh 🌱 <script>alert('xss')</script> -- DROP TABLE Recipes;",
        "fridgeIngredients": ["nấm đùi gà 🍄", "đậu hũ non (organic)", "cà chua bi & ớt chuông"],
    }
    r, llm = call_api(settings, store, slots_json(), payload)
    assert r.status_code == 200
    prompt_text = llm.generate_calls[0][1][0].text
    assert "Ăn chay dưỡng sinh 🌱" in prompt_text
    assert "nấm đùi gà 🍄" in prompt_text


# --- 3. KIỂM TRA XỬ LÝ LỖI GEMINI PHẢN HỒI BẤT THƯỜNG (GEMINI MALFORMED RESPONSES) ---

def test_gemini_returns_non_object_root(settings, store):
    """Gemini trả về mảng hoặc chuỗi thay vì object {"slots": ...}."""
    for bad_root in ("[]", '"chỉ là chuỗi"', "12345", "true", "null"):
        r, _ = call_api(settings, store, bad_root)
        assert r.status_code == 502
        assert r.json()["errorCode"] == "GEMINI_ERROR"


def test_gemini_missing_fields_in_slot(settings, store):
    """Mỗi ô thiếu dayOfWeek, mealType hoặc recipeId phải bị 502."""
    for missing_field in ("dayOfWeek", "mealType", "recipeId"):
        def override(slots, field=missing_field):
            bad = [dict(s) for s in slots]
            bad[0].pop(field, None)
            return bad

        r, _ = call_api(settings, store, slots_json(override=override))
        assert r.status_code == 502
        assert r.json()["errorCode"] == "GEMINI_ERROR"


def test_gemini_returns_float_or_null_recipe_id(settings, store):
    """recipeId là float (100.5) hoặc null phải bị 502."""
    for bad_id in (100.5, None, "100", {}):
        def override(slots, target_id=bad_id):
            bad = [dict(s) for s in slots]
            bad[0]["recipeId"] = target_id
            return bad

        r, _ = call_api(settings, store, slots_json(override=override))
        assert r.status_code == 502
        assert r.json()["errorCode"] == "GEMINI_ERROR"


def test_gemini_returns_negative_or_unauthorized_recipe_id(settings, store):
    """recipeId âm hoặc là 0 khi không có trong ứng viên phải bị 502."""
    def override(slots):
        bad = [dict(s) for s in slots]
        bad[0]["recipeId"] = -100
        return bad

    r, _ = call_api(settings, store, slots_json(override=override))
    assert r.status_code == 502
    assert r.json()["errorCode"] == "GEMINI_ERROR"


def test_gemini_raises_unexpected_exception(settings, store):
    """Khi thư viện ném lỗi bất ngờ, phải bọc thành 502 chứ không để lộ traceback 500."""
    llm = FakeLLM(fail=gemini_error("Lỗi mạng nội bộ Vertex AI"))
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        r = c.post("/meal/generate", json=REQ, headers=HEADERS)
    assert r.status_code == 502
    assert r.json()["errorCode"] == "GEMINI_ERROR"


# --- 4. KIỂM TRA HIỆU QUẢ CỦA TIMEOUT 4 GIÂY ---

class DelayLLM(FakeLLM):
    def __init__(self, delay_sec: float, reply: str):
        super().__init__(reply)
        self.delay_sec = delay_sec

    async def generate(self, system, messages, *, temperature, max_output_tokens, json_schema=None):
        await asyncio.sleep(self.delay_sec)
        return await super().generate(system, messages, temperature=temperature,
                                      max_output_tokens=max_output_tokens, json_schema=json_schema)


def test_timeout_exact_edge(settings, store, monkeypatch):
    """Nếu chạy trong hạn mức thì 200; nếu vượt quá hạn mức thì 504."""
    monkeypatch.setattr(settings, "meal_timeout_seconds", 0.05)

    # Chạy nhanh (0.01s < 0.05s) -> 200 OK
    fast_llm = DelayLLM(0.01, slots_json())
    with TestClient(create_app(settings, llm=fast_llm, store=store)) as c:
        r_fast = c.post("/meal/generate", json=REQ, headers=HEADERS)
    assert r_fast.status_code == 200

    # Chạy chậm (0.10s > 0.05s) -> 504 AI_TIMEOUT
    slow_llm = DelayLLM(0.10, slots_json())
    with TestClient(create_app(settings, llm=slow_llm, store=store)) as c:
        r_slow = c.post("/meal/generate", json=REQ, headers=HEADERS)
    assert r_slow.status_code == 504
    assert r_slow.json()["errorCode"] == "AI_TIMEOUT"
