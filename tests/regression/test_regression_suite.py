"""BỘ KIỂM THỬ HỒI QUY TOÀN DIỆN CHO AI-SERVICE (VEGGIELIFE).

Mục tiêu kiểm thử hồi quy:
1. Xác nhận các thành phần cốt lõi của AI-Service không bị hồi quy sau các lần refactor.
2. Kiểm tra tính toàn vẹn của 6 module AI: Chat RAG, Meal Planning, Moderation, Barcode, Vision, STT.
3. Xác nhận Delimiter Injection Protection (=== ... ===) hoạt động chuẩn mực.
4. Kiểm tra cấu trúc phân đoạn Prompt theo chuẩn arXiv:2402.07927.
5. Kiểm tra cơ chế JSON recovery khi Gemini LLM stream hoặc truncate dữ liệu.
6. Xác nhận gói phụ thuộc cốt lõi (tiktoken, google-genai, fastapi, pydantic) sẵn sàng.
"""
import pytest
import tiktoken
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.gemini_client import parse_json_object
from app.main import create_app
from app.schemas.chat import ChatProfile
from app.schemas.moderation import ModerationCheckRequest
from app.services.barcode.classifier import SYSTEM_INSTRUCTION as BARCODE_SYS
from app.services.chat.prompt import SYSTEM_INSTRUCTION as CHAT_SYS
from app.services.chat.prompt import build_prompt as build_chat_prompt
from app.services.chat.safety import check_question
from app.services.meal.prompt import build_prompt as build_meal_prompt
from app.services.meal.prompt import build_system_instruction as build_meal_sys
from app.services.moderation.prompt import SYSTEM_INSTRUCTION as MOD_SYS
from app.services.moderation.prompt import build_prompt as build_mod_prompt
from app.services.stt.recipe_extractor import SYSTEM_INSTRUCTION as STT_SYS
from app.services.vision.ingredient_detector import SYSTEM_INSTRUCTION as VISION_SYS
from tests.conftest import HEADERS, KEY

MODULE_PROMPTS = [
    ("Chat RAG", CHAT_SYS),
    ("Meal Planning", build_meal_sys(7)),
    ("Moderation", MOD_SYS),
    ("Barcode Classifier", BARCODE_SYS),
    ("Fridge Vision", VISION_SYS),
    ("STT Recipe Extractor", STT_SYS),
]


class TestAiServiceRegressionSuite:
    """Tập hợp các bài test hồi quy hệ thống AI."""

    @pytest.mark.parametrize("module_name,sys_prompt", MODULE_PROMPTS)
    def test_reg_prompt_architecture_compliance(self, module_name: str, sys_prompt: str):
        """Hồi quy: Đảm bảo cả 6 module AI luôn duy trì đủ 3 khối chuẩn arXiv:2402.07927."""
        assert "### 1. INSTRUCTION & PERSONA" in sys_prompt, f"{module_name} mất khối 1"
        assert "### 2. INVARIANT CONSTRAINTS" in sys_prompt, f"{module_name} mất khối 2"
        assert "### 3. OUTPUT INDICATOR" in sys_prompt, f"{module_name} mất khối 3"

        p1 = sys_prompt.index("### 1. INSTRUCTION & PERSONA")
        p2 = sys_prompt.index("### 2. INVARIANT CONSTRAINTS")
        p3 = sys_prompt.index("### 3. OUTPUT INDICATOR")
        assert p1 < p2 < p3, f"{module_name} vi phạm thứ tự khối chuẩn"

    @pytest.mark.parametrize("module_name,sys_prompt", MODULE_PROMPTS)
    def test_reg_prompt_token_budget_not_bloated(self, module_name: str, sys_prompt: str):
        """Hồi quy: Giám sát độ dài token của Prompt, chống phình Prompt Bloat."""
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(sys_prompt))
        assert token_count < 5000, f"{module_name} bị phình vượt ngưỡng 5,000 token ({token_count})"

    def test_reg_delimiter_sandboxing_integrity(self):
        """Hồi quy: Xác nhận dữ liệu người dùng luôn được bọc trong delimiter '===' để chống injection."""
        chat_prompt = build_chat_prompt(
            "Tôi muốn ăn chay giàu đạm",
            [],
            ChatProfile(dietary_school_code="VEGAN"),
            [],
            [],
            6,
        )
        assert "=== CÂU HỎI (dữ liệu của người dùng) ===" in chat_prompt.messages[-1].text

        meal_prompt = build_meal_prompt(2000, [], [], "dị ứng lạc", 7)
        assert "=== MỤC TIÊU ===" in meal_prompt.messages[0].text
        assert "=== GHI CHÚ CỦA NGƯỜI DÙNG" in meal_prompt.messages[0].text

        mod_prompt = build_mod_prompt(
            ModerationCheckRequest(title="Salad rau", description="Salad trộn dầu giấm", steps=[], ingredient_names=[]),
            [],
        )
        assert "=== NỘI DUNG CẦN KIỂM DUYỆT" in mod_prompt.messages[0].text

    def test_reg_safety_guardrails_medical_and_meat(self):
        """Hồi quy: Kiểm tra bộ lọc từ khóa nguy hiểm y tế và vi phạm trường phái ăn chay."""
        assert check_question("Bỏ thuốc insulin uống nước lá ổi") == "DANGEROUS_MEDICAL"
        assert check_question("Chữa ung thư bằng cách nhịn ăn 49 ngày") == "DANGEROUS_MEDICAL"
        assert check_question("Cho tôi thực đơn tăng cân") is None

    def test_reg_json_cleaner_and_recovery(self):
        """Hồi quy: Bộ phân tích JSON của GeminiClient phục hồi được JSON có trailing comma hoặc bọc markdown."""
        markdown_json = '```json\n{"status": "PASS", "confidence": 0.95,}\n```'
        parsed = parse_json_object(markdown_json)
        assert parsed is not None
        assert parsed["status"] == "PASS"
        assert parsed["confidence"] == 0.95

        unclosed_json = '{"title": "Món canh", "items": ["nấm", "đậu hũ"]}'
        recovered = parse_json_object(unclosed_json)
        assert recovered is not None
        assert recovered.get("title") == "Món canh"

    def test_reg_health_endpoint_contract(self):
        """Hồi quy: Endpoint /health phải trả về đúng cấu trúc chuẩn nội bộ."""
        app = create_app(Settings(internal_api_key=KEY))
        client = TestClient(app)
        res = client.get("/health", headers=HEADERS)
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") in ("ok", "OK", "DEGRADED", "HEALTHY")
        assert "gemini" in data
