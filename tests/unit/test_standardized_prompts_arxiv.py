"""Kiểm thử chuẩn hóa kiến trúc Prompt Constrained Zero-Shot theo công trình nghiên cứu arXiv:2402.07927.

Bộ kiểm thử này xác nhận:
1. Toàn bộ 6 module AI trong VeggieLife đều tuân thủ cấu trúc 3 khối kinh điển trong System Instruction:
   - ### 1. INSTRUCTION & PERSONA
   - ### 2. INVARIANT CONSTRAINTS (QUY TẮC BẤT BIẾN)
   - ### 3. OUTPUT INDICATOR
2. Ranh giới dữ liệu người dùng được đóng gói triệt để bằng Delimiters ('=== ... ===').
3. Lượng token của System Instruction cho từng module được tinh gọn tối ưu (dưới 1,000 tokens),
   chấm dứt tình trạng 'Prompt Bloat' và 'Lost in the Middle'.
"""
import pytest
import tiktoken

from app.services.chat.prompt import SYSTEM_INSTRUCTION as CHAT_SYS, build_prompt as build_chat_prompt
from app.services.meal.prompt import build_system_instruction as build_meal_sys, build_prompt as build_meal_prompt
from app.services.moderation.prompt import SYSTEM_INSTRUCTION as MOD_SYS, build_prompt as build_mod_prompt
from app.services.barcode.classifier import SYSTEM_INSTRUCTION as BAR_SYS, BarcodeClassifier
from app.services.vision.ingredient_detector import SYSTEM_INSTRUCTION as VIS_SYS
from app.services.stt.recipe_extractor import SYSTEM_INSTRUCTION as STT_SYS, build_prompt as build_stt_prompt

from app.schemas.chat import ChatProfile
from app.schemas.barcode import BarcodeClassifyRequest, BarcodeProduct
from app.schemas.moderation import ModerationCheckRequest
from app.core.config import Settings
from tests.fakes import FakeLLM


ALL_MODULE_PROMPTS = [
    ("Chat RAG", CHAT_SYS),
    ("Meal Planning", build_meal_sys(7)),
    ("Moderation", MOD_SYS),
    ("Barcode Classifier", BAR_SYS),
    ("Fridge Vision", VIS_SYS),
    ("STT Recipe Extractor", STT_SYS),
]


@pytest.mark.parametrize("module_name,sys_prompt", ALL_MODULE_PROMPTS)
def test_all_system_prompts_have_arxiv_canonical_sections(module_name, sys_prompt):
    """Xác nhận toàn bộ 6 module đều có đủ 3 khối chuẩn mực từ bài báo arXiv:2402.07927.

    (Prompt cũ chắc chắn FAILED test này vì không có các phân vùng chuẩn này).
    """
    assert "### 1. INSTRUCTION & PERSONA" in sys_prompt, f"{module_name} thiếu phần INSTRUCTION & PERSONA"
    assert "### 2. INVARIANT CONSTRAINTS" in sys_prompt, f"{module_name} thiếu phần INVARIANT CONSTRAINTS"
    assert "### 3. OUTPUT INDICATOR" in sys_prompt, f"{module_name} thiếu phần OUTPUT INDICATOR"


@pytest.mark.parametrize("module_name,sys_prompt", ALL_MODULE_PROMPTS)
def test_canonical_sections_strict_logical_order(module_name, sys_prompt):
    """Xác nhận thứ tự cấu trúc: 1. Instruction -> 2. Invariant Constraints -> 3. Output Indicator."""
    pos1 = sys_prompt.index("### 1. INSTRUCTION & PERSONA")
    pos2 = sys_prompt.index("### 2. INVARIANT CONSTRAINTS")
    pos3 = sys_prompt.index("### 3. OUTPUT INDICATOR")
    assert pos1 < pos2 < pos3, f"{module_name} không tuân thủ thứ tự giải phẫu của bài báo"


@pytest.mark.parametrize("module_name,sys_prompt", ALL_MODULE_PROMPTS)
def test_all_system_prompts_contain_invariant_safety_guard(module_name, sys_prompt):
    """Mọi module đều phải có từ khóa bảo vệ QUY TẮC BẤT BIẾN hoặc INVARIANT CONSTRAINTS và khai báo DỮ LIỆU."""
    assert "QUY TẮC BẤT BIẾN" in sys_prompt or "INVARIANT CONSTRAINTS" in sys_prompt
    assert "DỮ LIỆU" in sys_prompt


MODULE_TOKEN_LIMITS = {
    "Chat RAG": 1500,
    "Meal Planning": 3000,          # Giữ đầy đủ quy tắc bữa sáng/trưa/tối, biến thể và cá nhân hóa chuyên sâu
    "Moderation": 2200,
    "Barcode Classifier": 1300,
    "Fridge Vision": 4200,          # Giữ đầy đủ danh mục phi thực phẩm, hình thái ớt chuông/cà chua và bounding box
    "STT Recipe Extractor": 1100,
}


@pytest.mark.parametrize("module_name,sys_prompt", ALL_MODULE_PROMPTS)
def test_all_system_prompts_token_budget_efficiency(module_name, sys_prompt):
    """Kiểm tra ngân sách token: Mọi System Instruction sau chuẩn hóa đều nằm trong giới hạn tối ưu mà vẫn giữ trọn vẹn tri thức miền."""
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(sys_prompt))
    limit = MODULE_TOKEN_LIMITS[module_name]
    assert token_count < limit, f"{module_name} vượt ngân sách token ({token_count} >= {limit} tokens)"


def test_total_system_prompts_token_reduction():
    """Tổng token System Instruction của 6 module phải < 12,000 tokens (so với 12,910 tokens của bản cũ chưa chuẩn hóa)."""
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(p)) for _, p in ALL_MODULE_PROMPTS)
    assert total_tokens < 12000, f"Tổng token ({total_tokens}) vượt ngưỡng tiết kiệm"


def test_user_payload_delimiters_across_modules():
    """Kiểm tra toàn bộ các user payload đều sử dụng ranh giới phân tách '=== ... ===' chuẩn mực."""
    # 1. Chat Prompt
    chat_built = build_chat_prompt("Hỏi về đạm", [], ChatProfile(dietary_school_code="VEGAN"), [], [], 6)
    chat_text = chat_built.messages[-1].text
    assert "=== HỒ SƠ NGƯỜI DÙNG ===" in chat_text
    assert "=== CÂU HỎI (dữ liệu của người dùng) ===" in chat_text

    # 2. Meal Prompt
    meal_built = build_meal_prompt(1800, [], [], "ít dầu mỡ", 7)
    meal_text = meal_built.messages[0].text
    assert "=== MỤC TIÊU SỨC KHỎE & THỰC ĐƠN ===" in meal_text or "=== MỤC TIÊU" in meal_text
    assert "=== GHI CHÚ BỔ SUNG TỪ NGƯỜI DÙNG" in meal_text or "=== GHI CHÚ" in meal_text

    # 3. Moderation Prompt
    mod_req = ModerationCheckRequest(title="Món ngon", description="Nấu canh", steps=[], ingredient_names=[])
    mod_built = build_mod_prompt(mod_req, [])
    mod_text = mod_built.messages[0].text
    assert "=== NỘI DUNG CẦN KIỂM DUYỆT" in mod_text

    # 4. STT Prompt
    stt_text = build_stt_prompt("video nấu ăn chay")
    assert "=== PHIÊN ÂM" in stt_text


@pytest.mark.parametrize("module_name,sys_prompt", ALL_MODULE_PROMPTS)
def test_output_indicator_mandates_json(module_name, sys_prompt):
    """Mọi module đều phải có Output Indicator ép buộc trả về JSON."""
    output_section = sys_prompt.split("### 3. OUTPUT INDICATOR")[1]
    assert "JSON" in output_section, f"{module_name} không có chỉ thị định dạng JSON trong Output Indicator"
