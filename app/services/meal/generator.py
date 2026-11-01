"""Điều phối /meal/generate (FN-MEAL-01): dựng lời nhắc → Gemini (JSON Schema) → kiểm lại 21 ô và recipeId ở tầng mã.

Backend còn kiểm lại calo ±10% và dị ứng; ở đây chỉ bảo đảm đầu ra đúng hình dạng và đúng danh sách ứng viên.
"""
import asyncio
import logging

from app.core.config import Settings
from app.core.errors import ai_timeout, gemini_error
from app.core.gemini_client import LLMClient, Message, parse_json_object
from app.schemas.chat import Audit, Usage
from app.schemas.meal import (
    MealGenerateRequest,
    MealGenerateResponse,
    MealSlot,
    MealSwapByNoteRequest,
    MealSwapByNoteResponse,
)
from app.services.meal.prompt import DAYS, MEALS, RESPONSE_SCHEMA, build_prompt

log = logging.getLogger("veggielife.meal")

MAX_TEMPERATURE = 0.4  # B.4: tác vụ có cấu trúc ≤ 0,4


def validate_slots(raw: object, allowed_ids: set[int]) -> list[MealSlot]:
    """Đủ 21 ô duy nhất (ngày × bữa), recipeId thuộc ứng viên. Sai thì lỗi rõ ràng, không tự vá ô."""
    if not isinstance(raw, list) or len(raw) != 21:
        raise gemini_error("Gemini không trả đủ 21 ô thực đơn.")
    slots: dict[tuple[str, str], MealSlot] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise gemini_error("Gemini trả ô thực đơn sai định dạng.")
        day, meal, rid = item.get("dayOfWeek"), item.get("mealType"), item.get("recipeId")
        if day not in DAYS or meal not in MEALS:
            raise gemini_error("Gemini trả ngày hoặc bữa không hợp lệ.")
        if isinstance(rid, bool) or not isinstance(rid, int) or rid not in allowed_ids:
            raise gemini_error("Gemini chọn recipeId ngoài danh sách ứng viên.")
        if (day, meal) in slots:
            raise gemini_error("Gemini trả trùng ô thực đơn.")
        slots[(day, meal)] = MealSlot(day_of_week=day, meal_type=meal, recipe_id=rid)
    return [slots[(d, m)] for d in DAYS for m in MEALS]


class MealGenerator:
    def __init__(self, settings: Settings, llm: LLMClient):
        self._s = settings
        self._llm = llm

    async def generate(self, req: MealGenerateRequest) -> MealGenerateResponse:
        prompt = build_prompt(req.target_daily_calories, req.candidate_recipes, req.fridge_ingredients,
                              req.custom_instructions, req.attempt)
        # Lần xếp lại tăng nhẹ temperature để có phương án khác, vẫn ≤ 0,4.
        temperature = min(MAX_TEMPERATURE, self._s.meal_temperature + 0.05 * (req.attempt - 1))
        try:
            gen = await asyncio.wait_for(
                self._llm.generate(
                    prompt.system,
                    prompt.messages,
                    temperature=temperature,
                    max_output_tokens=self._s.meal_max_output_tokens,
                    json_schema=RESPONSE_SCHEMA,
                ),
                timeout=self._s.meal_timeout_seconds,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ai_timeout("Quá thời gian xếp thực đơn (4 giây).") from exc

        data = parse_json_object(gen.text)
        if data is None:
            raise gemini_error("Gemini không trả JSON thực đơn hợp lệ.")
        slots = validate_slots(data.get("slots"), {r.recipe_id for r in req.candidate_recipes})
        return MealGenerateResponse(
            slots=slots,
            usage=Usage(model=gen.model, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                        latency_ms=gen.latency_ms, temperature=gen.temperature),
            audit=Audit(prompt=prompt.as_audit_text(), raw_response=gen.text))

    async def swap_by_note(self, req: MealSwapByNoteRequest) -> MealSwapByNoteResponse:
        excluded = set(req.excluded_recipe_ids) | {req.current_recipe_id}
        valid_candidates = [r for r in req.candidate_recipes if r.recipe_id not in excluded]
        if not valid_candidates:
            valid_candidates = req.candidate_recipes

        candidate_lines = []
        for r in valid_candidates[:60]:
            ing_names = [i.ingredient_name for i in r.ingredients if i.ingredient_name]
            ing_str = ", ".join(ing_names[:8]) if ing_names else "rau củ thanh đạm"
            candidate_lines.append(
                f"- #{r.recipe_id}: {r.title} ({round(r.calories_kcal)} kcal) [Nguyên liệu: {ing_str}]"
            )
        candidates_text = "\n".join(candidate_lines)

        system_instruction = (
            "Bạn là chuyên gia dinh dưỡng và đầu bếp món chay của VeggieLife.\n"
            "Nhiệm vụ: Chọn MỘT món chay phù hợp nhất từ danh sách ứng viên để đổi món theo yêu cầu của người dùng.\n"
            "QUY TẮC:\n"
            "1. Chỉ chọn recipeId có trong danh sách ứng viên.\n"
            "2. ƯU TIÊN SỐ 1: Món có nguyên liệu hoặc tên thỏa mãn yêu cầu của người dùng (ví dụ: 'có ớt', 'nhiều nấm', 'bí đỏ').\n"
            "3. Ưu tiên số 2: Lượng calo gần với calo mục tiêu.\n"
            "4. Trả về đúng định dạng JSON: {\"selectedRecipeId\": <id>, \"reason\": \"<lý do chọn món>\"}."
        )

        user_prompt = (
            f"MỤC TIÊU ĐỔI BỮA ĂN:\n"
            f"- Bữa: {req.meal_type} | Trường phái: {req.dietary_school_code} | Calo mục tiêu: {round(req.target_calories)} kcal\n"
            f"- Ghi chú / Yêu cầu người dùng: \"{req.swap_note}\"\n\n"
            f"DANH SÁCH ỨNG VIÊN:\n"
            f"{candidates_text}\n\n"
            f"Hãy chọn món tối ưu nhất đáp ứng đúng yêu cầu của người dùng và trả về JSON."
        )

        schema = {
            "type": "object",
            "properties": {
                "selectedRecipeId": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["selectedRecipeId", "reason"],
            "additionalProperties": False,
        }

        try:
            gen = await asyncio.wait_for(
                self._llm.generate(
                    system_instruction,
                    [Message(role="user", text=user_prompt)],
                    temperature=0.2,
                    max_output_tokens=300,
                    json_schema=schema,
                ),
                timeout=3.5,
            )
            data = parse_json_object(gen.text)
            if data and isinstance(data.get("selectedRecipeId"), int):
                sel_id = data["selectedRecipeId"]
                if sel_id in {r.recipe_id for r in valid_candidates}:
                    return MealSwapByNoteResponse(
                        success=True,
                        selected_recipe_id=sel_id,
                        reason=data.get("reason", f"AI chọn theo ghi chú: {req.swap_note}"),
                        usage=Usage(
                            model=gen.model,
                            prompt_tokens=gen.prompt_tokens,
                            completion_tokens=gen.completion_tokens,
                            latency_ms=gen.latency_ms,
                            temperature=gen.temperature,
                        ),
                    )
        except Exception as ex:
            log.warning("AI swap-by-note LLM call failed or timed out: %s. Using fallback scoring.", ex)

        # Fallback Engine nếu LLM không khả dụng
        import re
        stop_words = {
            "yêu", "cầu", "muốn", "thích", "không", "món", "cho", "với", "nhiều", "ít", "được", "mình",
            "ngày", "bữa", "ăn", "tôi", "có", "các", "của", "những", "và", "nhưng", "hoặc", "lại", "thì", "là"
        }
        tokens = [
            t.lower()
            for t in re.split(r"[\s,.;!?+\-/_()\[\]:\"']+", req.swap_note)
            if len(t) >= 2 and t.lower() not in stop_words
        ]

        scored = []
        for r in valid_candidates:
            score = 0
            title_lower = r.title.lower()
            desc_lower = (r.description or "").lower()
            ing_names_lower = [i.ingredient_name.lower() for i in r.ingredients if i.ingredient_name]

            clean_note = req.swap_note.strip().lower()
            if len(clean_note) >= 3:
                if clean_note in title_lower:
                    score += 30
                if any(clean_note in ing for ing in ing_names_lower):
                    score += 25
                if clean_note in desc_lower:
                    score += 15

            for t in tokens:
                if t in title_lower:
                    score += 10
                if any(t in ing for ing in ing_names_lower):
                    score += 12
                if t in desc_lower:
                    score += 4

            scored.append((score, -abs(r.calories_kcal - req.target_calories), r))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = scored[0][2]
        return MealSwapByNoteResponse(
            success=True,
            selected_recipe_id=chosen.recipe_id,
            reason=f"Đã chọn {chosen.title} theo yêu cầu: {req.swap_note}",
            usage=None,
        )
