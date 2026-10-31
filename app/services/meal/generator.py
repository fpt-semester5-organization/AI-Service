"""Điều phối /meal/generate (FN-MEAL-01): dựng lời nhắc → Gemini (JSON Schema) → kiểm lại 21 ô và recipeId ở tầng mã.

Backend còn kiểm lại calo ±10% và dị ứng; ở đây chỉ bảo đảm đầu ra đúng hình dạng và đúng danh sách ứng viên.
"""
import asyncio
import logging

from app.core.config import Settings
from app.core.errors import ai_timeout, gemini_error
from app.core.gemini_client import LLMClient, parse_json_object
from app.schemas.chat import Audit, Usage
from app.schemas.meal import MealGenerateRequest, MealGenerateResponse, MealSlot
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
