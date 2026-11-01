"""Điều phối /meal/generate và /meal/swap-by-note (FN-MEAL-01, FN-MEAL-03).

Dựng lời nhắc → Gemini (JSON Schema) → kiểm lại các ô và recipeId ở tầng mã.
Backend còn kiểm lại calo ±10% và dị ứng; ở đây bảo đảm đầu ra đúng hình dạng và đúng danh sách ứng viên.
"""
import asyncio
import logging

from app.core.config import Settings
from app.core.errors import ai_timeout, gemini_error
from app.core.gemini_client import LLMClient, Message, parse_json_object
from app.schemas.chat import Audit, CandidateRecipe, Usage
from app.schemas.meal import (
    MealGenerateRequest,
    MealGenerateResponse,
    MealSlot,
    SwapCandidateRecipe,
    SwapMealByNoteRequest,
    SwapMealByNoteResponse,
)
from app.services.meal.prompt import (
    DAYS,
    MEALS,
    RESPONSE_SCHEMA,
    SWAP_RESPONSE_SCHEMA,
    build_prompt,
    build_response_schema,
    build_swap_prompt,
    get_days_for_duration,
)


log = logging.getLogger("veggielife.meal")

MAX_TEMPERATURE = 0.4  # B.4: tác vụ có cấu trúc ≤ 0,4


VI_WEEKDAY_MAP: dict[str, str] = {
    "THỨ HAI": "MON",
    "THU HAI": "MON",
    "THỨ 2": "MON",
    "THU 2": "MON",
    "T2": "MON",
    "THỨ BA": "TUE",
    "THU BA": "TUE",
    "THỨ 3": "TUE",
    "THU 3": "TUE",
    "T3": "TUE",
    "THỨ TƯ": "WED",
    "THU TU": "WED",
    "THỨ 4": "WED",
    "THU 4": "WED",
    "T4": "WED",
    "THỨ NĂM": "THU",
    "THU NAM": "THU",
    "THỨ 5": "THU",
    "THU 5": "THU",
    "T5": "THU",
    "THỨ SÁU": "FRI",
    "THU SAU": "FRI",
    "THỨ 6": "FRI",
    "THU 6": "FRI",
    "T6": "FRI",
    "THỨ BẢY": "SAT",
    "THU BAY": "SAT",
    "THỨ 7": "SAT",
    "THU 7": "SAT",
    "T7": "SAT",
    "CHỦ NHẬT": "SUN",
    "CHU NHAT": "SUN",
    "CN": "SUN",
}


def resolve_day_index(day_str: str, target_days: list[str]) -> int | None:
    norm = day_str.strip().upper()
    if norm in target_days:
        return target_days.index(norm)
    if norm in VI_WEEKDAY_MAP:
        mapped = VI_WEEKDAY_MAP[norm]
        if mapped in target_days:
            return target_days.index(mapped)
    for prefix in ("DAY_", "DAY", "NGAY_", "NGAY", "NGÀY_", "NGÀY"):
        if norm.startswith(prefix):
            try:
                num = int(norm[len(prefix):].strip())
                if 1 <= num <= len(target_days):
                    return num - 1
            except ValueError:
                pass
    try:
        num = int(norm)
        if 1 <= num <= len(target_days):
            return num - 1
    except ValueError:
        pass
    return None


def validate_slots(
    raw: object,
    allowed: dict[int, CandidateRecipe] | set[int] | list[CandidateRecipe],
    duration_days: int = 7,
    start_date: str | None = None,
) -> list[MealSlot]:
    """Đủ duration_days * 3 ô duy nhất (ngày × bữa), recipeId thuộc ứng viên. Điền đầy đủ chứng minh calo, thời gian nấu, lượt tim, lý do."""
    expected_count = duration_days * 3
    if not isinstance(raw, list) or len(raw) != expected_count:
        actual = len(raw) if isinstance(raw, list) else "không phải danh sách"
        raise gemini_error(f"Gemini không trả đủ {expected_count} ô thực đơn (thực tế: {actual}).")

    if isinstance(allowed, dict):
        allowed_recipes = allowed
        allowed_ids = set(allowed.keys())
    elif isinstance(allowed, list):
        allowed_recipes = {r.recipe_id: r for r in allowed}
        allowed_ids = set(allowed_recipes.keys())
    else:
        allowed_ids = allowed
        allowed_recipes = {}

    target_days = get_days_for_duration(duration_days, start_date)
    slots: dict[tuple[str, str], MealSlot] = {}

    for item in raw:
        if not isinstance(item, dict):
            raise gemini_error("Gemini trả ô thực đơn sai định dạng.")
        day_raw, meal_raw, rid = item.get("dayOfWeek"), item.get("mealType"), item.get("recipeId")
        if not isinstance(day_raw, str):
            raise gemini_error("Gemini trả ngày hoặc bữa không hợp lệ.")
        day_idx = resolve_day_index(day_raw, target_days)
        meal = meal_raw.strip().upper() if isinstance(meal_raw, str) else None
        if day_idx is None or meal not in MEALS:
            raise gemini_error("Gemini trả ngày hoặc bữa không hợp lệ.")
        day = target_days[day_idx]
        if isinstance(rid, bool) or not isinstance(rid, int) or rid not in allowed_ids:
            raise gemini_error("Gemini chọn recipeId ngoài danh sách ứng viên.")
        if (day, meal) in slots:
            raise gemini_error("Gemini trả trùng ô thực đơn.")

        r = allowed_recipes.get(rid)
        kcal = item.get("caloriesKcal")
        if kcal is None and r is not None:
            kcal = r.calories_kcal
        cook_time = item.get("cookTimeMinutes")
        if cook_time is None and r is not None:
            cook_time = int(r.cook_time_minutes) if r.cook_time_minutes else None
        fav_count = item.get("favoriteCount")
        if fav_count is None and r is not None:
            fav_count = r.favorite_count or r.vote_count or r.view_count or 0
        match_reason = item.get("matchReason") or item.get("reason")
        if not match_reason:
            title = r.title if r else f"Món {rid}"
            match_reason = f"{title} thanh đạm, cung cấp khoảng {kcal or 'đầy đủ'} kcal, thời gian nấu nhanh gọn."

        slots[(day, meal)] = MealSlot(
            day_of_week=day,
            meal_type=meal,
            recipe_id=rid,
            calories_kcal=kcal,
            cook_time_minutes=cook_time,
            favorite_count=fav_count,
            match_reason=match_reason,
        )

    return [slots[(d, m)] for d in target_days for m in MEALS]


class MealGenerator:
    def __init__(self, settings: Settings, llm: LLMClient):
        self._s = settings
        self._llm = llm

    async def generate(self, req: MealGenerateRequest) -> MealGenerateResponse:
        duration_days = req.duration_days
        start_date = req.start_date
        prompt = build_prompt(
            req.target_daily_calories,
            req.candidate_recipes,
            req.fridge_ingredients,
            req.custom_instructions,
            req.attempt,
            duration_days=duration_days,
            start_date=start_date,
        )
        schema = build_response_schema(duration_days, start_date) if (duration_days != 7 or start_date) else RESPONSE_SCHEMA
        # Lần xếp lại tăng nhẹ temperature để có phương án khác, vẫn ≤ 0,4.
        temperature = min(MAX_TEMPERATURE, self._s.meal_temperature + 0.05 * (req.attempt - 1))
        try:
            gen = await asyncio.wait_for(
                self._llm.generate(
                    prompt.system,
                    prompt.messages,
                    temperature=temperature,
                    max_output_tokens=self._s.meal_max_output_tokens,
                    json_schema=schema,
                ),
                timeout=self._s.meal_timeout_seconds,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ai_timeout(f"Quá thời gian xếp thực đơn ({int(self._s.meal_timeout_seconds)} giây).") from exc

        data = parse_json_object(gen.text)
        if data is None:
            log.error("Gemini không trả JSON thực đơn hợp lệ. Raw response: %s", gen.text[:500])
            raise gemini_error("Gemini không trả JSON thực đơn hợp lệ.")

        recipe_dict = {r.recipe_id: r for r in req.candidate_recipes}
        try:
            slots = validate_slots(data.get("slots"), recipe_dict, duration_days=duration_days, start_date=start_date)
        except Exception as exc:
            log.error("Lỗi validate_slots: %s. Slots nhận được: %s", exc, data.get("slots"))
            raise
        return MealGenerateResponse(
            slots=slots,
            usage=Usage(
                model=gen.model,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                latency_ms=gen.latency_ms,
                temperature=gen.temperature,
            ),
            audit=Audit(prompt=prompt.as_audit_text(), raw_response=gen.text),
        )

    async def swap_by_note(self, req: SwapMealByNoteRequest) -> SwapMealByNoteResponse:
        excluded = set(req.excluded_recipe_ids) | {req.current_recipe_id}
        valid_candidates = [r for r in req.candidate_recipes if r.recipe_id not in excluded]
        if not valid_candidates:
            valid_candidates = [r for r in req.candidate_recipes if r.recipe_id != req.current_recipe_id] or req.candidate_recipes

        allowed_recipes = {r.recipe_id: r for r in valid_candidates}
        prompt = build_swap_prompt(
            req.current_recipe_id,
            req.swap_note,
            req.dietary_school_code,
            req.meal_type,
            req.target_calories,
            list(excluded),
            valid_candidates,
        )

        gen = None
        data = None
        try:
            gen = await asyncio.wait_for(
                self._llm.generate(
                    prompt.system,
                    prompt.messages,
                    temperature=0.3,
                    max_output_tokens=self._s.meal_max_output_tokens,
                    json_schema=SWAP_RESPONSE_SCHEMA,
                ),
                timeout=self._s.meal_timeout_seconds,
            )
            data = parse_json_object(gen.text)
        except Exception as ex:
            log.warning("AI swap-by-note LLM call failed or timed out: %s. Using fallback scoring.", ex)

        if isinstance(data, dict):
            sel_id = data.get("selectedRecipeId")
            if isinstance(sel_id, int) and sel_id in allowed_recipes:
                selected_r = allowed_recipes[sel_id]
                reason = data.get("reason") or f"Món ăn phù hợp theo ghi chú: {req.swap_note}"
                kcal = data.get("caloriesKcal")
                if kcal is None:
                    kcal = selected_r.calories_kcal
                cook_time = data.get("cookTimeMinutes")
                if cook_time is None:
                    cook_time = int(selected_r.cook_time_minutes) if selected_r.cook_time_minutes else None
                fav = data.get("favoriteCount")
                if fav is None:
                    fav = selected_r.favorite_count or selected_r.vote_count or selected_r.view_count or 0

                return SwapMealByNoteResponse(
                    success=True,
                    selected_recipe_id=sel_id,
                    reason=reason,
                    calories_kcal=kcal,
                    cook_time_minutes=cook_time,
                    favorite_count=fav,
                    usage=Usage(
                        model=gen.model,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_ms=gen.latency_ms,
                        temperature=gen.temperature,
                    ) if gen else None,
                    audit=Audit(prompt=prompt.as_audit_text(), raw_response=gen.text) if gen else None,
                )

        # Fallback Engine nếu LLM không khả dụng hoặc trả kết quả không hợp lệ
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
        clean_note = req.swap_note.strip().lower()
        for r in valid_candidates:
            score = 0
            title_lower = r.title.lower()
            desc_lower = (getattr(r, "description", None) or "").lower()
            ing_list = getattr(r, "ingredients", [])
            ing_names_lower = [i.ingredient_name.lower() for i in ing_list if getattr(i, "ingredient_name", None)]

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

            r_kcal = r.calories_kcal if r.calories_kcal is not None else 0.0
            scored.append((score, -abs(r_kcal - req.target_calories), r))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = scored[0][2]
        return SwapMealByNoteResponse(
            success=True,
            selected_recipe_id=chosen.recipe_id,
            reason=f"Đã chọn {chosen.title} theo yêu cầu: {req.swap_note}",
            calories_kcal=chosen.calories_kcal,
            cook_time_minutes=int(chosen.cook_time_minutes) if chosen.cook_time_minutes else None,
            favorite_count=chosen.favorite_count or chosen.vote_count or chosen.view_count or 0,
            usage=Usage(
                model=gen.model,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                latency_ms=gen.latency_ms,
                temperature=gen.temperature,
            ) if gen else None,
            audit=Audit(prompt=prompt.as_audit_text(), raw_response=gen.text if gen else "Fallback scoring"),
        )

