"""Điều phối /moderation/check (FN-POST-03/04/07): tải ảnh an toàn → Gemini (văn bản + ảnh) → chuẩn hóa.

Chỉ trả điểm và lý do. Ngưỡng 85%, Red Alert (kiểm tất định theo nhãn trường phái) và trạng thái bài do Backend quyết.
"""
import logging

import httpx

from app.core.config import Settings
from app.core.errors import gemini_error
from app.core.gemini_client import LLMClient, parse_json_object
from app.schemas.chat import Audit, Usage
from app.schemas.moderation import ModerationCheckRequest, ModerationCheckResponse
from app.services.moderation.images import fetch_images
from app.services.moderation.prompt import RESPONSE_SCHEMA, VIOLATION_TYPES, build_prompt

log = logging.getLogger("veggielife.moderation")

MAX_REASONS = 5
MAX_REASON_CHARS = 300


class ContentModerator:
    def __init__(self, settings: Settings, llm: LLMClient, http_client: httpx.AsyncClient | None = None):
        self._s = settings
        self._llm = llm
        self._http = http_client

    async def check(self, req: ModerationCheckRequest) -> ModerationCheckResponse:
        images = await fetch_images(req.image_urls, self._s, self._http)
        prompt = build_prompt(req, images)
        gen = await self._llm.generate(prompt.system, prompt.messages, temperature=min(0.4, self._s.moderation_temperature),
                                       max_output_tokens=self._s.moderation_max_output_tokens, json_schema=RESPONSE_SCHEMA)
        data = parse_json_object(gen.text)
        if data is None:
            raise gemini_error("Gemini không trả JSON kiểm duyệt hợp lệ.")
        score, vtype = data.get("violationScore"), data.get("violationType")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or vtype not in VIOLATION_TYPES:
            raise gemini_error("Gemini trả điểm hoặc loại vi phạm không hợp lệ.")
        reasons = [r.strip()[:MAX_REASON_CHARS] for r in (data.get("reasons") or [])
                   if isinstance(r, str) and r.strip()][:MAX_REASONS]
        audit_prompt = prompt.as_audit_text() + (f"\n[IMAGES {len(images)}]" if images else "")
        return ModerationCheckResponse(
            violation_score=round(min(1.0, max(0.0, float(score))), 4), violation_type=vtype, reasons=reasons,
            usage=Usage(model=gen.model, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                        latency_ms=gen.latency_ms, temperature=gen.temperature),
            audit=Audit(prompt=audit_prompt, raw_response=gen.text))
