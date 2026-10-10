"""Điều phối /chat/answer: lọc an toàn → RAG → dựng lời nhắc → Gemini → dựng sources từ kết quả tìm thật."""
import logging

from app.core.config import Settings
from app.core.errors import unsafe_content
from app.core.gemini_client import LLMClient, parse_json_object
from app.schemas.chat import Audit, ChatAnswerRequest, ChatAnswerResponse, ChatSource, Usage
from app.services.chat.prompt import NO_CONTEXT_PHRASE, RESPONSE_SCHEMA, build_prompt
from app.services.chat.safety import check_question
from app.services.rag.retriever import Retriever
from app.services.rag.types import Hit

log = logging.getLogger("veggielife.chat")


def _sources(hits: list[Hit]) -> list[ChatSource]:
    """sources = các đoạn thực sự đưa vào lời nhắc, gộp theo (loại, mã), giữ điểm cao nhất; không lấy từ văn bản AI sinh."""
    best: dict[tuple[str, str], Hit] = {}
    for h in hits:
        key = (h.source_type, h.source_id)
        if key not in best or h.score > best[key].score:
            best[key] = h
    ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return [ChatSource(source_type=h.source_type, source_id=h.source_id, title=h.title, score=round(h.score, 4))
            for h in ordered]


class ChatAnswerer:
    def __init__(self, settings: Settings, llm: LLMClient, retriever: Retriever):
        self._s = settings
        self._llm = llm
        self._retriever = retriever

    async def answer(self, req: ChatAnswerRequest) -> ChatAnswerResponse:
        violation = check_question(req.question)
        if violation:
            log.info("Câu hỏi bị chặn: %s", violation)
            raise unsafe_content()

        hits = await self._retriever.search(req.question)  # Qdrant lỗi/quá 3 giây → 503, không bỏ qua RAG
        prompt = build_prompt(req.question, req.history, req.profile, hits, req.candidate_recipes,
                              self._s.chat_history_turns)
        gen = await self._llm.generate(
            prompt.system, prompt.messages, temperature=self._s.chat_temperature,
            max_output_tokens=self._s.chat_max_output_tokens, json_schema=RESPONSE_SCHEMA)

        parsed = parse_json_object(gen.text)
        if parsed and isinstance(parsed.get("answer"), str) and parsed["answer"].strip():
            answer = parsed["answer"].strip()
            raw_ids = parsed.get("recommendedRecipeIds") or []
        else:
            answer, raw_ids = gen.text.strip(), []

        if not hits and NO_CONTEXT_PHRASE.lower() not in answer.lower():
            # AIQ-2: không có đoạn nào đạt ngưỡng thì luôn nói rõ "không có trong kho"; không phó mặc cho mô hình.
            answer = f"{NO_CONTEXT_PHRASE}. {answer}"

        allowed = {r.recipe_id for r in req.candidate_recipes}
        recommended: list[int] = []
        for rid in raw_ids:
            if isinstance(rid, int) and not isinstance(rid, bool) and rid in allowed and rid not in recommended:
                recommended.append(rid)

        return ChatAnswerResponse(
            answer=answer, sources=_sources(hits), rag_used=bool(hits), recommended_recipe_ids=recommended,
            usage=Usage(model=gen.model, prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                        latency_ms=gen.latency_ms, temperature=gen.temperature),
            audit=Audit(prompt=prompt.as_audit_text(), raw_response=gen.text))
