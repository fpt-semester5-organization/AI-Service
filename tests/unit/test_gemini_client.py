"""GeminiClient với client Google giả: kiểm cấu hình yêu cầu, retry và ánh xạ lỗi (không gọi mạng)."""
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings
from app.core.errors import AiServiceError
from app.core.gemini_client import GeminiClient, Message, parse_json_object
from app.services.chat.prompt import RESPONSE_SCHEMA


def settings(**kw) -> Settings:
    return Settings(google_cloud_project="p", gemini_timeout_seconds=1, gemini_max_retries=2, _env_file=None, **kw)


class FakeModels:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = []

    async def _next(self, kind, **kw):
        self.calls.append((kind, kw))
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b

    async def generate_content(self, **kw): return await self._next("generate", **kw)
    async def embed_content(self, **kw): return await self._next("embed", **kw)
    async def count_tokens(self, **kw): return await self._next("count", **kw)


def client_with(behaviors, **kw):
    c = GeminiClient(settings(**kw))
    models = FakeModels(behaviors)
    c._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return c, models


def server_error(code=503):
    return genai_errors.ServerError(code, {"error": {"message": "x"}})


def gen_response(text='{"answer":"a","recommendedRecipeIds":[]}'):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=7))


async def test_generate_builds_expected_request_and_reads_usage():
    c, m = client_with([gen_response()])
    g = await c.generate("SYS", [Message("user", "hi")], temperature=0.3, max_output_tokens=256, json_schema=RESPONSE_SCHEMA)
    kw = m.calls[0][1]
    cfg: types.GenerateContentConfig = kw["config"]
    assert kw["model"] == "gemini-2.5-flash" and cfg.system_instruction == "SYS"
    assert cfg.response_mime_type == "application/json" and cfg.temperature == 0.3 and cfg.max_output_tokens == 256
    assert kw["contents"][0].role == "user" and kw["contents"][0].parts[0].text == "hi"
    assert (g.prompt_tokens, g.completion_tokens, g.model, g.temperature) == (11, 7, "gemini-2.5-flash", 0.3)


async def test_retries_transient_errors_then_succeeds():
    c, m = client_with([server_error(503), httpx.ConnectTimeout("t"), gen_response()])
    g = await c.generate("S", [Message("user", "x")], temperature=0, max_output_tokens=10)
    assert len(m.calls) == 3 and g.text


async def test_gives_up_after_retries_with_502():
    c, m = client_with([server_error(500)] * 3)
    with pytest.raises(AiServiceError) as e:
        await c.generate("S", [Message("user", "x")], temperature=0, max_output_tokens=10)
    assert e.value.status_code == 502 and e.value.error_code == "GEMINI_ERROR" and len(m.calls) == 3


async def test_client_error_is_not_retried():
    c, m = client_with([genai_errors.ClientError(400, {"error": {"message": "bad"}})])
    with pytest.raises(AiServiceError) as e:
        await c.generate("S", [Message("user", "x")], temperature=0, max_output_tokens=10)
    assert e.value.error_code == "GEMINI_ERROR" and len(m.calls) == 1


async def test_timeout_maps_to_504():
    c, _ = client_with([httpx.ReadTimeout("t")] * 3)
    with pytest.raises(AiServiceError) as e:
        await c.count_tokens("x")
    assert e.value.status_code == 504 and e.value.error_code == "AI_TIMEOUT"


async def test_empty_text_is_gemini_error():
    c, _ = client_with([gen_response("  ")])
    with pytest.raises(AiServiceError) as e:
        await c.generate("S", [Message("user", "x")], temperature=0, max_output_tokens=10)
    assert e.value.error_code == "GEMINI_ERROR"


async def test_embed_requests_task_and_768_dims():
    vec = SimpleNamespace(values=[0.1] * 768)
    c, m = client_with([SimpleNamespace(embeddings=[vec])])
    out = await c.embed("đạm", task="RETRIEVAL_QUERY")
    cfg = m.calls[0][1]["config"]
    assert len(out) == 768 and cfg.task_type == "RETRIEVAL_QUERY" and cfg.output_dimensionality == 768
    assert m.calls[0][1]["model"] == "gemini-embedding-001"


async def test_embed_wrong_dimension_is_rejected():
    c, _ = client_with([SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * 3072)])])
    with pytest.raises(AiServiceError) as e:
        await c.embed("x", task="RETRIEVAL_DOCUMENT")
    assert e.value.error_code == "GEMINI_ERROR"


async def test_count_tokens():
    c, _ = client_with([SimpleNamespace(total_tokens=42)])
    assert await c.count_tokens("x") == 42


def test_not_configured_without_project():
    assert GeminiClient(Settings(google_cloud_project="", _env_file=None)).configured() is False
    assert GeminiClient(settings()).configured() is True


async def test_missing_project_raises_gemini_error():
    with pytest.raises(AiServiceError) as e:
        await GeminiClient(Settings(google_cloud_project="", _env_file=None)).count_tokens("x")
    assert "GOOGLE_CLOUD_PROJECT" in e.value.message


def test_parse_json_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object("không phải json") is None and parse_json_object("[1]") is None
