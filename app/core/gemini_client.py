"""Lớp gọi Gemini duy nhất của AI service, qua Vertex AI (D120): timeout, retry, ánh xạ lỗi.

Mọi module khác chỉ phụ thuộc giao thức `LLMClient` để thay bằng bản giả khi test.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings
from app.core.errors import ai_timeout, gemini_error

log = logging.getLogger("veggielife.gemini")


@dataclass
class Generation:
    """Kết quả một lần sinh văn bản."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    temperature: float


@dataclass
class ImagePart:
    """Ảnh gửi kèm lời nhắc (Vision, kiểm duyệt ảnh): byte thô + MIME đã được kiểm chữ ký tệp."""
    data: bytes
    mime_type: str


@dataclass
class Message:
    role: str  # "user" | "model"
    text: str
    images: tuple[ImagePart, ...] = ()  # chỉ dùng cho lượt "user"; thứ tự: văn bản trước, ảnh sau


class LLMClient(Protocol):
    async def generate(self, system: str, messages: list[Message], *, temperature: float,
                       max_output_tokens: int, json_schema: dict[str, Any] | None = None) -> Generation: ...

    async def embed(self, text: str, *, task: str) -> list[float]: ...

    async def count_tokens(self, text: str) -> int: ...

    def configured(self) -> bool: ...


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (408, 429) or (exc.code or 0) >= 500
    return False


class GeminiClient:
    def __init__(self, settings: Settings):
        self._s = settings
        self._client: genai.Client | None = None

    def configured(self) -> bool:
        return bool(self._s.google_cloud_project) if self._s.google_genai_use_vertexai else True

    def _get(self) -> genai.Client:
        if self._client is None:
            if not self.configured():
                raise gemini_error("Chưa cấu hình GOOGLE_CLOUD_PROJECT cho Vertex AI.")
            cred = self._s.google_application_credentials
            if cred:
                # Thư viện xác thực của Google chỉ đọc biến môi trường thật, không đọc .env.
                os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(Path(cred).expanduser().resolve()))
            self._client = genai.Client(
                vertexai=self._s.google_genai_use_vertexai,
                project=self._s.google_cloud_project or None,
                location=self._s.google_cloud_location,
                http_options=types.HttpOptions(timeout=int(self._s.gemini_timeout_seconds * 1000)),
            )
        return self._client

    async def _call(self, fn):
        attempts = self._s.gemini_max_retries + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                return await asyncio.wait_for(fn(), timeout=self._s.gemini_timeout_seconds + 2)
            except Exception as exc:  # noqa: BLE001 — ánh xạ về lỗi của hợp đồng
                last = exc
                if i < attempts - 1 and _retryable(exc):
                    await asyncio.sleep(0.5 * (2 ** i))
                    continue
                break
        log.warning("Gemini lỗi sau %s lần: %r", attempts, last)
        if isinstance(last, (httpx.TimeoutException, asyncio.TimeoutError)):
            raise ai_timeout() from last
        if isinstance(last, genai_errors.APIError) and last.code == 408:
            raise ai_timeout() from last
        raise gemini_error(f"Gemini/Vertex AI lỗi: {type(last).__name__}") from last

    async def generate(self, system: str, messages: list[Message], *, temperature: float,
                       max_output_tokens: int, json_schema: dict[str, Any] | None = None) -> Generation:
        client = self._get()
        contents = [types.Content(role=m.role, parts=[types.Part(text=m.text), *(
            types.Part.from_bytes(data=i.data, mime_type=i.mime_type) for i in m.images)]) for m in messages]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if json_schema else None,
            response_schema=json_schema,
        )
        started = time.perf_counter()
        resp = await self._call(lambda: client.aio.models.generate_content(
            model=self._s.gemini_model, contents=contents, config=cfg))
        latency = int((time.perf_counter() - started) * 1000)
        usage = getattr(resp, "usage_metadata", None)
        text = resp.text or ""
        if not text.strip():
            raise gemini_error("Gemini trả về nội dung rỗng.")
        return Generation(
            text=text, model=self._s.gemini_model,
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            latency_ms=latency, temperature=temperature)

    async def embed(self, text: str, *, task: str) -> list[float]:
        client = self._get()
        cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=self._s.embedding_dimensions)
        resp = await self._call(lambda: client.aio.models.embed_content(
            model=self._s.embedding_model, contents=text, config=cfg))
        if not resp.embeddings or not resp.embeddings[0].values:
            raise gemini_error("Embedding rỗng.")
        values = list(resp.embeddings[0].values)
        if len(values) != self._s.embedding_dimensions:
            raise gemini_error(f"Embedding có {len(values)} chiều, cần {self._s.embedding_dimensions}.")
        return values

    async def count_tokens(self, text: str) -> int:
        client = self._get()
        resp = await self._call(lambda: client.aio.models.count_tokens(
            model=self._s.gemini_model, contents=text))
        return int(resp.total_tokens or 0)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Đọc JSON từ văn bản mô hình trả; chịu được rào markdown ```json."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    try:
        v = json.loads(t)
    except ValueError:
        return None
    return v if isinstance(v, dict) else None
