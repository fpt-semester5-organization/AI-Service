"""Lớp gọi Gemini duy nhất của AI service, qua Vertex AI (D120): timeout, retry, ánh xạ lỗi.

Mọi module khác chỉ phụ thuộc giao thức `LLMClient` để thay bằng bản giả khi test.
"""
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import BASE_DIR, Settings
from app.core.errors import AiServiceError, ai_timeout, gemini_error

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
                       max_output_tokens: int, json_schema: dict[str, Any] | None = None,
                       timeout: float | None = None) -> Generation: ...

    async def embed(self, text: str, *, task: str) -> list[float]: ...

    async def count_tokens(self, text: str) -> int: ...

    def configured(self) -> bool: ...


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError)):
        return False
    if isinstance(exc, genai_errors.APIError):
        if exc.code in (408, 504):
            return False
        return exc.code == 429 or (exc.code or 0) >= 500
    return False


class GeminiClient:
    def __init__(self, settings: Settings):
        self._s = settings
        self._client: genai.Client | None = None
        self._current_key_idx: int = 0

    def _find_service_account_file(self) -> Path | None:
        cred = self._s.google_application_credentials
        if cred:
            p = Path(cred).expanduser()
            if p.is_file():
                return p.resolve()
            base_p = (BASE_DIR / p).resolve()
            if base_p.is_file():
                return base_p
        # Tự động tìm trong thư mục secrets/ nếu chưa cấu hình
        secrets_dir = (BASE_DIR / "secrets").resolve()
        if secrets_dir.is_dir():
            for p in sorted(secrets_dir.glob("*.json")):
                if p.is_file():
                    return p.resolve()
        return None

    def configured(self) -> bool:
        if self._s.google_genai_use_vertexai:
            return bool(self._s.google_cloud_project)
        return bool(self._s.gemini_api_key or os.environ.get("GEMINI_API_KEY"))

    def _get_api_keys(self) -> list[str]:
        raw = self._s.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        return [k.strip() for k in raw.split(",") if k.strip()]

    def _get(self) -> genai.Client:
        if self._client is None:
            if not self.configured():
                raise gemini_error("Chưa cấu hình GOOGLE_CLOUD_PROJECT cho Vertex AI hoặc GEMINI_API_KEY.")
            cred_file = self._find_service_account_file()
            use_vertex = self._s.google_genai_use_vertexai
            keys = self._get_api_keys()
            project_id = self._s.google_cloud_project

            if use_vertex:
                if cred_file and cred_file.is_file():
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_file)
                    if not project_id:
                        try:
                            sa_data = json.loads(cred_file.read_text(encoding="utf-8"))
                            if isinstance(sa_data, dict) and sa_data.get("project_id"):
                                project_id = sa_data["project_id"]
                        except Exception:
                            pass
                elif not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and keys:
                    try:
                        import google.auth
                        google.auth.default()
                    except Exception:
                        log.warning("Không tìm thấy Google ADC credentials. Tự động chuyển sang Google AI Studio bằng GEMINI_API_KEY.")
                        use_vertex = False
                elif not cred_file and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and keys:
                    use_vertex = False

            if not self.configured() and not project_id and not (keys and not use_vertex):
                raise gemini_error("Chưa cấu hình GOOGLE_CLOUD_PROJECT cho Vertex AI hoặc GEMINI_API_KEY.")

            api_key = keys[self._current_key_idx % len(keys)] if (keys and not use_vertex) else None

            self._client = genai.Client(
                vertexai=use_vertex,
                project=project_id or None if use_vertex else None,
                location=self._s.google_cloud_location if use_vertex else None,
                api_key=api_key,
            )
        return self._client

    def _rotate_key_or_mode(self):
        keys = self._get_api_keys()
        if len(keys) > 1 and isinstance(self._client, genai.Client):
            self._current_key_idx = (self._current_key_idx + 1) % len(keys)
            log.warning("Đã tự động chuyển sang API key dự phòng số %s", self._current_key_idx + 1)
            self._client = None


    async def _call(self, fn, timeout: float | None = None):
        effective_timeout = timeout or self._s.gemini_timeout_seconds
        attempts = self._s.gemini_max_retries + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                if effective_timeout and effective_timeout > 0:
                    return await asyncio.wait_for(fn(), timeout=effective_timeout)
                return await fn()
            except Exception as exc:  # noqa: BLE001 — ánh xạ về lỗi của hợp đồng
                last = exc
                if isinstance(exc, genai_errors.APIError) and exc.code in (429, 503):
                    self._rotate_key_or_mode()
                if i < attempts - 1 and _retryable(exc):
                    delay = 0.5 * (2 ** i)
                    if isinstance(exc, genai_errors.APIError) and exc.code == 429:
                        delay = max(delay, 2.0 + 1.0 * i)
                    log.warning("Gemini lỗi lần %s (mã %s), thử lại sau %.1fs: %r", i + 1, getattr(exc, "code", None), delay, exc)
                    await asyncio.sleep(delay)
                    continue
                break
        log.warning("Gemini lỗi sau %s lần: %r", attempts, last)
        if isinstance(last, (httpx.TimeoutException, asyncio.TimeoutError)):
            raise ai_timeout() from last
        if isinstance(last, genai_errors.APIError) and last.code in (408, 504):
            raise ai_timeout() from last
        if isinstance(last, AiServiceError):
            raise last
        err_msg = getattr(last, "message", None) or str(last)
        raise gemini_error(f"Gemini/Vertex AI lỗi ({type(last).__name__}): {err_msg}") from last

    def _resolve_model(self) -> str:
        model = self._s.gemini_model.strip()
        if self._s.google_genai_use_vertexai:
            return model
        if model.lower() in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-3.8-flash"):
            return "gemini-3.5-flash"
        return model


    async def generate(self, system: str, messages: list[Message], *, temperature: float,
                       max_output_tokens: int, json_schema: dict[str, Any] | None = None,
                       timeout: float | None = None) -> Generation:
        client = self._get()
        contents = [types.Content(role=m.role, parts=[types.Part(text=m.text), *(
            types.Part.from_bytes(data=i.data, mime_type=i.mime_type) for i in m.images)]) for m in messages]
        effective_timeout = timeout or self._s.gemini_timeout_seconds
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json" if json_schema else None,
            "response_schema": json_schema,
        }
        if effective_timeout and effective_timeout > 0:
            cfg_kwargs["http_options"] = types.HttpOptions(timeout=int(effective_timeout * 1000))
        target_model = self._resolve_model()
        model_lower = target_model.lower()
        if any(tag in model_lower for tag in ("2.5", "2.0", "3.0", "3.1", "3.5", "3.6", "3.7", "3.8", "flash", "thinking")):
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        cfg = types.GenerateContentConfig(**cfg_kwargs)
        started = time.perf_counter()
        resp = await self._call(lambda: self._get().aio.models.generate_content(
            model=target_model, contents=contents, config=cfg), timeout=effective_timeout)
        latency = int((time.perf_counter() - started) * 1000)
        usage = getattr(resp, "usage_metadata", None)
        text = resp.text or ""
        if not text.strip():
            if getattr(resp, "candidates", None) and resp.candidates:
                c0 = resp.candidates[0]
                if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                    text = "".join(getattr(p, "text", "") or "" for p in c0.content.parts)
                finish_reason = getattr(c0, "finish_reason", None)
                if not text.strip() and finish_reason:
                    log.warning("Gemini trả nội dung rỗng do finish_reason=%s", finish_reason)
                    text = '{"items": []}'
        if not text.strip():
            raise gemini_error("Gemini trả về nội dung rỗng.")

        return Generation(
            text=text, model=target_model,
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
        target_model = self._resolve_model()
        resp = await self._call(lambda: self._get().aio.models.count_tokens(
            model=target_model, contents=text))
        return int(resp.total_tokens or 0)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Đọc JSON object (dict) từ văn bản mô hình trả; chịu được rào markdown ```json, text xung quanh, trailing comma và JSON bị cắt ngắn."""
    t = text.strip()
    if not t:
        return None

    # 1. Thử parse trực tiếp
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
        return None
    except Exception:
        pass

    # 2. Thử tìm khối code markdown ```json ... ``` hoặc ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    if match:
        snippet = match.group(1).strip()
        try:
            v = json.loads(snippet)
            if isinstance(v, dict):
                return v
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                v = json.loads(cleaned)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass

    # 3. Thử tìm cặp ngoặc nhọn ngoài cùng { ... }
    first_brace = t.find("{")
    last_brace = t.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        snippet = t[first_brace : last_brace + 1]
        try:
            v = json.loads(snippet)
            if isinstance(v, dict):
                return v
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", snippet)
            try:
                v = json.loads(cleaned)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass

    # 4. Thử phục hồi JSON bị cắt ngắn (do cạn token hoặc ngắt kết nối giữa chừng)
    if first_brace != -1:
        last_idx = len(t)
        for _ in range(10):
            rbrace = t.rfind("}", 0, last_idx)
            if rbrace == -1 or rbrace <= first_brace:
                break
            sub = t[first_brace : rbrace + 1].strip()
            open_brackets = sub.count("[") - sub.count("]")
            open_braces = sub.count("{") - sub.count("}")
            suffix = ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
            attempt = sub + suffix
            cleaned = re.sub(r",\s*([}\]])", r"\1", attempt)
            try:
                val = json.loads(cleaned)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
            last_idx = rbrace

    return None
