"""Lỗi trả cho Backend: `{ "errorCode": "...", "message": "..." }` (Sprint Plan Phụ lục B.3)."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class AiServiceError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def unauthorized() -> AiServiceError:
    return AiServiceError("UNAUTHORIZED", "Thiếu hoặc sai X-Internal-Api-Key.", 401)


def validation_error(message: str) -> AiServiceError:
    return AiServiceError("VALIDATION_ERROR", message, 400)


def invalid_source_type(value: str) -> AiServiceError:
    return AiServiceError("INVALID_SOURCE_TYPE", f"sourceType không hợp lệ: {value}", 400)


def unsafe_content() -> AiServiceError:
    return AiServiceError("UNSAFE_CONTENT", "Câu hỏi vi phạm tiêu chuẩn an toàn.", 400)


def gemini_error(message: str = "Gemini/Vertex AI trả lỗi.") -> AiServiceError:
    return AiServiceError("GEMINI_ERROR", message, 502)


def qdrant_unavailable(message: str = "Qdrant không phản hồi hoặc quá thời gian.") -> AiServiceError:
    return AiServiceError("QDRANT_UNAVAILABLE", message, 503)


def ai_timeout(message: str = "Quá thời gian chờ AI.") -> AiServiceError:
    return AiServiceError("AI_TIMEOUT", message, 504)


def media_invalid(message: str) -> AiServiceError:
    return AiServiceError("MEDIA_INVALID", message, 400)


def image_invalid(message: str) -> AiServiceError:
    return AiServiceError("IMAGE_INVALID", message, 400)


def duration_invalid(message: str) -> AiServiceError:
    return AiServiceError("DURATION_INVALID", message, 400)


def file_too_large(message: str) -> AiServiceError:
    return AiServiceError("FILE_TOO_LARGE", message, 413)


def no_speech() -> AiServiceError:
    return AiServiceError("NO_SPEECH", "Không nhận ra lời thoại trong video.", 422)


def media_download_failed(message: str = "Không tải được video từ videoUrl.") -> AiServiceError:
    return AiServiceError("MEDIA_DOWNLOAD_FAILED", message, 502)


def not_a_recipe() -> AiServiceError:
    return AiServiceError("NOT_A_RECIPE", "Lời thoại trong video không phải công thức nấu ăn.", 422)


def stt_error(message: str = "Whisper hoặc ffmpeg xử lý lỗi.") -> AiServiceError:
    return AiServiceError("STT_ERROR", message, 502)


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(AiServiceError)
    async def _ai_error(_: Request, exc: AiServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"errorCode": exc.error_code, "message": exc.message})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return JSONResponse(
            status_code=400,
            content={"errorCode": "VALIDATION_ERROR", "message": f"Dữ liệu đầu vào không hợp lệ: {loc} — {first.get('msg', '')}"},
        )
