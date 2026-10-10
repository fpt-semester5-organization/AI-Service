"""Chỉ Backend được gọi: kiểm tiêu đề `X-Internal-Api-Key` (SEC-11). Chưa cấu hình khóa thì từ chối mọi yêu cầu."""
import hmac

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from app.core.errors import unauthorized

# scheme_name để Swagger UI hiện nút Authorize (chỉ bật khi ENABLE_DOCS=true)
_key_header = APIKeyHeader(name="X-Internal-Api-Key", auto_error=False, scheme_name="InternalApiKey")


async def require_internal_key(request: Request, x_internal_api_key: str | None = Depends(_key_header)) -> None:
    expected = request.app.state.settings.internal_api_key
    if not expected or not x_internal_api_key:
        raise unauthorized()
    if not hmac.compare_digest(x_internal_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise unauthorized()
