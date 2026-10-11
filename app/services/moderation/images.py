"""Tải ảnh của bài đăng để kiểm duyệt: chỉ https tới host cho phép (chống SSRF), tải có giới hạn, kiểm chữ ký tệp."""
import ipaddress
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.config import Settings
from app.core.errors import file_too_large, image_invalid, media_download_failed, validation_error
from app.core.gemini_client import ImagePart
from app.core.images import sniff_image

MAX_REDIRECTS = 3
CHUNK = 64 * 1024


def validate_image_url(url: str, allowed_hosts: list[str]) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise validation_error("imageUrls phải là địa chỉ https hợp lệ.")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise validation_error("imageUrls không được chứa thông tin đăng nhập hoặc cổng khác 443.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise validation_error("imageUrls không được là địa chỉ IP.")
    if host not in allowed_hosts:
        raise validation_error("imageUrls không thuộc máy chủ được phép.")
    return parts.geturl()


async def _download(client: httpx.AsyncClient, url: str, settings: Settings) -> bytes:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    current = validate_image_url(url, settings.moderation_allowed_image_hosts)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            async with client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    current = validate_image_url(urljoin(current, resp.headers["location"]),
                                                 settings.moderation_allowed_image_hosts)
                    continue
                if resp.status_code != 200:
                    raise media_download_failed(f"Không tải được ảnh (HTTP {resp.status_code}).")
                buf = bytearray()
                async for chunk in resp.aiter_bytes(CHUNK):
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise file_too_large(f"Ảnh vượt quá {settings.max_upload_mb} MB.")
                return bytes(buf)
        except httpx.HTTPError as exc:
            raise media_download_failed("Không tải được ảnh từ imageUrls.") from exc
    raise media_download_failed("Ảnh chuyển hướng quá nhiều lần.")


async def fetch_images(urls: list[str], settings: Settings, client: httpx.AsyncClient | None = None) -> list[ImagePart]:
    """Tải từng ảnh; ảnh không phải JPG/PNG/WEBP hợp lệ → 400 IMAGE_INVALID (không bỏ qua im lặng)."""
    if len(urls) > settings.moderation_max_images:
        raise validation_error(f"Tối đa {settings.moderation_max_images} ảnh mỗi lần kiểm duyệt.")
    if not urls:
        return []
    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(settings.moderation_image_timeout_seconds),
                                       follow_redirects=False)
    try:
        parts: list[ImagePart] = []
        for url in urls:
            data = await _download(http, url, settings)
            info = sniff_image(data)
            if info is None:
                raise image_invalid("Ảnh trong imageUrls phải là JPG, PNG hoặc WEBP hợp lệ.")
            parts.append(ImagePart(data=data, mime_type=info.mime_type))
        return parts
    finally:
        if own:
            await http.aclose()
