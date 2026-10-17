"""Nhận video: kiểm `videoUrl` (chống SSRF), tải streaming có giới hạn, kiểm chữ ký tệp (SEC-7, BR-AI-STT-03)."""
import asyncio
import ipaddress
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlsplit

import httpx
from fastapi import UploadFile

from app.core.config import Settings
from app.core.errors import file_too_large, media_download_failed, media_invalid, validation_error

CHUNK = 1024 * 1024
MAX_REDIRECTS = 3
# MP4/MOV: 4 byte kích thước hộp đầu, rồi tên hộp. `ftyp` là phổ biến nhất; MOV cũ có thể mở đầu bằng moov/mdat/wide/free/skip/pnot.
_BOX_TYPES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


def validate_video_url(url: str, allowed_hosts: list[str]) -> str:
    """Chỉ nhận https tới đúng host được phép (Cloudinary). Không IP, không userinfo, không cổng lạ."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise validation_error("videoUrl phải là địa chỉ https hợp lệ.")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise validation_error("videoUrl không được chứa thông tin đăng nhập hoặc cổng khác 443.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise validation_error("videoUrl không được là địa chỉ IP.")
    if host not in allowed_hosts:
        raise validation_error("videoUrl không thuộc máy chủ được phép.")
    return parts.geturl()


def has_video_signature(head: bytes) -> bool:
    return len(head) >= 12 and head[4:8] in _BOX_TYPES


def check_signature(path: Path) -> None:
    with path.open("rb") as f:
        head = f.read(16)
    if not has_video_signature(head):
        raise media_invalid("Tệp không phải video MP4/MOV hợp lệ.")


@contextmanager
def workspace() -> Iterator[Path]:
    """Thư mục tạm cho một tác vụ; luôn bị xóa khi xong, kể cả khi lỗi (không để lại tệp dở dang)."""
    with tempfile.TemporaryDirectory(prefix="stt-") as d:
        yield Path(d)


async def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    total = 0
    with dest.open("wb") as out:
        while chunk := await upload.read(CHUNK):
            total += len(chunk)
            if total > max_bytes:
                raise file_too_large(f"Video vượt quá {max_bytes // (1024 * 1024)} MB.")
            out.write(chunk)
    if total == 0:
        raise media_invalid("Tệp rỗng.")


async def download_video(url: str, dest: Path, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
    """Tải `url` về `dest`, dừng ngay khi vượt giới hạn dung lượng. Tự theo chuyển hướng nhưng kiểm lại từng bước."""
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    hosts = settings.stt_allowed_hosts
    current = validate_video_url(url, hosts)
    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=30.0), follow_redirects=False)

    async def run() -> None:
        nonlocal current
        for _ in range(MAX_REDIRECTS + 1):
            async with http.stream("GET", current) as resp:
                if resp.is_redirect:
                    current = validate_video_url(urljoin(current, resp.headers.get("location", "")), hosts)
                    continue
                if resp.status_code != 200:
                    raise media_download_failed(f"Máy chủ video trả HTTP {resp.status_code}.")
                declared = int(resp.headers.get("content-length") or 0)
                if declared > max_bytes:
                    raise file_too_large(f"Video vượt quá {settings.stt_max_upload_mb} MB.")
                total = 0
                with dest.open("wb") as out:
                    async for chunk in resp.aiter_bytes(CHUNK):
                        total += len(chunk)
                        if total > max_bytes:
                            raise file_too_large(f"Video vượt quá {settings.stt_max_upload_mb} MB.")
                        out.write(chunk)
                if total == 0:
                    raise media_download_failed("Video tải về rỗng.")
                return
        raise media_download_failed("Quá nhiều lần chuyển hướng.")

    try:
        await asyncio.wait_for(run(), timeout=settings.stt_download_timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise media_download_failed("Tải video quá thời gian.") from exc
    except httpx.HTTPError as exc:
        raise media_download_failed(f"Không tải được video: {type(exc).__name__}.") from exc
    finally:
        if own:
            await http.aclose()
