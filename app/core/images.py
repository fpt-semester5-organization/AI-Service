"""Nhận dạng ảnh bằng chữ ký tệp (magic bytes) và đọc kích thước từ tiêu đề JPG/PNG/WEBP, không cần Pillow.

Không tin Content-Type hay đuôi tệp do client gửi (BR-AI-CV-01, SEC). Chỉ đọc tiêu đề, không giải mã ảnh.
"""
import struct
from dataclasses import dataclass

PNG = "image/png"
JPEG = "image/jpeg"
WEBP = "image/webp"
ALLOWED_MIME = (JPEG, PNG, WEBP)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
# SOF0..SOF15 trừ DHT (C4), JPG (C8), DAC (CC): mang kích thước khung ảnh
_JPEG_SOF = {m for m in range(0xC0, 0xD0)} - {0xC4, 0xC8, 0xCC}


@dataclass(frozen=True)
class ImageInfo:
    mime_type: str
    width: int
    height: int


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:          # byte đệm
            i += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD9)):   # marker không có độ dài
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in _JPEG_SOF:
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker == 0xDA:          # bắt đầu dữ liệu quét mà chưa thấy SOF
            return None
        i += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    kind = data[12:16]
    if kind == b"VP8 ":
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            return None
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            return None
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if kind == b"VP8X" and len(data) >= 30:
        return (int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1)
    return None


def sniff_image(data: bytes) -> ImageInfo | None:
    """Trả (MIME, rộng, cao) nếu `data` là JPG/PNG/WEBP có tiêu đề đọc được; ngược lại None."""
    size: tuple[int, int] | None
    if data.startswith(_PNG_SIG):
        mime, size = PNG, _png_size(data)
    elif data.startswith(b"\xff\xd8\xff"):
        mime, size = JPEG, _jpeg_size(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime, size = WEBP, _webp_size(data)
    else:
        return None
    if size is None or size[0] <= 0 or size[1] <= 0:
        return None
    return ImageInfo(mime, size[0], size[1])
