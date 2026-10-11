"""Tạo ảnh PNG/JPEG/WEBP tối thiểu theo kích thước cho test (không commit tệp ảnh, không cần Pillow)."""
import struct
import zlib


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    row = b"\x00" + b"\x00" * ((width + 7) // 8)       # lọc 0 + điểm ảnh 1 bit
    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b""))


def jpeg(width: int, height: int) -> bytes:
    """JPEG chỉ có tiêu đề hợp lệ (APP0 + SOF0 + EOI): đủ để đọc kích thước, không giải mã được."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def webp(width: int, height: int, kind: str = "VP8X") -> bytes:
    if kind == "VP8X":
        body = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    elif kind == "VP8L":
        bits = (width - 1) | ((height - 1) << 14)
        body = b"VP8L" + struct.pack("<I", 5) + b"\x2f" + struct.pack("<I", bits)
    else:  # "VP8 "
        body = b"VP8 " + struct.pack("<I", 10) + b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height)
    body = body + b"\x00" * 4
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body
