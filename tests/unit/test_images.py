import pytest

from app.core.images import JPEG, PNG, WEBP, sniff_image
from tests.image_helpers import jpeg, png, webp


@pytest.mark.parametrize("data,mime", [
    (png(640, 480), PNG), (jpeg(1280, 720), JPEG),
    (webp(800, 600, "VP8X"), WEBP), (webp(800, 600, "VP8L"), WEBP), (webp(800, 600, "VP8 "), WEBP)])
def test_sniff_reads_mime_and_size(data, mime):
    info = sniff_image(data)
    assert info is not None and info.mime_type == mime
    assert (info.width, info.height) in {(640, 480), (1280, 720), (800, 600)}


@pytest.mark.parametrize("data", [
    b"", b"GIF89a" + b"\x00" * 40, b"%PDF-1.4" + b"\x00" * 40, b"\x89PNG\r\n\x1a\n" + b"\x00" * 4,
    b"\xff\xd8\xff", b"RIFF\x00\x00\x00\x00WEBP", b"<html>" * 10])
def test_sniff_rejects_other_or_truncated(data):
    assert sniff_image(data) is None


def test_sniff_ignores_extension_claims():
    # nội dung quyết định, không phải tên/MIME client khai: PNG giả dạng .jpg vẫn là PNG
    assert sniff_image(png(10, 10)).mime_type == PNG
