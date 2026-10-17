import httpx
import pytest

from app.core.errors import AiServiceError
from app.services.stt import media
from tests.stt_helpers import stt_settings

HOSTS = ["res.cloudinary.com"]
OK = "https://res.cloudinary.com/demo/video/upload/v1/a.mp4"
MP4_HEAD = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20


@pytest.mark.parametrize("url", [
    "http://res.cloudinary.com/a.mp4",                 # không phải https
    "https://evil.example.com/a.mp4",                  # host lạ
    "https://res.cloudinary.com.evil.com/a.mp4",       # giả mạo hậu tố
    "https://169.254.169.254/latest/meta-data",        # IP metadata
    "https://127.0.0.1/a.mp4",
    "https://user:pw@res.cloudinary.com/a.mp4",        # userinfo
    "https://res.cloudinary.com:8443/a.mp4",           # cổng lạ
    "file:///etc/passwd",
    "ftp://res.cloudinary.com/a.mp4",
    "not a url",
])
def test_validate_video_url_rejects(url):
    with pytest.raises(AiServiceError) as e:
        media.validate_video_url(url, HOSTS)
    assert e.value.error_code == "VALIDATION_ERROR" and e.value.status_code == 400


def test_validate_video_url_accepts_allowed_host_case_insensitive():
    assert media.validate_video_url("https://RES.cloudinary.com/x.mp4", HOSTS).endswith("/x.mp4")
    assert media.validate_video_url("https://res.cloudinary.com:443/x.mp4", HOSTS)


@pytest.mark.parametrize("head,ok", [
    (MP4_HEAD, True),
    (b"\x00\x00\x00\x08wide\x00\x00\x00\x00\x00\x00\x00\x00", True),    # MOV cũ
    (b"%PDF-1.7\n1 0 obj\n<<>>", False),
    (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", False),
    (b"short", False),
    (b"", False),
])
def test_has_video_signature(head, ok):
    assert media.has_video_signature(head) is ok


def test_check_signature_rejects_wrong_content(tmp_path):
    f = tmp_path / "x.mp4"
    f.write_bytes(b"MZ" + b"\x00" * 64)
    with pytest.raises(AiServiceError) as e:
        media.check_signature(f)
    assert e.value.error_code == "MEDIA_INVALID"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_download_ok(tmp_path):
    body = MP4_HEAD + b"x" * 1000
    async with _client(lambda r: httpx.Response(200, content=body)) as c:
        dest = tmp_path / "v.bin"
        await media.download_video(OK, dest, stt_settings(), client=c)
    assert dest.read_bytes() == body


async def test_download_404_is_media_download_failed(tmp_path):
    async with _client(lambda r: httpx.Response(404)) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", stt_settings(), client=c)
    assert e.value.error_code == "MEDIA_DOWNLOAD_FAILED" and e.value.status_code == 502


async def test_download_rejects_declared_oversize(tmp_path):
    s = stt_settings(stt_max_upload_mb=1)
    async with _client(lambda r: httpx.Response(200, content=b"x", headers={"content-length": str(2 * 1024 * 1024)})) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", s, client=c)
    assert e.value.error_code == "FILE_TOO_LARGE" and e.value.status_code == 413


async def test_download_stops_when_stream_exceeds_limit(tmp_path):
    s = stt_settings(stt_max_upload_mb=1)
    big = b"x" * (3 * 1024 * 1024)
    async with _client(lambda r: httpx.Response(200, content=big)) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", s, client=c)
    assert e.value.error_code == "FILE_TOO_LARGE"


async def test_download_limit_boundary_exact_size_passes(tmp_path):
    s = stt_settings(stt_max_upload_mb=1)
    body = MP4_HEAD + b"x" * (1024 * 1024 - len(MP4_HEAD))     # đúng 1 MB
    async with _client(lambda r: httpx.Response(200, content=body)) as c:
        await media.download_video(OK, tmp_path / "v.bin", s, client=c)
    assert (tmp_path / "v.bin").stat().st_size == 1024 * 1024


async def test_download_follows_redirect_only_to_allowed_host(tmp_path):
    def evil(r):
        return httpx.Response(302, headers={"location": "https://169.254.169.254/x"})
    async with _client(evil) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", stt_settings(), client=c)
    assert e.value.error_code == "VALIDATION_ERROR"

    calls = []

    def good(r):
        calls.append(str(r.url))
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "/other/b.mp4"})
        return httpx.Response(200, content=MP4_HEAD)
    async with _client(good) as c:
        await media.download_video(OK, tmp_path / "w.bin", stt_settings(), client=c)
    assert calls[1].endswith("/other/b.mp4")


async def test_download_rejects_redirect_loop(tmp_path):
    async with _client(lambda r: httpx.Response(302, headers={"location": OK})) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", stt_settings(), client=c)
    assert e.value.error_code == "MEDIA_DOWNLOAD_FAILED"


async def test_download_network_error_is_media_download_failed(tmp_path):
    def boom(r):
        raise httpx.ConnectError("down")
    async with _client(boom) as c:
        with pytest.raises(AiServiceError) as e:
            await media.download_video(OK, tmp_path / "v.bin", stt_settings(), client=c)
    assert e.value.error_code == "MEDIA_DOWNLOAD_FAILED"


def test_workspace_is_removed_even_on_error():
    seen = {}
    with pytest.raises(RuntimeError):
        with media.workspace() as w:
            (w / "a.bin").write_bytes(b"1")
            seen["w"] = w
            raise RuntimeError("boom")
    assert not seen["w"].exists()
