import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.services.stt.whisper import WhisperEngine
from app.stt_main import create_stt_app
from tests.fakes import FakeLLM
from tests.stt_helpers import HEADERS, FakeWhisperModel, make_video, needs_ffmpeg, stt_settings

URL = "https://res.cloudinary.com/demo/video/upload/v1/a.mp4"
RECIPE = {"title": "Canh nấm", "servings": 2, "ingredients": [{"name": "Nấm", "quantity": 200, "unit": "g"}],
          "steps": ["Nấu."]}


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    return {name: make_video(d / f"{name}.mp4", sec, sample_rate=sr)
            for name, sec, sr in [("ok", 12, 44100), ("short", 9, 44100), ("long", 301, 44100), ("low", 12, 8000)]} \
        if __import__("shutil").which("ffmpeg") else {}


def build(video_bytes: bytes | None = None, *, model: FakeWhisperModel | None = None, llm_reply: str | None = None,
          status: int = 200, **over):
    settings = stt_settings(**over)
    model = model or FakeWhisperModel()
    engine = WhisperEngine(settings, model_factory=lambda s: model)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(status, content=video_bytes or b"")))
    llm = FakeLLM(llm_reply if llm_reply is not None else json.dumps(RECIPE, ensure_ascii=False))
    app = create_stt_app(settings, llm=llm, engine=engine, preload=False, http_client=http)
    return app, model, llm


def post(client, body=None):
    return client.post("/stt/video-to-recipe", headers=HEADERS, json={"videoUrl": URL} if body is None else body)


def test_auth_required():
    app, _, _ = build()
    with TestClient(app) as c:
        assert c.post("/stt/video-to-recipe", json={"videoUrl": URL}).status_code == 401
        assert c.post("/stt/video-to-recipe", headers={"X-Internal-Api-Key": "sai"}, json={"videoUrl": URL}).status_code == 401
        assert c.get("/health").status_code == 401
        assert c.post("/stt/transcribe", files={"file": ("a.mp4", b"x")}).status_code == 401


def test_health_reports_whisper_state():
    app, _, _ = build()
    with TestClient(app) as c:
        r = c.get("/health", headers=HEADERS).json()
    assert r == {"status": "ok", "whisper": "not_loaded", "gemini": "configured"}


def test_validation_errors():
    app, _, _ = build()
    with TestClient(app) as c:
        assert post(c, {}).json()["errorCode"] == "VALIDATION_ERROR"
        r = post(c, {"videoUrl": "https://evil.example.com/a.mp4"})
        assert r.status_code == 400 and r.json()["errorCode"] == "VALIDATION_ERROR"


@needs_ffmpeg
def test_video_to_recipe_happy_path(videos):
    app, model, llm = build(videos["ok"].read_bytes())
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["rawTranscript"] == "Hôm nay mình nấu canh nấm"
    assert d["sttModelVersion"] == "faster-whisper-small-int8"
    assert 11 <= d["durationSeconds"] <= 13
    assert d["recipe"] == {"title": "Canh nấm", "servings": 2,
                           "ingredients": [{"name": "Nấm", "quantity": 200.0, "unit": "g"}], "steps": ["Nấu."]}
    assert d["usage"]["model"] == "fake-model" and "Hôm nay mình nấu" in d["audit"]["prompt"]
    kw = model.calls[0]
    assert kw["language"] == "vi" and kw["vad_filter"] is True and kw["path"].endswith("audio.wav")


@needs_ffmpeg
@pytest.mark.parametrize("name,status,code", [
    ("short", 400, "DURATION_INVALID"), ("long", 400, "DURATION_INVALID"), ("low", 400, "MEDIA_INVALID")])
def test_video_rules_rejected(videos, name, status, code):
    app, model, _ = build(videos[name].read_bytes())
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == status and r.json()["errorCode"] == code
    assert model.calls == []                                           # không tốn công chạy Whisper


def test_non_video_content_is_media_invalid():
    app, model, _ = build(b"%PDF-1.7 not a video at all, really not")
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 400 and r.json()["errorCode"] == "MEDIA_INVALID" and model.calls == []


def test_download_failure_maps_to_502():
    app, _, _ = build(b"", status=404)
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 502 and r.json()["errorCode"] == "MEDIA_DOWNLOAD_FAILED"


@needs_ffmpeg
def test_no_speech(videos):
    app, _, _ = build(videos["ok"].read_bytes(), model=FakeWhisperModel(text=""))
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 422 and r.json()["errorCode"] == "NO_SPEECH"


@needs_ffmpeg
def test_whisper_failure_maps_to_stt_error(videos):
    app, _, _ = build(videos["ok"].read_bytes(), model=FakeWhisperModel(fail=RuntimeError("oom")))
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 502 and r.json()["errorCode"] == "STT_ERROR"


@needs_ffmpeg
def test_transcript_without_recipe(videos):
    empty = json.dumps({"title": "", "servings": None, "ingredients": [], "steps": []})
    app, _, _ = build(videos["ok"].read_bytes(), llm_reply=empty)
    with TestClient(app) as c:
        r = post(c)
    assert r.status_code == 422 and r.json()["errorCode"] == "NOT_A_RECIPE"


@needs_ffmpeg
def test_transcribe_upload_and_size_limit(videos):
    app, _, llm = build()
    with TestClient(app) as c:
        r = c.post("/stt/transcribe", headers=HEADERS, files={"file": ("a.mp4", videos["ok"].read_bytes(), "video/mp4")})
        assert r.status_code == 200
        assert r.json() == {"durationSeconds": r.json()["durationSeconds"], "language": "vi",
                            "sttModelVersion": "faster-whisper-small-int8", "rawTranscript": "Hôm nay mình nấu canh nấm"}
        assert llm.generate_calls == []                                 # /stt/transcribe không gọi Gemini
    app2, _, _ = build(stt_max_upload_mb=1)
    with TestClient(app2) as c:
        big = b"\x00\x00\x00\x18ftypmp42" + b"x" * (2 * 1024 * 1024)
        r = c.post("/stt/transcribe", headers=HEADERS, files={"file": ("a.mp4", big, "video/mp4")})
    assert r.status_code == 413 and r.json()["errorCode"] == "FILE_TOO_LARGE"


def test_temp_files_are_removed_after_request(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    app, _, _ = build(b"%PDF not video")
    with TestClient(app) as c:
        post(c)
    assert list(tmp_path.iterdir()) == []


def test_concurrency_is_serialized():
    import asyncio
    settings = stt_settings()
    active, peak = 0, 0

    class Slow(FakeWhisperModel):
        def transcribe(self, path, **kw):
            nonlocal active, peak
            import time
            active += 1
            peak = max(peak, active)
            time.sleep(0.05)
            active -= 1
            return super().transcribe(path, **kw)

    engine = WhisperEngine(settings, model_factory=lambda s: Slow())

    async def run():
        await asyncio.gather(*(engine.transcribe(__import__("pathlib").Path("x.wav")) for _ in range(4)))
    asyncio.run(run())
    assert peak == 1


def test_warmup_runs_a_short_silent_transcription_and_tolerates_nothing_else():
    np = pytest.importorskip("numpy")
    model = FakeWhisperModel(text="")
    engine = WhisperEngine(stt_settings(), model_factory=lambda s: model)
    engine.warmup()
    assert engine.loaded
    audio = model.calls[0]["path"]
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32 and not audio.any()
    assert model.calls[0]["beam_size"] == 1


def test_startup_survives_failing_warmup(monkeypatch):
    pytest.importorskip("numpy")
    model = FakeWhisperModel(fail=RuntimeError("không có tính toán"))
    engine = WhisperEngine(stt_settings(), model_factory=lambda s: model)
    app = create_stt_app(stt_settings(), llm=FakeLLM(), engine=engine, preload=True)
    with TestClient(app) as c:                       # lifespan chạy load + warmup lỗi nhưng không sập
        assert c.get("/health", headers=HEADERS).json()["whisper"] == "loaded"
