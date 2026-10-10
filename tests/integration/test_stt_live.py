"""STT chạy thật: Whisper thật + Vertex AI thật (tốn credit, đánh dấu `live`, chạy tay).

  STT_LIVE_VIDEO=/đường/dẫn/video.mp4 pytest -m live tests/integration/test_stt_live.py -s

Video mẫu để ngoài repo. `videoUrl` là địa chỉ Cloudinary giả; vận chuyển HTTP được thay bằng tệp cục bộ.
"""
import os
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.stt_main import create_stt_app

pytestmark = pytest.mark.live
VIDEO = os.environ.get("STT_LIVE_VIDEO", "")
URL = "https://res.cloudinary.com/demo/video/upload/v1/sample.mp4"


@pytest.fixture(scope="module")
def client():
    if not VIDEO or not Path(VIDEO).is_file():
        pytest.skip("đặt STT_LIVE_VIDEO tới một video thật")
    data = Path(VIDEO).read_bytes()
    settings = Settings(internal_api_key="live-key")          # đọc .env: Vertex, WHISPER_*
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=data)))
    with TestClient(create_stt_app(settings, http_client=http)) as c:
        yield c


H = {"X-Internal-Api-Key": "live-key"}


def test_health_after_model_loaded(client):
    assert client.get("/health", headers=H).json()["whisper"] == "loaded"


def test_video_to_recipe_real(client):
    t = time.perf_counter()
    r = client.post("/stt/video-to-recipe", headers=H, json={"videoUrl": URL})
    took = time.perf_counter() - t
    assert r.status_code == 200, r.text
    d = r.json()
    print(f"\n[live] {d['durationSeconds']} giây video · {took:.1f} giây xử lý · {d['sttModelVersion']}")
    print(f"[live] {d['recipe']['title']} · {len(d['recipe']['ingredients'])} nguyên liệu · {len(d['recipe']['steps'])} bước")
    assert len(d["rawTranscript"].split()) > 50
    assert d["recipe"]["title"] and d["recipe"]["ingredients"] and d["recipe"]["steps"]
    assert all(set(i) == {"name", "quantity", "unit"} for i in d["recipe"]["ingredients"])
    assert d["usage"]["promptTokens"] > 0 and d["audit"]["rawResponse"]


def test_transcribe_upload_real(client):
    with open(VIDEO, "rb") as f:
        r = client.post("/stt/transcribe", headers=H, files={"file": ("v.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    assert r.json()["language"] == "vi" and r.json()["rawTranscript"]
