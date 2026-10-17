"""Công cụ test STT: tạo video bằng ffmpeg thật (không commit video), Whisper giả."""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings

KEY = "test-key"
HEADERS = {"X-Internal-Api-Key": KEY}
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="cần ffmpeg và ffprobe")


def stt_settings(**over) -> Settings:
    base = dict(internal_api_key=KEY, google_cloud_project="test", stt_allowed_video_hosts="res.cloudinary.com",
                _env_file=None)
    base.update(over)
    return Settings(**base)


def make_video(path: Path, seconds: float, *, sample_rate: int = 44100, audio: bool = True) -> Path:
    """MP4 (H.264 + AAC) tổng hợp: hình đen và âm thuần 440 Hz."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=10:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd += ["-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


class FakeWhisperModel:
    """Giống faster_whisper.WhisperModel.transcribe: trả (segments, info)."""

    def __init__(self, text: str = "Hôm nay mình nấu canh nấm", fail: Exception | None = None):
        self.text = text
        self.fail = fail
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append({"path": path, **kwargs})
        segs = [SimpleNamespace(text=t) for t in self.text.split("|")] if self.text else []
        return iter(segs), SimpleNamespace(language="vi", duration=12.0)
