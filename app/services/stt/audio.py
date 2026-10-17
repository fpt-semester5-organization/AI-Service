"""ffprobe (thời lượng, tần số lấy mẫu) và ffmpeg (trích WAV 16 kHz mono 16-bit) — BR-AI-STT-01."""
import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.errors import duration_invalid, media_invalid, stt_error


@dataclass
class MediaInfo:
    duration: float
    sample_rate: int


async def _run(cmd: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    if shutil.which(cmd[0]) is None:
        raise stt_error(f"Thiếu {cmd[0]} trong môi trường chạy.")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise stt_error(f"{cmd[0]} quá thời gian.") from exc
    return proc.returncode or 0, out, err


async def probe(path: Path, settings: Settings) -> MediaInfo:
    code, out, _ = await _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        settings.stt_ffmpeg_timeout_seconds)
    if code != 0:
        raise media_invalid("Không đọc được video (tệp hỏng hoặc sai định dạng).")
    try:
        data = json.loads(out)
        audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if audio is None:
            raise media_invalid("Video không có luồng âm thanh.")
        duration = float(data.get("format", {}).get("duration") or audio.get("duration") or 0)
        sample_rate = int(audio.get("sample_rate") or 0)
    except (ValueError, TypeError) as exc:
        raise media_invalid("Không đọc được thông tin video.") from exc
    return MediaInfo(duration=duration, sample_rate=sample_rate)


def check_limits(info: MediaInfo, settings: Settings) -> None:
    if info.duration < settings.stt_min_seconds or info.duration > settings.stt_max_seconds:
        raise duration_invalid(
            f"Thời lượng {info.duration:.1f} giây nằm ngoài khoảng "
            f"{settings.stt_min_seconds:.0f}–{settings.stt_max_seconds:.0f} giây.")
    if info.sample_rate < settings.stt_min_sample_rate:
        raise media_invalid(
            f"Tần số lấy mẫu âm thanh {info.sample_rate} Hz thấp hơn {settings.stt_min_sample_rate} Hz.")


async def extract_wav(src: Path, dst: Path, settings: Settings) -> None:
    code, _, err = await _run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(dst)], settings.stt_ffmpeg_timeout_seconds)
    if code != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise stt_error(f"ffmpeg không trích được âm thanh: {err.decode('utf-8', 'replace')[:200]}")
