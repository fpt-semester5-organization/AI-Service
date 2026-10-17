import pytest

from app.core.errors import AiServiceError
from app.services.stt import audio
from tests.stt_helpers import make_video, needs_ffmpeg, stt_settings

S = stt_settings()


@pytest.mark.parametrize("seconds,ok", [(9.9, False), (10.0, True), (10.1, True), (299.9, True), (300.0, True), (300.1, False)])
def test_duration_boundaries(seconds, ok):
    info = audio.MediaInfo(duration=seconds, sample_rate=44100)
    if ok:
        audio.check_limits(info, S)
    else:
        with pytest.raises(AiServiceError) as e:
            audio.check_limits(info, S)
        assert e.value.error_code == "DURATION_INVALID"


@pytest.mark.parametrize("rate,ok", [(8000, False), (15999, False), (16000, True), (22050, True), (48000, True)])
def test_sample_rate_boundaries(rate, ok):
    info = audio.MediaInfo(duration=60, sample_rate=rate)
    if ok:
        audio.check_limits(info, S)
    else:
        with pytest.raises(AiServiceError) as e:
            audio.check_limits(info, S)
        assert e.value.error_code == "MEDIA_INVALID"


@needs_ffmpeg
async def test_probe_and_extract_real_video(tmp_path):
    v = make_video(tmp_path / "v.mp4", 12, sample_rate=44100)
    info = await audio.probe(v, S)
    assert 11.5 <= info.duration <= 12.6 and info.sample_rate == 44100
    wav = tmp_path / "a.wav"
    await audio.extract_wav(v, wav, S)
    assert wav.stat().st_size > 12 * 16000 * 2 * 0.9          # WAV 16 kHz mono 16-bit
    probed = await audio.probe(wav, S)
    assert probed.sample_rate == 16000


@needs_ffmpeg
async def test_probe_rejects_video_without_audio(tmp_path):
    v = make_video(tmp_path / "silent.mp4", 12, audio=False)
    with pytest.raises(AiServiceError) as e:
        await audio.probe(v, S)
    assert e.value.error_code == "MEDIA_INVALID"


@needs_ffmpeg
async def test_probe_rejects_corrupt_file(tmp_path):
    f = tmp_path / "bad.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 50)
    with pytest.raises(AiServiceError) as e:
        await audio.probe(f, S)
    assert e.value.error_code == "MEDIA_INVALID"


@needs_ffmpeg
async def test_low_sample_rate_video_is_rejected(tmp_path):
    v = make_video(tmp_path / "low.mp4", 12, sample_rate=8000)
    info = await audio.probe(v, S)
    with pytest.raises(AiServiceError) as e:
        audio.check_limits(info, S)
    assert e.value.error_code == "MEDIA_INVALID"
