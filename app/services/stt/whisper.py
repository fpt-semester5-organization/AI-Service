"""faster-whisper: nạp model một lần, phiên âm trong luồng riêng, mỗi instance chỉ 1 tác vụ (SCA-2, EFF-3)."""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings
from app.core.errors import stt_error

log = logging.getLogger("veggielife.whisper")


@dataclass
class Transcription:
    text: str
    language: str
    duration: float


def _default_factory(s: Settings) -> Any:
    from faster_whisper import WhisperModel  # nhập muộn: ai-service không cài faster-whisper
    return WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type,
                        cpu_threads=s.whisper_cpu_threads)


class WhisperEngine:
    def __init__(self, settings: Settings, model_factory: Callable[[Settings], Any] | None = None):
        self._s = settings
        self._factory = model_factory or _default_factory
        self._model: Any = None
        self._sem = asyncio.Semaphore(max(1, settings.stt_concurrency))

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        return f"faster-whisper-{self._s.whisper_model}-{self._s.whisper_compute_type}"

    def load(self) -> None:
        if self._model is None:
            log.info("Nạp Whisper %s (%s, %s)", self._s.whisper_model, self._s.whisper_device,
                     self._s.whisper_compute_type)
            self._model = self._factory(self._s)

    def _transcribe_sync(self, wav: Path) -> Transcription:
        self.load()
        segments, info = self._model.transcribe(
            str(wav), language=self._s.whisper_language or None, beam_size=self._s.whisper_beam_size, vad_filter=True,
            condition_on_previous_text=False)  # tránh lặp câu khi mô hình "kẹt"
        text = " ".join(t for seg in segments if (t := seg.text.strip()))
        return Transcription(text=text, language=info.language, duration=float(info.duration))

    async def transcribe(self, wav: Path) -> Transcription:
        async with self._sem:
            try:
                return await asyncio.to_thread(self._transcribe_sync, wav)
            except Exception as exc:  # noqa: BLE001 — mọi lỗi Whisper đổi sang STT_ERROR
                log.exception("Whisper lỗi")
                raise stt_error(f"Whisper lỗi: {type(exc).__name__}") from exc
