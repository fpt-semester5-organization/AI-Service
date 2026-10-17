"""Điều phối STT (FN-POST-10): kiểm tệp → ffprobe → WAV 16 kHz → Whisper → (Gemini). Luôn dọn tệp tạm."""
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import UploadFile

from app.core.config import Settings
from app.core.errors import no_speech
from app.core.gemini_client import LLMClient
from app.schemas.stt import Audit, TranscribeResponse, Usage, VideoToRecipeResponse
from app.services.stt import audio, media
from app.services.stt.recipe_extractor import extract_recipe
from app.services.stt.whisper import Transcription, WhisperEngine

log = logging.getLogger("veggielife.stt")


@dataclass
class Prepared:
    transcription: Transcription
    duration_seconds: int


class SttPipeline:
    def __init__(self, settings: Settings, engine: WhisperEngine, llm: LLMClient,
                 http_client: httpx.AsyncClient | None = None):
        self._s = settings
        self._engine = engine
        self._llm = llm
        self._http = http_client

    async def _run(self, video: Path, work: Path) -> Prepared:
        media.check_signature(video)
        info = await audio.probe(video, self._s)
        audio.check_limits(info, self._s)
        wav = work / "audio.wav"
        await audio.extract_wav(video, wav, self._s)
        result = await self._engine.transcribe(wav)
        if not result.text.strip():
            raise no_speech()
        return Prepared(result, int(round(info.duration)))

    async def transcribe_upload(self, upload: UploadFile) -> TranscribeResponse:
        with media.workspace() as work:
            video = work / "input.bin"
            await media.save_upload(upload, video, self._s.stt_max_upload_mb * 1024 * 1024)
            p = await self._run(video, work)
        return TranscribeResponse(duration_seconds=p.duration_seconds, language=p.transcription.language,
                                  stt_model_version=self._engine.model_version, raw_transcript=p.transcription.text)

    async def video_to_recipe(self, video_url: str) -> VideoToRecipeResponse:
        with media.workspace() as work:
            video = work / "input.bin"
            await media.download_video(video_url, video, self._s, client=self._http)
            p = await self._run(video, work)
        ex = await extract_recipe(self._llm, self._s, p.transcription.text)
        g = ex.generation
        return VideoToRecipeResponse(
            duration_seconds=p.duration_seconds, stt_model_version=self._engine.model_version,
            raw_transcript=p.transcription.text, recipe=ex.recipe,
            usage=Usage(model=g.model, prompt_tokens=g.prompt_tokens, completion_tokens=g.completion_tokens,
                        latency_ms=g.latency_ms, temperature=g.temperature),
            audit=Audit(prompt=ex.prompt, raw_response=g.text))
