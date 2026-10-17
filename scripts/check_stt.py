"""G0/G5: chạy thử STT trên tệp thật, in thời gian, RAM đỉnh và bản phiên âm của từng model.

  python -m scripts.check_stt <video> [--models small medium] [--save out.json]

Không gọi Gemini. Dùng để chọn WHISPER_MODEL (thời gian, RAM, độ chính xác tiếng Việt).
"""
import argparse
import asyncio
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

from app.core.config import Settings
from app.core.gemini_client import GeminiClient
from app.services.stt import audio
from app.services.stt.recipe_extractor import extract_recipe
from app.services.stt.whisper import WhisperEngine


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--models", nargs="+", default=["small"])
    ap.add_argument("--save", type=Path)
    ap.add_argument("--threads", type=int, default=0, help="giả lập số vCPU của Cloud Run (vd. 2)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--recipe", action="store_true", help="gọi Gemini (Vertex) trích công thức từ phiên âm")
    a = ap.parse_args()
    base = Settings()
    info = await audio.probe(a.video, base)
    print(f"{a.video.name}: {info.duration:.1f} giây, {info.sample_rate} Hz")
    out = []
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        t = time.perf_counter()
        await audio.extract_wav(a.video, wav, base)
        print(f"  ffmpeg → WAV: {time.perf_counter() - t:.1f} giây")
        for m in a.models:
            eng = WhisperEngine(base.model_copy(update={"whisper_model": m, "whisper_cpu_threads": a.threads, "whisper_beam_size": a.beam}))
            t = time.perf_counter()
            eng.load()
            load_s = time.perf_counter() - t
            t = time.perf_counter()
            r = await eng.transcribe(wav)
            dt = time.perf_counter() - t
            print(f"  [{m} · {a.threads or 'auto'} luồng · beam {a.beam}] nạp {load_s:.1f}s · phiên âm {dt:.1f}s (x{info.duration / dt:.1f} so với thời lượng) "
                  f"· RAM đỉnh {peak_rss_mb():.0f} MB · {len(r.text.split())} từ")
            print(f"     {r.text}")
            recipe = None
            if a.recipe:
                t = time.perf_counter()
                ex = await extract_recipe(GeminiClient(base), base, r.text)
                recipe = ex.recipe.model_dump(by_alias=True)
                print(f"  [Gemini] {time.perf_counter() - t:.1f}s · {ex.generation.prompt_tokens}+{ex.generation.completion_tokens} token")
                print("     " + json.dumps(recipe, ensure_ascii=False))
            out.append({"model": m, "loadSeconds": round(load_s, 1), "seconds": round(dt, 1),
                        "peakRssMb": round(peak_rss_mb()), "text": r.text, "recipe": recipe})
    if a.save:
        a.save.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
