"""G0: kiểm Vertex AI ở region đã cấu hình — model sinh văn bản, embedding (768 chiều), count_tokens.

Chạy:  python -m scripts.check_vertex        (đọc .env; cần GOOGLE_CLOUD_PROJECT và thông tin xác thực)
Thoát 0 nếu cả ba đạt; 1 nếu có mục lỗi (in lý do để xử lý: bật API, cấp quyền, hoặc model chưa có ở region).
"""
import asyncio
import sys
import time

from app.core.config import Settings
from app.core.errors import AiServiceError
from app.core.gemini_client import GeminiClient, Message


async def main() -> int:
    s = Settings()
    print(f"Project={s.google_cloud_project or '(chưa đặt)'} Location={s.google_cloud_location} "
          f"Model={s.gemini_model} Embedding={s.embedding_model}/{s.embedding_dimensions}")
    if not s.google_cloud_project:
        print("THIẾU GOOGLE_CLOUD_PROJECT (đặt trong .env).")
        return 1
    c = GeminiClient(s)
    ok = True

    async def step(name, coro):
        nonlocal ok
        t = time.perf_counter()
        try:
            out = await coro
            print(f"OK    {name} ({int((time.perf_counter() - t) * 1000)} ms): {out}")
        except AiServiceError as e:
            ok = False
            cause = repr(e.__cause__)[:300] if e.__cause__ else ""
            print(f"LỖI   {name}: {e.error_code} {e.message} {cause}")

    async def gen():
        g = await c.generate("Trả lời đúng một từ.", [Message("user", "Nói: ok")], temperature=0, max_output_tokens=16)
        return f"'{g.text.strip()[:30]}' prompt={g.prompt_tokens} out={g.completion_tokens}"

    async def emb():
        v = await c.embed("đậu phụ giàu đạm thực vật", task="RETRIEVAL_DOCUMENT")
        q = await c.embed("đạm thực vật", task="RETRIEVAL_QUERY")
        return f"{len(v)} chiều, {len(q)} chiều (query)"

    async def cnt():
        return f"{await c.count_tokens('Người ăn chay cần khoảng 0,8 gam đạm mỗi kg cân nặng.')} token"

    await step("generate_content", gen())
    await step("embed_content", emb())
    await step("count_tokens", cnt())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
