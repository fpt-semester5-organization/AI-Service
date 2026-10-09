# veggielife-ai-service

Nơi **duy nhất** chứa nghiệp vụ AI của VeggieLife (D118). **Python 3.11 · FastAPI · Gemini 2.5 Flash (văn bản, thị giác, embedding) · Qdrant · faster-whisper · FFmpeg · Docker.**

> Repo hiện chỉ có **cây thư mục**. File `.gitkeep` chỉ giữ thư mục rỗng trong Git; khi thư mục đã có file thật thì xóa `.gitkeep` đi.
> Cập nhật theo bộ tài liệu chuẩn ngày 09/10/2026 (Decision Log tới D119). Đặc tả nằm trong `docs/` (bảng số tệp ở `docs/README.md`).

- **Chỉ backend .NET được gọi service này**, qua header `X-Internal-Api-Key`. Frontend không gọi trực tiếp.
- Service này là nơi duy nhất giữ khóa Gemini, truy cập Qdrant và dựng lời nhắc.
- Service **không** đọc SQL Server hay Redis nghiệp vụ. Mọi dữ liệu cần thiết (hồ sơ, dị ứng, lịch sử chat, công thức ứng viên đã lọc, nội dung đã duyệt) do Backend **gửi kèm trong yêu cầu**.
- Mọi phản hồi AI kèm `usage` (model, token, temperature, độ trễ) và `audit` (prompt, kết quả thô) để Backend ghi `AIQuotaLogs` và `AuditLogs`.

---

## 1. Hai bản triển khai cùng một repo (D119)

| Dịch vụ | Dockerfile | Chạy ở | Endpoint | Ghi chú |
|---|---|---|---|---|
| **ai-service** | `Dockerfile` | VPS, Docker Compose cùng Qdrant (cùng mạng nội bộ, không mở cổng ra Internet) | `/health`, `/chat/answer`, `/meal/generate`, `/moderation/check`, `/vision/detect-ingredients`, `/rag/*` | Backend gọi qua `AiService__BaseUrl` |
| **stt-service** | `Dockerfile.stt` | Google Cloud Run (`--min-instances 0 --memory 4Gi --cpu 2`) | `/health`, `/stt/transcribe`, `/stt/video-to-recipe` | Job nền, chịu được cold start; Backend gọi qua `SttService__BaseUrl` |

Hai dịch vụ dùng chung `core/` (cấu hình, kiểm `X-Internal-Api-Key`) và cùng một khóa nội bộ.

---

## 2. Cây thư mục

```
veggielife-ai-service/
├── README.md
├── docs/                     bản sao đặc tả (xem docs/README.md)
├── .github/workflows/        CI: ruff + pytest (service container Qdrant) + build 2 image; deploy ai-service lên VPS, stt-service lên Cloud Run
├── Dockerfile                ai-service (VPS)
├── Dockerfile.stt            stt-service (Cloud Run): cài ffmpeg, tải model Whisper khi build
├── app/
│   ├── main.py               (tạo khi code) tạo FastAPI app, include router theo bản triển khai
│   ├── core/                 config.py (pydantic-settings) · security.py (X-Internal-Api-Key) · gemini_client.py (lớp gọi Gemini duy nhất: timeout 10 giây, retry 2 lần)
│   ├── api/routers/          health · chat · meal · moderation · vision · rag · stt
│   ├── services/
│   │   ├── rag/              chunker · embedder · qdrant_store · retriever · indexer · nutrition_loader
│   │   ├── chat/             lọc an toàn, dựng ngữ cảnh [S1]..[S5], gọi Gemini, dựng sources
│   │   ├── meal/             xếp 21 bữa theo JSON Schema
│   │   ├── moderation/       chấm điểm vi phạm văn bản + ảnh
│   │   ├── vision/           ingredient_detector (ảnh tủ lạnh)
│   │   └── stt/              audio (FFmpeg → WAV 16 kHz mono) · whisper (faster-whisper int8) · recipe_extractor (Gemini)
│   └── schemas/              Pydantic request/response, alias camelCase khớp DTO bên .NET
├── knowledge/nutrition/      ≥ 15 tệp Markdown tài liệu dinh dưỡng chay (front-matter: id, title, topic, sources[], updated)
└── tests/                    pytest; tests/rag/golden.json (20 câu hỏi vàng + câu ngoài phạm vi)
```

---

## 3. Hợp đồng với Backend

Nguồn chuẩn là **Phụ lục B.3 của Sprint Plan** (`plan/16. Context implement/09. Sprint Plan.docx`); bảng dưới chỉ tóm tắt. JSON camelCase; lỗi `{ "errorCode": "...", "message": "..." }`.

| Method | Path | Dùng cho (FN) | Ra (200) |
|---|---|---|---|
| GET | `/health` | Health check | `{status, qdrant, gemini}` |
| POST | `/chat/answer` | Chat RAG (FN-CHAT-02) | `answer`, `sources`, `ragUsed`, `recommendedRecipeIds`, `usage`, `audit` |
| POST | `/meal/generate` | Thực đơn tuần (FN-MEAL-01) | `slots` (21 ô `dayOfWeek`, `mealType`, `recipeId`), `usage`, `audit` |
| POST | `/moderation/check` | Kiểm duyệt (FN-POST-03/04/07) | `violationScore`, `violationType`, `reasons`, `usage`, `audit` |
| POST | `/vision/detect-ingredients` | Ảnh tủ lạnh (FN-VIS-01) | `items[]` (`name`, `confidence`, `freshnessStatus`, `isAnimalProduct`) |
| POST | `/rag/upsert` | Nạp một nguồn (FN-CHAT-07) | `chunks`, `embeddingTokens` |
| POST | `/rag/delete` | Xóa mọi đoạn của một nguồn | `deleted` |
| POST | `/rag/reindex/nutrition` | Nạp lại `knowledge/nutrition/*.md` | `files`, `chunks`, `embeddingTokens` |
| GET | `/rag/status` | Trạng thái kho | `collection`, `totalPoints`, `bySource`, `lastIndexedAt`, `failed` |
| POST | `/stt/transcribe` | Video → phiên âm (stt-service) | `durationSeconds`, `language`, `sttModelVersion`, `rawTranscript` |
| POST | `/stt/video-to-recipe` | Video → công thức (FN-POST-10, stt-service) | `rawTranscript`, `recipe` (title, servings, ingredients, steps) |

**Mã lỗi HTTP:**

| HTTP | Khi nào |
|---|---|
| 400 | Đầu vào sai: `DURATION_INVALID`, `IMAGE_INVALID`, `UNSAFE_CONTENT`, `INVALID_SOURCE_TYPE` |
| 413 | Tệp quá lớn |
| 502 | Gemini/Whisper lỗi: `GEMINI_ERROR` |
| 503 | Qdrant không phản hồi hoặc quá 3 giây: `QDRANT_UNAVAILABLE` |
| 504 | Quá thời gian: `AI_TIMEOUT` |

**Khi lỗi:** service trả mã lỗi rõ ràng, **không** tự bỏ qua RAG. `ragUsed = false` chỉ dùng khi không có đoạn nào đạt ngưỡng điểm.

---

## 4. RAG và Qdrant

| Mục | Giá trị |
|---|---|
| Collection | `veggielife_knowledge`, tạo khi khởi động nếu chưa có; Cosine; payload index `sourceType`, `sourceId` |
| Embedding | `gemini-embedding-001`, 768 chiều; `RETRIEVAL_DOCUMENT` khi nạp, `RETRIEVAL_QUERY` khi hỏi (xác nhận tên model khi code) |
| Tìm kiếm | top-5, ngưỡng điểm 0,55, timeout 3 giây |
| Tách đoạn | ≈ 500 token, chồng lấn 50; tiêu đề ở đầu mỗi đoạn; nguồn ngắn = 1 đoạn |
| Id điểm | UUIDv5 của `"{sourceType}:{sourceId}:{chunkIndex}"`, namespace `6f1c0f62-4f0b-4c7e-9a52-1d6d9a0c2b11` (chạy lại không trùng) |
| `sourceType` hợp lệ | `RECIPE`, `POST`, `VIDEO_TRANSCRIPT`, `VIDEO_RECIPE`, `NUTRITION_DOC`, `E_NUMBER`, `ALLERGEN`, `INGREDIENT`; loại khác → 400 |
| `sourceId` | Luôn là chuỗi chữ số (khóa chính SQL; `NUTRITION_DOC` dùng front-matter `id`) để Backend đổi sang `sources[] = {type, id, title}` (D119) |
| Không bao giờ nạp | Hồ sơ sức khỏe, dị ứng, lịch sử chat, bản nháp, nội dung chưa duyệt |
| Chất lượng | Bộ 20 câu hỏi vàng: nguồn đúng trong top-5 ≥ 80%; câu ngoài phạm vi không có điểm vượt 0,55 |

---

## 5. Quy tắc

1. **Router mỏng:** router chỉ nhận/trả schema; logic nằm trong `services/`.
2. **Gemini chỉ qua `core/gemini_client.py`**: timeout 10 giây, retry 2 lần. Dùng JSON Schema và `temperature` ≤ 0,4 cho tác vụ có cấu trúc. Hình dạng JSON gửi Gemini theo Sprint Plan Phụ lục B.4.
3. **Không bịa nguồn:** `sources` dựng từ các đoạn tìm được thực sự đưa vào lời nhắc, không lấy từ văn bản AI sinh.
4. **Thực đơn và gợi ý món:** chỉ chọn `recipeId` trong danh sách ứng viên Backend gửi. `customInstructions` đặt sau lớp bảo vệ trường phái (D109). Backend kiểm tra lại 21 ô và calo ±10%.
5. **Kiểm duyệt:** chỉ trả điểm và lý do. Ngưỡng 85% và Red Alert do Backend quyết.
6. **Giới hạn input:**
   - ảnh JPG/PNG/WEBP, ≥ 640×480, ≤ 10 MB;
   - video MP4/MOV ≤ 100 MB, 10 giây–5 phút.
7. **Cấu hình:** mọi giá trị đọc từ `core/config.py`. Không commit `.env`.
8. **Nhánh / commit / PR:** như Backend, ví dụ `feature/VL-<số>-<mô-tả>`, commit `feat(rag): VL-50 add retriever [FN-CHAT-02]`.

---

## 6. Biến môi trường (`.env.example`)

- **Chung hai dịch vụ:** `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-2.5-flash`, `GEMINI_TIMEOUT_SECONDS=10`, `INTERNAL_API_KEY`, `MAX_UPLOAD_MB`, `PORT`.
- **stt-service:** `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_DEVICE`.
- **ai-service:** `QDRANT_URL`, `QDRANT_API_KEY` (rỗng ở local), `QDRANT_COLLECTION=veggielife_knowledge`, `EMBEDDING_MODEL=gemini-embedding-001`, `EMBEDDING_DIMENSIONS=768`, `RAG_TOP_K=5`, `RAG_MIN_SCORE=0.55`, `RAG_TIMEOUT_SECONDS=3`.

---

## 7. Chạy local và kiểm tra

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # stt cần ffmpeg: brew install ffmpeg
uvicorn app.main:app --reload --port 8080
```

Qdrant và ai-service chạy chung bằng `docker compose` ở repo Backend (`docker compose up -d`). stt-service chạy local trên cổng 8082 bằng `Dockerfile.stt`.

Trước khi báo xong:

```bash
ruff check .
pytest -q
```

---

## 8. Chuyển code từ prototype (chỉ phần STT)

Mở song song **Whisper Prototype** của nhóm; **không** copy cả thư mục prototype vào repo.

| File trong prototype | Chuyển vào |
|---|---|
| `main.py` (`/transcribe`, `/video-to-recipe`) | `api/routers/stt.py` + `services/stt/*` |
| `gemini_service.py` | `services/stt/recipe_extractor` (gọi qua `core/gemini_client.py`) |
| `Dockerfile`, `deploy.sh` | `Dockerfile.stt` và lệnh deploy Cloud Run |

**Không chuyển:**
- file `.env` có khóa thật;
- web UI test (`templates/`, `static/`);
- `/url-to-recipe` và `url_service.py` (TikTok/Reels là Phase 2);
- `/summarize-recipe`, `/video-chat`;
- HappyCow Prototype, `delivery_matcher.py`, `hybrid_pipeline.py` (thực đơn quán là ngoài MVP, D100).

**Ngoài MVP, không làm:** OCR nhãn thành phần (`/vision/ocr-label`), menu-sync quán, URL-to-recipe.
