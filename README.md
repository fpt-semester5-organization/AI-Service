# veggielife-ai-service

Microservice AI của VeggieLife, xử lý các tác vụ media nặng. **Python 3.11 · FastAPI · faster-whisper · FFmpeg · Gemini 2.5 Flash · Docker · Google Cloud Run.**

> Repo hiện chỉ có **cây thư mục**. File `.gitkeep` chỉ giữ thư mục rỗng trong Git; khi thư mục đã có file thật thì xóa `.gitkeep` đi.
> Code ban đầu **chuyển từ prototype** trong repo tài liệu, không viết lại từ đầu (xem mục 3).

**Chỉ backend .NET được gọi service này**, qua header `X-Internal-Api-Key`. Frontend không gọi trực tiếp.
Các tác vụ AI dạng văn bản (chatbot, lập thực đơn, kiểm duyệt nội dung) do backend .NET tự gọi Gemini, **không** nằm ở đây.

---

## 1. Cây thư mục

```
veggielife-ai-service/
├── README.md
├── .github/workflows/        CI: lint + pytest; deploy Cloud Run
├── app/
│   ├── main.py               (tạo khi code) tạo FastAPI app, include routers, load model Whisper 1 lần khi khởi động
│   ├── core/                 config.py (đọc .env bằng pydantic-settings) · security.py (kiểm X-Internal-Api-Key)
│   ├── api/
│   │   └── routers/          health.py · stt.py · vision.py (mỗi file = 1 nhóm endpoint)
│   ├── services/
│   │   ├── stt/              audio (FFmpeg → WAV 16 kHz mono) · whisper (faster-whisper int8) · recipe_extractor (Gemini) · url_downloader (yt-dlp)
│   │   └── vision/           ingredient_detector (ảnh tủ lạnh) · label_ocr (ảnh nhãn → E-number), đều dùng Gemini Vision
│   └── schemas/              Pydantic request/response, alias camelCase để khớp JSON của backend
└── tests/                    pytest: test_health, test_security, test_stt, test_vision
```

Các file gốc khi khởi tạo (đặt ở root): `requirements.txt`, `Dockerfile`, `deploy.sh`, `.env.example`, `.gitignore`.

---

## 2. Endpoint (đề xuất, chốt cùng `AiServiceClient` bên backend)

| Method | Path | Dùng cho | Rule |
|---|---|---|---|
| GET | `/health` | Cloud Run health check | |
| POST | `/stt/transcribe` | Video → `Raw_Transcript` (lưu VideoTranscripts) | BR-AI-STT-* |
| POST | `/stt/video-to-recipe` | Video → `Extracted_Recipe_JSON` | BR-AI-STT-* |
| POST | `/stt/url-to-recipe` | Link TikTok/Reels | ⚠ chờ chốt có thuộc MVP không |
| POST | `/vision/detect-ingredients` | Ảnh tủ lạnh → nguyên liệu + confidence | D32 (Gemini Vision, không dùng YOLO), BR-AI-CV-01 |
| POST | `/vision/ocr-label` | Ảnh nhãn thành phần → mã E-number | BR-GUEST-07 |

> **Thực đơn quán và đối soát ShopeeFood/GrabFood: ngoài phạm vi MVP** (theo quyết định D100, chỉ hiển thị thông tin quán cơ bản bên backend .NET; bảng `VenueMenuItems` giữ trong Data Model nhưng không hiện thực).

---

## 3. Chuyển code từ prototype

Nhóm có prototype: **Whisper Prototype** (STT video → công thức). Xin Leader bản mới nhất, tải về máy rồi mở song song với repo này. **Không** copy cả thư mục prototype vào repo.

| Prototype | File trong prototype | Chuyển vào |
|---|---|---|
| Whisper | `main.py` (`/transcribe`, `/video-to-recipe`, `/url-to-recipe`) | `api/routers/stt.py` + `services/stt/*` |
| Whisper | `gemini_service.py` | `services/stt/recipe_extractor` |
| Whisper | `url_service.py` | `services/stt/url_downloader` |
| Whisper | `Dockerfile`, `deploy.sh` | root repo |

> **Thực đơn quán và đối soát ShopeeFood/GrabFood: ngoài phạm vi MVP** (không chuyển HappyCow Prototype, `delivery_matcher.py` hay `hybrid_pipeline.py`).

Prototype có file `.env` chứa API key thật: **không** copy file này, chỉ dựa vào `.env.example`.

**Không chuyển:**
- web UI test (`templates/`, `static/`);
- Prototype HappyCow / Thực đơn quán / ShopeeFood / GrabFood (đã loại khỏi phạm vi MVP theo quyết định D100);
- `/summarize-recipe`, `/video-chat`, trừ khi FR yêu cầu.

---

## 4. Quy tắc

1. **Router mỏng:** router chỉ nhận/trả schema; logic nằm trong `services/`.
2. **Cấu hình:** mọi giá trị đọc từ `core/config.py`, không gọi `os.environ` rải rác. Không commit `.env`.
3. **JSON:** schema trả về dùng camelCase, khớp DTO bên .NET.
4. **Giới hạn input:** giới hạn dung lượng upload. Ảnh ≤ 10 MB, định dạng JPG/PNG/WEBP, độ phân giải ≥ 640×480 (BR-AI-CV). Giới hạn video: chờ chốt.
5. **Báo lỗi cho backend:** lỗi của Gemini/Whisper trả mã lỗi rõ ràng, để backend hoàn lượt dùng cho Guest và ghi AIQuotaLogs.
6. **Nhánh / commit / PR:** theo Jira giống 2 repo kia, ví dụ `feature/VL-70-ocr-label`.

## 5. Khởi tạo và chạy local

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # cần ffmpeg: brew install ffmpeg
uvicorn app.main:app --reload --port 8080
```

`requirements.txt`: gộp từ prototype, bỏ `jinja2`, thêm `pydantic-settings`, `pytest`.
Biến môi trường:
- `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`, `WHISPER_DEVICE`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `INTERNAL_API_KEY`
- `MAX_UPLOAD_MB`
- `PORT`

Deploy lên Cloud Run với `--min-instances 0 --memory 4Gi --cpu 2` (giống prototype).

## 6. Đang chờ chốt

- **Ngưỡng Vision:** giá trị 3 ngưỡng confidence (BR-AI-CV-01).
- **Video:** giới hạn dung lượng và thời lượng.
- **URL-to-recipe:** có thuộc MVP không.
- **Lượt AI của Guest:** 05 lượt dùng chung hay tách theo loại.

Danh sách đầy đủ các mục còn thiếu: hỏi Leader (file tổng hợp "Dữ liệu còn thiếu" của nhóm).
