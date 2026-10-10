# Kế hoạch triển khai STT (video → phiên âm → công thức) — stt-service

**Lập:** 10/10/2026 · **Trạng thái:** Draft, chờ Leader duyệt · **Người làm theo Sprint Plan:** Huỳnh Việt Phát (S5-P2); Nguyễn Văn Minh Tâm duyệt (PR đụng kiến trúc)
**Căn cứ:** D118, D119, D120 (Decision Log); FN-POST-09, FN-POST-10 (`docs/09. Task Cards.docx`); BR-AI-STT-01..03, BR-GUEST-02, BR-ADM-AI-002/003; NFR CO-13/14/16, SI-2/3, CI-4/5, PER-7, SEC-7, SCA-2, EFF-3; Sprint Plan Phụ lục B.1, B.3, B.4 (`plan/16. Context implement/09. Sprint Plan.docx`); prototype `prototype/1. Whisper Prototype`.

---

## 1. Phạm vi

**Làm trong kế hoạch này (chỉ repo AI service, bản triển khai stt-service):**
- Khung stt-service: app riêng, `/health`, khóa `X-Internal-Api-Key`, `Dockerfile.stt`.
- `POST /stt/video-to-recipe`: nhận `videoUrl` (Cloudinary) → tải → kiểm tra → Whisper → Gemini → công thức.
- `POST /stt/transcribe`: nhận tệp multipart → phiên âm (dùng để thử và chẩn đoán).
- Kiểm thử, golden mẫu, CI, compose dev, bản triển khai Cloud Run.

**Không làm ở đây:**

| Phần | Task |
|---|---|
| Backend: tải video lên Cloudinary, `stt_queue`, `SttWorker`, ghi `VideoTranscripts`/`Recipes`, hạn mức Guest, SignalR, `IAiServiceClient.VideoToRecipeAsync` | S6-P3, FN-POST-08/09/10 |
| FE: `VideoPlayer` + `TranscriptPanel` | Sprint 7 |
| Nạp phiên âm vào kho RAG (`VIDEO_TRANSCRIPT`, `VIDEO_RECIPE`) | Backend gọi `/rag/upsert` đã có |
| Tải video từ link TikTok/Reels, `/stt/url-to-recipe`, `/summarize-recipe`, `/video-chat` | Phase 2 / ngoài MVP |

---

## 2. Quy tắc nghiệp vụ phải hiện thực

| Quy tắc | Hiện thực trong stt-service |
|---|---|
| BR-AI-STT-01 | Thời lượng 10 giây–5 phút, tần số lấy mẫu âm thanh ≥ 16 kHz; kiểm bằng `ffprobe` trước khi chạy Whisper |
| BR-AI-STT-03 | Chỉ MP4/MOV, ≤ 100 MB; kiểm magic bytes; chỉ nhận video từ máy chủ Cloudinary, không tải link TikTok/Reels |
| BR-AI-STT-02 | Kết quả luôn là bản nháp (Backend gắn `Recipes.Status = DRAFT`); service không quyết định công khai |
| SEC-7 | Kiểm chữ ký tệp, định dạng, kích thước, thời lượng |
| PER-7 | Video 5 phút xong ≤ 90 giây (chạy nền ở Backend) |
| SCA-2 | 1 tác vụ STT mỗi instance tại một thời điểm (Cloud Run `concurrency=1` + semaphore trong app) |
| EFF-3 | Whisper `int8`, giữ bộ nhớ < 4 GiB |
| BR-ADM-AI-003 | Trả `usage` + `audit` để Backend ghi `AIQuotaLogs(SUMMARIZE_VIDEO)` và `AuditLogs` |

---

## 3. Luồng xử lý

```
POST /stt/video-to-recipe {videoUrl}
  ─► kiểm videoUrl: https, host ∈ danh sách cho phép (res.cloudinary.com), không IP nội bộ  (chống SSRF)
  ─► tải về tệp tạm: streaming, dừng khi vượt 100 MB, timeout; kiểm magic bytes MP4/MOV (ftyp)
  ─► ffprobe: có luồng âm thanh, thời lượng 10–300 giây, sample rate ≥ 16 kHz
  ─► ffmpeg: trích WAV 16 kHz mono 16-bit
  ─► faster-whisper (int8, beam 5, VAD) trong threadpool, semaphore = 1  ─► rawTranscript, sttModelVersion
  ─► rỗng/không có lời thoại ─► lỗi NO_SPEECH
  ─► Gemini 2.5 Flash (Vertex, qua core/gemini_client): JSON Schema, temperature ≤ 0,4,
     phiên âm đặt trong khối "dữ liệu" (không phải chỉ dẫn), thiếu số lượng ─► null, không đoán
  ─► kiểm lại ở tầng mã: schema, title không rỗng, ingredients/steps là mảng, quantity số hoặc null
  ─► xóa tệp tạm (luôn, kể cả khi lỗi) ─► trả kết quả

POST /stt/transcribe (multipart file)  ─► cùng các bước kiểm tệp, Whisper; không gọi Gemini
```

---

## 4. Hợp đồng endpoint (Sprint Plan Phụ lục B.3)

Header `X-Internal-Api-Key` bắt buộc (sai/thiếu → 401). JSON camelCase.

| Method | Path | Vào | Ra (200) |
|---|---|---|---|
| GET | `/health` | — | `{status: "ok", whisper: "loaded"/"not_loaded", gemini: "configured"/"missing"}` |
| POST | `/stt/transcribe` | multipart `file` (MP4/MOV ≤ 100 MB) | `durationSeconds`, `language`, `sttModelVersion`, `rawTranscript` |
| POST | `/stt/video-to-recipe` | `{"videoUrl": "https://res.cloudinary.com/..."}` | `durationSeconds`, `sttModelVersion`, `rawTranscript`, `recipe {title, servings (int/null), ingredients[{name, quantity (số/null), unit (chuỗi/null)}], steps[]}` |

**Đề xuất thêm (cần Leader chốt, vì B.3 chưa có nhưng NFR SI-3 yêu cầu phản hồi AI kèm usage):** `/stt/video-to-recipe` trả thêm `usage {model, promptTokens, completionTokens, latencyMs, temperature}` (phần Gemini) và `audit {prompt, rawResponse}`, giống `/chat/answer`.

**Mã lỗi** (B.3 chỉ nêu HTTP 400/413/502/504; các `errorCode` bên dưới là đề xuất):

| HTTP | `errorCode` | Khi nào | Backend ánh xạ |
|---|---|---|---|
| 400 | `VALIDATION_ERROR` | Đầu vào sai định dạng, `videoUrl` không hợp lệ hoặc host không được phép | `ERR_VALIDATION` |
| 400 | `MEDIA_INVALID` | Không phải MP4/MOV, hỏng, không có luồng âm thanh, sample rate < 16 kHz | — (chờ Leader chốt) |
| 400 | `DURATION_INVALID` | Ngoài 10 giây–5 phút | `ERR_STT_DURATION_INVALID` (422) |
| 413 | `FILE_TOO_LARGE` | Vượt 100 MB | — |
| 422 | `NO_SPEECH` | Không nhận ra lời thoại | `VideoTranscripts.Status = FAILED` |
| 502 | `MEDIA_DOWNLOAD_FAILED` | Không tải được `videoUrl` | thử lại theo `SttWorker` |
| 502 | `STT_ERROR` | Whisper/ffmpeg lỗi | thử lại, quá 3 lần thì `FAILED` |
| 502 | `GEMINI_ERROR` | Vertex AI lỗi | tính vào ngắt mạch (`ERR_AI_CIRCUIT_OPEN`) |
| 504 | `AI_TIMEOUT` | Quá thời gian | tính vào ngắt mạch |

`MEDIA_INVALID`, `NO_SPEECH`, `MEDIA_DOWNLOAD_FAILED`, `STT_ERROR`, `FILE_TOO_LARGE` chưa có trong B.3 hay Data Dictionary mục 5.

---

## 5. Cấu hình

Dùng chung `.env.example` hiện có (Vertex, `INTERNAL_API_KEY`), thêm:

```
WHISPER_MODEL=small                  # base | small (prototype dùng small); chốt sau khi đo mục G2
WHISPER_COMPUTE_TYPE=int8
WHISPER_DEVICE=cpu
WHISPER_LANGUAGE=vi                  # cố định tiếng Việt; để trống = tự nhận diện
STT_MIN_SECONDS=10
STT_MAX_SECONDS=300
STT_MIN_SAMPLE_RATE=16000
STT_MAX_UPLOAD_MB=100
STT_ALLOWED_VIDEO_HOSTS=res.cloudinary.com
STT_DOWNLOAD_TIMEOUT_SECONDS=60
STT_CONCURRENCY=1
```

Mọi giá trị đọc từ `core/config.py` (NFR MOD-3), không rải rác trong mã. Trên Cloud Run dùng service account gắn sẵn (vai trò *Vertex AI User*), không có tệp JSON.

---

## 6. Tệp sẽ tạo

```
app/stt_main.py                       app FastAPI riêng cho stt-service: /health + router stt; nạp Whisper khi khởi động
app/api/routers/stt.py                /stt/transcribe, /stt/video-to-recipe (router mỏng)
app/schemas/stt.py                    Pydantic camelCase theo mục 4
app/services/stt/media.py             kiểm videoUrl (chống SSRF), tải streaming, magic bytes, tệp tạm
app/services/stt/audio.py             ffprobe (thời lượng, sample rate), ffmpeg → WAV 16 kHz mono
app/services/stt/whisper.py           WhisperEngine: nạp model 1 lần, phiên âm trong threadpool, semaphore
app/services/stt/recipe_extractor.py  prompt + JSON Schema + kiểm lại đầu ra (qua core/gemini_client)
app/services/stt/pipeline.py          điều phối các bước ở mục 3, dọn tệp tạm
Dockerfile.stt                        python 3.11-slim, cài ffmpeg, tải trước model Whisper khi build, chạy không root
requirements-stt.txt                  faster-whisper, python-multipart (thêm vào cùng requirements hiện có)
scripts/check_stt.py                  chạy thử trên tệp thật, in thời gian từng bước
tests/unit/ test_stt_*.py             media (SSRF, magic bytes), audio, recipe_extractor, pipeline (Whisper giả)
tests/integration/test_stt_*.py       ffmpeg thật, tệp mẫu sinh bằng ffmpeg (không commit video)
docker-compose.dev.yml                thêm stt-service cổng 8082
.github/workflows/ci.yml              thêm build Dockerfile.stt
```

Giữ nguyên `app/main.py` (ai-service): không cài `faster-whisper` vào image ai-service để image nhẹ.

---

## 7. Lịch và tiêu chí xong

Theo Sprint Plan, S5-P2 diễn ra D3–D4 của Sprint 5 (14–15/10); phần Backend S6-P3 dùng kết quả này ở Sprint 6.

| Ngày | Giai đoạn | Việc | Xong khi |
|---|---|---|---|
| 10/10 ✔ | G0 Chuẩn bị | Chạy prototype với 3 video mẫu (S5-X5): ghi thời gian, RAM, độ chính xác tiếng Việt của `base` và `small`; kiểm `ffprobe`, `ffmpeg` có trên máy | Có bảng đo; chốt `WHISPER_MODEL`; xác nhận 100 MB vs giới hạn Cloud Run (mục 9) |
| 12/10 | G1 Khung | `stt_main.py`, `/health`, khóa nội bộ, `Dockerfile.stt`, compose, CI | `/health` 200 khi đúng khóa, 401 khi sai; image build và chạy được; `ruff` + `pytest` xanh |
| 13/10 | G2 Media và Whisper | `media.py`, `audio.py`, `whisper.py`, `/stt/transcribe` | Video mẫu 60 giây → phiên âm tiếng Việt; ngoài 10 giây–5 phút bị từ chối; sai magic bytes bị từ chối; không còn tệp tạm sau khi xong |
| 14/10 | G3 Công thức | `recipe_extractor.py`, `pipeline.py`, `/stt/video-to-recipe` | Video mẫu → JSON công thức đúng schema; thiếu số lượng ra `null`; phiên âm chứa câu chèn lệnh không làm đổi định dạng đầu ra |
| 15/10 | G4 An toàn và lỗi | Chống SSRF, giới hạn tải, ánh xạ mã lỗi, `concurrency=1`, ghi `usage`/`audit` | Mỗi mã lỗi ở mục 4 có test; URL host lạ/IP nội bộ bị từ chối; hai yêu cầu đồng thời được xếp hàng |
| 16–17/10 | G5 Cloud Run | Deploy staging (`--memory 4Gi --cpu 2 --min-instances 0`, `asia-southeast1`); ví dụ request/response cho Backend (S6-P3) | `/health` và một video 3 phút chạy thật qua URL Cloudinary; video 5 phút ≤ 90 giây (đo P95 trên 5 video); Backend gọi được bằng ví dụ |

---

## 7.1. Kết quả G0 (10/10, đo trên Apple M2, video mẫu của nhóm 63–76 giây, int8)

| Model | Luồng CPU | Beam | Thời gian / video 75,9 giây | Ước tính video 5 phút | RAM đỉnh | PER-7 (≤ 90 giây) |
|---|---|---|---|---|---|---|
| small | 2 | 5 | 20–25 giây | ~79–99 giây | ~0,8 GB | đạt sát |
| small | 2 | 1 | 16 giây | ~64 giây | ~0,9 GB | đạt |
| medium | 2 | 1 | 70 giây | ~276 giây | ~0,9 GB | không |
| medium | 2 | 5 | 83–127 giây | ~330–500 giây | ~1,1 GB | không |

- `medium` phiên âm tiếng Việt chính xác hơn `small`, nhưng sau bước Gemini thì công thức của `small` tốt ngang (Gemini sửa được "tiêu đen xay", "hạt nêm nấm chay"; còn bản `medium` vẫn để sai "tiêu đen say", "hành qua rô"). Gemini mất ~3,6 giây (≈ 630 + 480 token).
- **Chốt mặc định: `WHISPER_MODEL=small`, beam 5**; `medium` là tùy chọn khi chấp nhận chậm hơn ~4 lần. Trên Cloud Run đặt `WHISPER_CPU_THREADS` bằng số vCPU; nếu video 5 phút vượt 90 giây khi đo ở G5, tăng lên 4 vCPU hoặc hạ beam xuống 1.
- Số đo trên máy M2 chỉ để chọn model, chưa thay cho đo P95 trên Cloud Run thật (G5).

## 8. Kiểm thử

- **Đơn vị** (không cần mạng, Whisper và Gemini giả): kiểm `videoUrl`, magic bytes, thời lượng, sample rate, parse JSON công thức, dọn tệp tạm, ánh xạ lỗi.
- **Tích hợp:** `ffmpeg` thật với tệp sinh tại chỗ (âm thanh tổng hợp 12 giây, 4 giây, 6 phút) để kiểm giới hạn thời lượng, không commit video.
- **Chạy thật** (đánh dấu `live`, chỉ chạy tay hoặc khi merge `develop`): Whisper thật + Vertex thật trên video mẫu của nhóm; đo thời gian từng bước. Video mẫu để ngoài repo, đường dẫn qua biến môi trường.
- **Ca biên bắt buộc** (Task Cards): 9 giây / 10 giây / 5 phút / 5 phút 1 giây; sample rate 8 kHz vs 16 kHz; 100 MB vs 101 MB.

---

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| **Cloud Run giới hạn thân yêu cầu HTTP/1 là 32 MiB**, trong khi `/stt/transcribe` nhận video tới 100 MB qua multipart (prototype cũng ghi 32 MB) | Luồng chính của Backend là `VideoToRecipeAsync(Uri)` nên dùng `videoUrl`, không bị giới hạn này. `/stt/transcribe` chỉ dùng để thử/chẩn đoán, ghi rõ giới hạn tệp nhỏ; báo Leader sửa B.3 nếu cần |
| Khởi động nguội (min-instances 0) cộng thời gian nạp model | Tải trước trọng số model vào image khi build; Backend đặt timeout ≥ 150 giây và thử lại (đã có trong `SttWorker`, 3 lần) |
| Whisper `small` tốn RAM hoặc chậm trên 2 vCPU | Đo ở G0; nếu video 5 phút > 90 giây thì dùng `base` hoặc bật CPU boost; ghi vào Decision Log |
| SSRF qua `videoUrl` | Chỉ https, chỉ host cho phép, chặn IP nội bộ, không theo chuyển hướng sang host khác |
| Tệp lớn lấp đầy đĩa Cloud Run (bộ nhớ trong) | Tải streaming và dừng ở 100 MB, xóa tệp tạm trong `finally` |
| Phiên âm sai với tiếng ồn nền hoặc phương ngữ | Đầu ra luôn là bản nháp để tác giả sửa (BR-AI-STT-01/02); VAD bật; ghi `sttModelVersion` |
| Gemini bịa số lượng nguyên liệu | Schema cho phép `null`; prompt ghi "không đoán"; kiểm lại ở tầng mã |
| Câu chèn lệnh trong lời thoại video (prompt injection) | Phiên âm đặt trong khối dữ liệu; đầu ra bị ép theo JSON Schema; thêm ca test |
| Vertex AI ở `asia-southeast1` thiếu model | Đã kiểm và chạy được với RAG (cùng project) |
| Chi phí Cloud Run/Vertex | Nằm trong free tier; đặt budget alert $10 như RAG |
| Hai bản triển khai dùng chung mã nhưng khác phụ thuộc | `requirements-stt.txt` tách riêng; test import đảm bảo ai-service không import `faster_whisper` |

---

## 10. Câu hỏi còn mở (cần Leader chốt)

- Thêm `usage`/`audit` vào phản hồi `/stt/video-to-recipe`, và mã lỗi mới ở mục 4, vào Phụ lục B.3 và Data Dictionary mục 5.
- Chốt `WHISPER_MODEL` (`base` hay `small`) sau khi đo ở G0.
- Giữ `/stt/transcribe` multipart (kèm giới hạn 32 MiB trên Cloud Run) hay bỏ khỏi hợp đồng và chỉ giữ `/stt/video-to-recipe`.
- `WHISPER_LANGUAGE`: cố định `vi` hay tự nhận diện (video có thể trộn từ tiếng Anh).
