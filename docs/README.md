# Tài liệu đặc tả (bản sao để code)

Thư mục này là **bản sao chỉ đọc** của bộ chuẩn `plan/16. Context implement` (đồng bộ ngày 09/10/2026, sau D118–D119). **Không sửa tay**: muốn đổi nội dung, sửa nguồn trong `scripts/master_spec/`, chạy `bash scripts/master_spec/final_package/make_context_implement.sh`, rồi đồng bộ lại thư mục này.

Hai tệp về kế hoạch và task Jira **không có ở đây**: `09. Sprint Plan.docx` và `10. Jira Backlog.xlsx`. Vì vậy các tệp phía sau đã được **đánh lại số**. Nội dung bên trong vẫn giữ **số gốc** của plan/16 khi nhắc tới tài liệu khác. Hãy tra số gốc ở bảng dưới.

| Số trong thư mục này | Tệp | Số gốc trong plan/16 |
|---|---|---|
| 00 | Document Index.docx | 00 |
| 01 | Business Rules.docx | 01 |
| 02 | Use Case Specification.docx | 02 |
| 03 | Functional Requirements.docx | 03 |
| 04 | Nonfunctional Requirements Specification.docx | 04 |
| 05 | Architecture and Deployment.docx | 05 |
| 06 | Logical Data Model Specification.docx | 06 |
| 07 | API Specification.docx | 07 |
| 08 | OpenAPI Specification.yaml | 08 |
| 09 | Task Cards.docx | **11** |
| 10 | Data Dictionary and Constants.docx | **12** |
| 11 | UI Screen Specification.docx | **13** |
| — | Sprint Plan.docx (không sao chép) | 09 |
| — | Jira Backlog.xlsx (không sao chép) | 10 |

## Phần cần tra trong Sprint Plan gốc

`09. Sprint Plan` không có ở đây nhưng vẫn là nguồn duy nhất cho:

- **Phụ lục B:** tên chuẩn interface Backend (`IAiServiceClient`, `IAiUsageTracker`, `IKnowledgeSyncQueue`, `IContentModerationService`…), cấu hình Qdrant, hợp đồng nội bộ AI service (`/chat/answer`, `/meal/generate`, `/moderation/check`, `/rag/*`, `/stt/*`, `/vision/*`), JSON Schema gửi Gemini.
- **§11:** quy tắc bắt buộc, lệnh kiểm tra trước khi báo xong, prompt mẫu cho agent.

Khi cần, mở `plan/16. Context implement/09. Sprint Plan.docx`.
