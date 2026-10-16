"""Cấu hình đọc từ biến môi trường / `.env` (D118, D120). Mọi giá trị cấu hình đi qua đây."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Bảo mật nội bộ (SEC-11)
    internal_api_key: str = ""

    # Gemini qua Vertex AI (D120)
    google_genai_use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "asia-southeast1"
    # Tệp JSON service account khi chạy local/VPS; Cloud Run dùng service account gắn sẵn (để trống).
    google_application_credentials: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = 10.0
    gemini_max_retries: int = 2
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    embedding_concurrency: int = 4

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "veggielife_knowledge"

    # RAG (D99; tinh chỉnh bằng golden set)
    rag_top_k: int = 5
    rag_min_score: float = 0.65
    rag_timeout_seconds: float = 3.0
    rag_chunk_tokens: int = 500
    rag_chunk_overlap: int = 50
    knowledge_dir: str = "knowledge/nutrition"

    # Giới hạn tải lên cho ảnh (BR-AI-CV-01): ≤ 10 MB, ≥ 640×480
    max_upload_mb: int = 10
    image_min_width: int = 640
    image_min_height: int = 480

    # Kiểm duyệt (FN-POST-03/04/07)
    moderation_temperature: float = 0.1
    moderation_max_output_tokens: int = 512
    moderation_max_images: int = 4
    moderation_image_hosts: str = "res.cloudinary.com"   # phân tách bằng dấu phẩy (chống SSRF)
    moderation_image_timeout_seconds: float = 10.0

    # Vision (FN-VIS-01)
    vision_temperature: float = 0.2
    vision_max_output_tokens: int = 1024

    # Chat
    chat_history_turns: int = 6
    chat_temperature: float = 0.3
    chat_max_output_tokens: int = 1024

    port: int = 8080

    # Swagger UI tại /docs: chỉ bật ở local/staging, KHÔNG bật ở production
    enable_docs: bool = False


    @property
    def moderation_allowed_image_hosts(self) -> list[str]:
        return [h.strip().lower() for h in self.moderation_image_hosts.split(",") if h.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
