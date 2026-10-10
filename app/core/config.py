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

    # STT (stt-service, D119): Whisper + ffmpeg, rồi Gemini trích công thức
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_language: str = "vi"           # để trống = tự nhận diện
    whisper_cpu_threads: int = 0           # 0 = để thư viện tự chọn (Cloud Run: đặt bằng số vCPU)
    whisper_beam_size: int = 5             # 1 = tham lam, nhanh hơn ~2x, kém chính xác hơn
    stt_min_seconds: float = 10.0          # BR-AI-STT-01
    stt_max_seconds: float = 300.0
    stt_min_sample_rate: int = 16000
    stt_max_upload_mb: int = 100           # BR-AI-STT-03
    stt_allowed_video_hosts: str = "res.cloudinary.com"   # phân tách bằng dấu phẩy (chống SSRF)
    stt_download_timeout_seconds: float = 60.0
    stt_ffmpeg_timeout_seconds: float = 120.0
    stt_concurrency: int = 1               # SCA-2: 1 tác vụ STT mỗi instance
    stt_recipe_temperature: float = 0.2    # B.4: tác vụ có cấu trúc ≤ 0,4
    stt_recipe_max_output_tokens: int = 2048

    # Chat
    chat_history_turns: int = 6
    chat_temperature: float = 0.3
    chat_max_output_tokens: int = 1024

    port: int = 8080

    # Swagger UI tại /docs: chỉ bật ở local/staging, KHÔNG bật ở production
    enable_docs: bool = False


    @property
    def stt_allowed_hosts(self) -> list[str]:
        return [h.strip().lower() for h in self.stt_allowed_video_hosts.split(",") if h.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
