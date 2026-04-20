"""
Application configuration using Pydantic BaseSettings.
All values can be overridden via environment variables or the .env file.
"""
from pathlib import Path
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google AI ───────────────────────────────────────────────────────────────
    # Option A: Gemini API key (AI Studio — simpler, free tier available)
    google_api_key: str = ""
    # Option B: GCP Service Account key file (Vertex AI — for production)
    google_application_credentials: str = "key.json"
    gcp_project_id: str = "gen-lang-client-0633910627"
    gcp_location: str = "us-central1"
    # Gemini model to use for summarization
    gemini_model: str = "gemini-2.5-flash"

    # ── OpenAI (optional — for WHISPER_API mode) ────────────────────────────────
    openai_api_key: str = ""

    # ── Faster-Whisper ──────────────────────────────────────────────────────────
    # Model sizes: tiny | base | small | medium | large-v3
    # CPU memory requirements: tiny=~400MB, medium=~2GB, large-v3=~4GB
    whisper_model: str = "medium"
    whisper_device: str = "cpu"         # cpu | cuda
    whisper_compute_type: str = "int8"  # int8 (CPU) | float16 (GPU)

    # ── Resource management ─────────────────────────────────────────────────────
    # Unload Whisper model from RAM after this many idle minutes
    auto_shutdown_idle_minutes: int = 30
    # Maximum audio file size to accept (bytes)
    max_upload_bytes: int = 600 * 1024 * 1024  # 600 MB

    # ── Paths ───────────────────────────────────────────────────────────────────
    # Override with DATA_DIR=/tmp/data on Cloud Run / Fly.io
    data_dir: Path = Path("data")
    # downloads_dir is always derived from data_dir — do NOT set this independently.
    # It is exposed here only so other modules can reference settings.downloads_dir.
    downloads_dir: Optional[Path] = None

    @model_validator(mode="after")
    def derive_downloads_dir(self) -> "Settings":
        """Always derive downloads_dir from data_dir so DATA_DIR env var works correctly."""
        object.__setattr__(self, "downloads_dir", self.data_dir / "downloads")
        return self

    # ── App ─────────────────────────────────────────────────────────────────────
    app_title: str = "Zoom Transcriber"
    # Base URL shown in responses (used by the Chrome extension to know where to post)
    base_url: str = "http://localhost:8000"

    # ── Auth ────────────────────────────────────────────────────────────────────
    # Comma-separated list of emails allowed to log in
    # Example: "alice@example.com,bob@example.com"
    allowed_emails: str = ""
    resend_api_key: str = ""
    # Allowed CORS origin — set to your Fly.io domain in production
    cors_origin: str = "http://localhost:8000"


settings = Settings()

# Ensure required directories exist at import time
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.downloads_dir.mkdir(parents=True, exist_ok=True)
