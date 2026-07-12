"""
Application configuration using Pydantic BaseSettings.
All values can be overridden via environment variables or the .env file.
"""
from pathlib import Path
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder values that mean "no real Resend key". When the key matches one
# of these, auth logs the magic link to the terminal instead of emailing it
# (local dev bypass — see app/api/auth.py). Real Resend keys start with "re_".
_RESEND_PLACEHOLDER_KEYS = {
    "", "dummy", "test", "changeme", "placeholder", "none",
    "your-resend-api-key", "re_your_key_here",
}


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
    # GPU (float16) VRAM: medium=~2.5GB, large-v3=~5GB — fits an RTX 4070 Ti (12GB)
    whisper_model: str = "medium"
    # "auto" picks cuda when a CUDA device is visible (requires the GPU-enabled
    # image + compose device reservation), otherwise falls back to cpu.
    whisper_device: str = "auto"          # auto | cpu | cuda
    # "auto" pairs with the device: float16 on cuda, int8 on cpu.
    whisper_compute_type: str = "auto"    # auto | int8 | float16 | int8_float16

    # ── Transcription quality (faster-whisper decode parameters) ───────────────
    # beam_size/best_of=5 is the reference Whisper setting; raising them
    # trades speed for a small accuracy gain (cheap on GPU, painful on CPU).
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    # False stops repetition loops ("hallucinated" duplicated lines) on long
    # lectures — each 30s window is decoded without the previous window's text.
    # The temperature-fallback ladder in transcriber.py handles the rare
    # low-confidence window that conditioning would have helped.
    whisper_condition_on_previous_text: bool = False
    # Domain vocabulary hint fed to the first decode window — improves spelling
    # of recurring technical terms. Example for a networking course:
    # WHISPER_INITIAL_PROMPT="שיעור ברשתות תקשורת: ניתוב, כתובות IP, סאבנט, פרוטוקול"
    whisper_initial_prompt: str = ""
    # VAD (Silero) tuning: silence longer than min_silence is skipped entirely
    # (the #1 source of hallucinated text), pad keeps word edges intact.
    whisper_vad_min_silence_ms: int = 700
    whisper_vad_speech_pad_ms: int = 400
    # Where model weights are downloaded/cached. In Docker this is overridden to
    # /app/models (ENV in Dockerfile) and mounted as a named volume — the path
    # must match the compose volume target or every container recreation
    # re-downloads the multi-GB model.
    whisper_cache_dir: Path = Path.home() / ".cache" / "faster_whisper"

    # ── ivrit-ai (Hebrew-tuned Whisper) ─────────────────────────────────────────
    # HuggingFace repo of a CT2-converted ivrit-ai model (faster-whisper compatible).
    # Override via env: IVRIT_AI_MODEL=ivrit-ai/whisper-v3-ct2
    # The model is downloaded on first use into whisper_cache_dir (mounted
    # as a Docker volume so it survives container restarts).
    ivrit_ai_model: str = "ivrit-ai/whisper-large-v3-turbo-ct2"

    # ── Resource management ─────────────────────────────────────────────────────
    # Unload Whisper model from RAM after this many idle minutes
    auto_shutdown_idle_minutes: int = 30
    # Maximum audio file size to accept (bytes)
    max_upload_bytes: int = 600 * 1024 * 1024  # 600 MB
    # How many pipelines may run concurrently. 1 = at most one download/
    # transcription/summarization in flight; extra submissions wait in the
    # queue. Raise only if RAM allows a second Whisper transcription.
    pipeline_concurrency: int = 1

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

    # ── Exam quality pipeline ───────────────────────────────────────────────────
    # When True, every generated exam goes through a critique + optional revise pass.
    # This costs one extra Gemini call (critique), and potentially a second (revise)
    # only when at least one question scores below the threshold.
    enable_exam_critique: bool = True
    # Average score threshold below which a question is sent for revision (1–5 scale).
    exam_critique_threshold: float = 3.5

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

    @property
    def resend_configured(self) -> bool:
        """True when a real (non-placeholder) Resend API key is set."""
        return self.resend_api_key.strip().lower() not in _RESEND_PLACEHOLDER_KEYS


settings = Settings()

# Ensure required directories exist at import time
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.downloads_dir.mkdir(parents=True, exist_ok=True)
