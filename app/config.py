"""
Application configuration using Pydantic BaseSettings.
All values can be overridden via environment variables or the .env file.
"""
from pathlib import Path
from typing import Literal, Optional
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

    # ── ivrit-ai (Hebrew-tuned Whisper) ─────────────────────────────────────────
    # HuggingFace repo of a CT2-converted ivrit-ai model (faster-whisper compatible).
    # Override via env: IVRIT_AI_MODEL=ivrit-ai/whisper-v3-ct2
    # The model is downloaded on first use into ~/.cache/faster_whisper (mounted
    # as a Docker volume so it survives container restarts).
    ivrit_ai_model: str = "ivrit-ai/whisper-large-v3-turbo-ct2"

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

    # ── Exam quality pipeline ───────────────────────────────────────────────────
    # When True, every generated exam goes through a critique + optional revise pass.
    # This costs one extra Gemini call (critique), and potentially a second (revise)
    # only when at least one question scores below the threshold.
    enable_exam_critique: bool = True
    # Average score threshold below which a question is sent for revision (1–5 scale).
    exam_critique_threshold: float = 3.8

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

    # ── LLM provider selection ─────────────────────────────────────────────────
    # Which backend handles summarization, critique, chat, and flashcards.
    # All three providers share a common interface; switching is a single
    # env-var change with no code edits required.
    llm_provider: Literal["gemini", "openrouter", "ollama"] = "gemini"

    # OpenRouter (https://openrouter.ai) — single API key, dozens of models
    openrouter_api_key:  str = ""
    openrouter_model:    str = "anthropic/claude-3.5-sonnet"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama (https://ollama.com) — local model runner. Not deployed on Fly.io;
    # used by self-hosted setups that want privacy / offline operation.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "llama3.1:70b"

    # ── LLM debugging ──────────────────────────────────────────────────────────
    # When True, persist raw LLM responses (synthesis + extraction) into
    # result_json.raw_llm_response for offline debugging. Disable in production
    # — raw responses can be 50KB+ per recording.
    llm_debug_raw_responses: bool = False

    # ── Task 1.2 — Gemini text diarization ────────────────────────────────────
    # When True, summarize_transcript runs an extra Gemini call before the
    # synthesis+extraction pair to label speakers in the transcript. Skipped
    # for GEMINI_DIRECT mode (audio model already perceives speakers) and for
    # non-Gemini providers.
    enable_diarization: bool = True

    # ── Task 2.2 — Diarization provider (gemini | pyannote) ──────────────────
    # "gemini"   — text-based diarization via the existing Gemini call (default,
    #              works on Fly.io; no extra deps).
    # "pyannote" — acoustic diarization via pyannote.audio (code-only; requires
    #              pyannote.audio + torch from requirements-heavy.txt; home server
    #              with GPU only; never deployed on Fly.io).
    diarization_provider: str = "gemini"
    # Pyannote pretrained model ID (HuggingFace Hub).
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    # HuggingFace token for gated models (pyannote requires accepting the license).
    hf_token: str = ""

    # ── Task 2.3 — WebSocket streaming (home-server only) ────────────────────
    # Never enable on Fly.io — the shared-cpu-1x machine cannot sustain
    # concurrent WebSocket connections + a loaded Whisper model.
    # Enable on a home server: ENABLE_STREAMING=true in .env
    enable_streaming: bool = False

    # ── Rate limiting ───────────────────────────────────────────────────────────
    # Maximum number of task-submission requests (POST /api/tasks and
    # POST /api/tasks/upload) allowed per IP address per minute.
    # Set to 0 to disable rate limiting entirely (e.g. during local dev).
    rate_limit_per_minute: int = 10

    # ── Content language ────────────────────────────────────────────────────────
    # "auto" — Gemini detects the lecture language and responds in kind (default).
    # ISO 639-1 code ("he", "en", "ar", "fr" …) — force a specific output language.
    lecture_language: str = "auto"

    # ── LTI 1.3 (institutional SSO) ────────────────────────────────────────────
    # OIDC state TTL — must comfortably exceed the worst-case round-trip from
    # /lti/login → user authenticates at the LMS → POST /lti/launch.
    lti_state_ttl_seconds: int = 300

    @model_validator(mode="after")
    def validate_llm_provider_credentials(self) -> "Settings":
        """Fail fast at boot if the chosen provider is missing credentials."""
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            raise ValueError(
                "llm_provider=openrouter requires openrouter_api_key to be set"
            )
        # gemini and ollama either don't need a key (ollama) or already
        # validate at first use (gemini's _get_client raises a clear error).
        return self


settings = Settings()

# Ensure required directories exist at import time
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.downloads_dir.mkdir(parents=True, exist_ok=True)
