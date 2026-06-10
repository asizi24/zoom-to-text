"""
Domain-level error model for the processing pipeline (Task 1.5).

Why a custom exception class instead of plain RuntimeError?
- Each pipeline stage (download → preprocess → transcribe → diarize →
  summarize → flashcards) wraps its native exceptions in a ProcessingError
  carrying a stable machine-readable code and a Hebrew user-facing message.
- The DB persists this as a JSON blob (`tasks.error_details`) so the History
  tab can show structured error info instead of a raw stack trace string.
- Future tooling (alerting, retry policies) can branch on the code field.

The classifier `classify_exception()` maps common exceptions (yt-dlp,
Gemini 429, ffmpeg, JSON parsing, timeouts) into ProcessingError instances
with the right code + user_message. New error classes are added by
extending the dispatch table in `_CLASSIFIERS`.
"""
import asyncio
from enum import Enum
from typing import Any


class ProcessingStage(str, Enum):
    DOWNLOAD     = "download"
    PREPROCESS   = "preprocess"
    TRANSCRIBE   = "transcribe"
    DIARIZE      = "diarize"
    SUMMARIZE    = "summarize"
    FLASHCARDS   = "flashcards"
    UNKNOWN      = "unknown"


class ProcessingError(Exception):
    """
    Structured pipeline error.

    `stage` — which step in the pipeline produced this error.
    `code`  — machine-readable identifier (snake_case). New codes must be
              added to the catalogue in this file's `_KNOWN_CODES` set so
              callers can match against a stable list.
    `user_message`      — Hebrew, user-facing. Shown in the UI history.
    `technical_details` — English, for logs and debugging. Goes to
                          `error_details.technical_details` in SQLite.
    """

    def __init__(
        self,
        stage: ProcessingStage,
        code: str,
        user_message: str,
        technical_details: str,
    ) -> None:
        super().__init__(user_message)
        self.stage = stage
        self.code = code
        self.user_message = user_message
        self.technical_details = technical_details

    def __str__(self) -> str:  # makes default logging human-readable
        return self.user_message

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "user_message": self.user_message,
            "technical_details": self.technical_details,
        }


# ── Catalogue of known codes ─────────────────────────────────────────────────
# Adding a new code? Add it here AND wire the matcher in _CLASSIFIERS below.
_KNOWN_CODES: frozenset[str] = frozenset({
    "zoom_cookies_expired",
    "zoom_download_failed",
    "audio_corrupted",
    "transcribe_failed",
    "diarize_failed",
    "llm_rate_limit",
    "llm_invalid_json",
    "llm_unsupported_mode",
    "timeout",
    "unknown",
})


# ── Classification ───────────────────────────────────────────────────────────

def classify_exception(
    exc: BaseException,
    default_stage: ProcessingStage,
) -> ProcessingError:
    """
    Map a raw exception into a ProcessingError.

    `default_stage` is used when the input doesn't itself carry a stage hint
    (most exceptions). A pre-existing ProcessingError passes through
    unchanged so wrappers can be nested without losing precision.
    """
    if isinstance(exc, ProcessingError):
        return exc

    msg = str(exc)
    low = msg.lower()

    # Timeout — both asyncio and stdlib variants
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "10 דקות" in msg:
        return ProcessingError(
            stage=default_stage,
            code="timeout",
            user_message="⏱️ הפעולה לקחה יותר מדי זמן ופסקה — נסה שוב",
            technical_details=msg or "asyncio.TimeoutError",
        )

    # Zoom download errors — special-cased because they have their own type
    try:
        from app.services.zoom_downloader import ZoomDownloadError
    except Exception:
        ZoomDownloadError = ()  # type: ignore[assignment]
    if isinstance(exc, ZoomDownloadError):
        if "cookie" in low or "🍪" in msg or "expired or login" in low:
            return ProcessingError(
                stage=ProcessingStage.DOWNLOAD,
                code="zoom_cookies_expired",
                user_message="🍪 העוגיות פגו או חסר login — רענן מהתוסף ונסה שוב",
                technical_details=msg,
            )
        return ProcessingError(
            stage=ProcessingStage.DOWNLOAD,
            code="zoom_download_failed",
            user_message=msg,
            technical_details=msg,
        )

    # LLM rate limit / quota
    if "429" in msg or "quota" in low or "rate limit" in low or "מכסת" in msg:
        return ProcessingError(
            stage=default_stage,
            code="llm_rate_limit",
            user_message="⚠️ מכסת ה-LLM הוצתה — נסה בעוד מספר דקות",
            technical_details=msg,
        )

    # Provider returned malformed JSON
    if "json" in low or "JSON" in msg or "malformed" in low or "תקין" in msg:
        return ProcessingError(
            stage=default_stage,
            code="llm_invalid_json",
            user_message="🔄 התשובה מה-LLM הגיעה במבנה לא תקין — נסה שוב",
            technical_details=msg,
        )

    # Catch-all
    return ProcessingError(
        stage=default_stage,
        code="unknown",
        user_message=f"שגיאה: {msg[:300]}" if msg else "שגיאה לא צפויה",
        technical_details=msg or repr(exc),
    )


def known_codes() -> frozenset[str]:
    """Public accessor for the classifier's code catalogue."""
    return _KNOWN_CODES
