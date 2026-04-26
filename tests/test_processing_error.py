"""
Phase A tests for Task 1.5 — ProcessingError class + classifier.
"""
import asyncio

import pytest


# ── ProcessingStage enum ──────────────────────────────────────────────────────

def test_processing_stage_has_required_values():
    from app.errors import ProcessingStage

    expected = {"DOWNLOAD", "PREPROCESS", "TRANSCRIBE",
                "DIARIZE", "SUMMARIZE", "FLASHCARDS", "UNKNOWN"}
    actual = {s.name for s in ProcessingStage}
    assert expected.issubset(actual), f"missing: {expected - actual}"


# ── ProcessingError shape ─────────────────────────────────────────────────────

def test_processing_error_carries_structured_fields():
    from app.errors import ProcessingError, ProcessingStage

    e = ProcessingError(
        stage=ProcessingStage.DOWNLOAD,
        code="zoom_cookies_expired",
        user_message="🍪 העוגיות פגו — רענן מהתוסף",
        technical_details="yt-dlp: HTTP 401",
    )
    assert e.stage == ProcessingStage.DOWNLOAD
    assert e.code == "zoom_cookies_expired"
    assert e.user_message == "🍪 העוגיות פגו — רענן מהתוסף"
    assert e.technical_details == "yt-dlp: HTTP 401"
    # str(e) yields the user message (so default logging is human-readable)
    assert str(e) == "🍪 העוגיות פגו — רענן מהתוסף"


def test_processing_error_is_exception_subclass():
    from app.errors import ProcessingError, ProcessingStage

    with pytest.raises(ProcessingError):
        raise ProcessingError(
            stage=ProcessingStage.UNKNOWN,
            code="x",
            user_message="x",
            technical_details="x",
        )


def test_processing_error_to_dict_for_persistence():
    """The class must serialize cleanly into the JSON column we'll add in Phase C."""
    from app.errors import ProcessingError, ProcessingStage

    e = ProcessingError(
        stage=ProcessingStage.SUMMARIZE,
        code="llm_rate_limit",
        user_message="⚠️ מכסת ה-LLM הוצתה — נסה בעוד מספר דקות",
        technical_details="HTTP 429 from Gemini",
    )
    d = e.to_dict()
    assert d == {
        "stage": "summarize",
        "code": "llm_rate_limit",
        "user_message": "⚠️ מכסת ה-LLM הוצתה — נסה בעוד מספר דקות",
        "technical_details": "HTTP 429 from Gemini",
    }


def test_processing_error_keeps_original_exception_chain():
    from app.errors import ProcessingError, ProcessingStage

    original = ValueError("inner")
    try:
        raise ProcessingError(
            stage=ProcessingStage.SUMMARIZE,
            code="x",
            user_message="x",
            technical_details="x",
        ) from original
    except ProcessingError as exc:
        assert exc.__cause__ is original


# ── Classifier ────────────────────────────────────────────────────────────────

def test_classify_yt_dlp_cookie_error_to_zoom_cookies_expired():
    """Wrapped ZoomDownloadError with cookie cue → zoom_cookies_expired."""
    from app.errors import ProcessingStage, classify_exception
    from app.services.zoom_downloader import ZoomDownloadError

    exc = ZoomDownloadError("🍪 Cookies expired or login required")
    pe = classify_exception(exc, default_stage=ProcessingStage.DOWNLOAD)
    assert pe.stage == ProcessingStage.DOWNLOAD
    assert pe.code == "zoom_cookies_expired"
    assert "עוגי" in pe.user_message or "cookie" in pe.user_message.lower()


def test_classify_zoom_download_error_passthrough():
    """A plain ZoomDownloadError without cookie cue keeps its own message."""
    from app.errors import ProcessingStage, classify_exception
    from app.services.zoom_downloader import ZoomDownloadError

    exc = ZoomDownloadError("Recording is not available for this account")
    pe = classify_exception(exc, default_stage=ProcessingStage.DOWNLOAD)
    assert pe.stage == ProcessingStage.DOWNLOAD
    assert pe.code == "zoom_download_failed"
    assert "Recording is not available" in pe.user_message


def test_classify_gemini_rate_limit():
    from app.errors import ProcessingStage, classify_exception

    exc = RuntimeError("Gemini API HTTP 429: quota exceeded")
    pe = classify_exception(exc, default_stage=ProcessingStage.SUMMARIZE)
    assert pe.stage == ProcessingStage.SUMMARIZE
    assert pe.code == "llm_rate_limit"
    assert "מכס" in pe.user_message  # Hebrew "quota"


def test_classify_timeout():
    from app.errors import ProcessingStage, classify_exception

    exc = asyncio.TimeoutError()
    pe = classify_exception(exc, default_stage=ProcessingStage.SUMMARIZE)
    assert pe.code == "timeout"
    assert pe.stage == ProcessingStage.SUMMARIZE


def test_classify_invalid_json_from_provider():
    from app.errors import ProcessingStage, classify_exception

    exc = RuntimeError("🔄 Gemini החזיר JSON לא תקין — זו שגיאה חולפת, נסה שוב")
    pe = classify_exception(exc, default_stage=ProcessingStage.SUMMARIZE)
    assert pe.code == "llm_invalid_json"


def test_classify_unknown_falls_back_to_default_stage():
    from app.errors import ProcessingStage, classify_exception

    exc = RuntimeError("totally unrecognized failure")
    pe = classify_exception(exc, default_stage=ProcessingStage.PREPROCESS)
    assert pe.stage == ProcessingStage.PREPROCESS
    assert pe.code == "unknown"
    # technical_details preserves the original message
    assert "totally unrecognized failure" in pe.technical_details


def test_classify_existing_processing_error_passthrough():
    """If the input is already a ProcessingError, return it unchanged."""
    from app.errors import ProcessingError, ProcessingStage, classify_exception

    original = ProcessingError(
        stage=ProcessingStage.DIARIZE,
        code="custom",
        user_message="custom",
        technical_details="custom",
    )
    out = classify_exception(original, default_stage=ProcessingStage.UNKNOWN)
    assert out is original
