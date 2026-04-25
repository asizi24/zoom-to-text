"""Unit tests for the LLM provider abstraction."""
import pytest


# ── errors.py ─────────────────────────────────────────────────────────────────

def test_provider_error_carries_structured_fields():
    from app.services.llm_providers.errors import ProviderError
    e = ProviderError(
        provider="gemini",
        stage="summarize",
        code="rate_limit",
        user_message="⚠️ מכסה הוצתה",
        technical_details="HTTP 429",
    )
    assert e.provider == "gemini"
    assert e.stage == "summarize"
    assert e.code == "rate_limit"
    assert e.user_message == "⚠️ מכסה הוצתה"
    assert e.technical_details == "HTTP 429"
    # str(e) yields the user message
    assert str(e) == "⚠️ מכסה הוצתה"


def test_provider_error_subclasses():
    from app.services.llm_providers.errors import (
        ProviderError,
        ProviderUnsupportedError,
        ProviderRateLimitError,
        ProviderTimeoutError,
        ProviderAuthError,
    )
    for sub in (
        ProviderUnsupportedError,
        ProviderRateLimitError,
        ProviderTimeoutError,
        ProviderAuthError,
    ):
        assert issubclass(sub, ProviderError)
