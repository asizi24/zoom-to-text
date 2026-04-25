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


# ── base.py ─────────────────────────────────────────────────────────────────

import asyncio
from app.services.llm_providers.errors import (
    ProviderRateLimitError,
    ProviderAuthError,
    ProviderUnsupportedError,
)


def test_audio_ref_is_dataclass():
    from app.services.llm_providers.base import AudioRef
    ref = AudioRef(provider_name="gemini", provider_specific_id="files/abc", raw=None)
    assert ref.provider_name == "gemini"
    assert ref.provider_specific_id == "files/abc"
    assert ref.raw is None


def test_default_provider_methods_raise_unsupported():
    """Subclasses inherit a sensible default that raises ProviderUnsupportedError."""
    from app.services.llm_providers.base import LLMProvider, AudioRef

    class Stub(LLMProvider):
        name = "stub"
        supports_audio_upload = False
        supports_streaming = True
        default_model = "stub-1"

        async def generate_text(self, prompt, **kw): return "ok"
        async def stream_text(self, contents, **kw):
            if False:
                yield ""

    s = Stub()
    with pytest.raises(ProviderUnsupportedError):
        asyncio.run(s.upload_audio("/tmp/x"))
    with pytest.raises(ProviderUnsupportedError):
        asyncio.run(s.generate_text_with_audio(
            AudioRef("stub", "x"), "prompt"
        ))
    # cleanup_audio is a no-op (does not raise)
    asyncio.run(s.cleanup_audio(AudioRef("stub", "x")))


@pytest.mark.asyncio
async def test_with_retry_succeeds_after_transient_error():
    from app.services.llm_providers.base import _with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderRateLimitError(
                provider="x", stage="t", code="429",
                user_message="rate limit", technical_details=""
            )
        return "result"

    out = await _with_retry(fn, max_retries=4, base_delay=0.0)
    assert out == "result"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_auth_errors():
    from app.services.llm_providers.base import _with_retry

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise ProviderAuthError(
            provider="x", stage="t", code="401",
            user_message="bad key", technical_details=""
        )

    with pytest.raises(ProviderAuthError):
        await _with_retry(fn, max_retries=4, base_delay=0.0)
    # Auth errors are terminal — never retried
    assert calls["n"] == 1


# ── config validation ─────────────────────────────────────────────────────────

def test_settings_default_provider_is_gemini():
    from app.config import settings
    # Default value when nothing is set in env
    assert settings.llm_provider == "gemini"


def test_settings_openrouter_requires_api_key(monkeypatch):
    """When llm_provider=openrouter, the api key must be set or startup fails."""
    from app.config import Settings
    with pytest.raises(ValueError, match="openrouter_api_key"):
        Settings(
            llm_provider="openrouter",
            openrouter_api_key="",
            google_api_key="x",  # ensure gemini check would pass
        )


def test_settings_accepts_valid_openrouter_config():
    from app.config import Settings
    s = Settings(
        llm_provider="openrouter",
        openrouter_api_key="sk-or-test-1",
    )
    assert s.llm_provider == "openrouter"
    assert s.openrouter_api_key == "sk-or-test-1"


def test_settings_ollama_does_not_require_api_key():
    from app.config import Settings
    s = Settings(llm_provider="ollama")
    assert s.llm_provider == "ollama"
    assert s.ollama_base_url == "http://localhost:11434"
