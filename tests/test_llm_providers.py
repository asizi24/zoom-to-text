"""Unit tests for the LLM provider abstraction."""
import json

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


# ── factory ───────────────────────────────────────────────────────────────────

def test_get_provider_returns_gemini_by_default(monkeypatch):
    from app.config import settings
    from app.services.llm_providers import get_provider, _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)
    _reset_provider_cache()
    p = get_provider()
    assert p.name == "gemini"


def test_get_provider_returns_openrouter_when_configured(monkeypatch):
    from app.config import settings
    from app.services.llm_providers import get_provider, _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-1", raising=False)
    _reset_provider_cache()
    p = get_provider()
    assert p.name == "openrouter"


def test_get_provider_returns_ollama_when_configured(monkeypatch):
    from app.config import settings
    from app.services.llm_providers import get_provider, _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "ollama", raising=False)
    _reset_provider_cache()
    p = get_provider()
    assert p.name == "ollama"


def test_get_provider_caches_instance(monkeypatch):
    from app.config import settings
    from app.services.llm_providers import get_provider, _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)
    _reset_provider_cache()
    a = get_provider()
    b = get_provider()
    assert a is b


# ── GeminiProvider ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gemini_provider_generate_text_uses_existing_helper(monkeypatch):
    """GeminiProvider.generate_text reuses summarizer._generate_with_retry."""
    from app.services.llm_providers.gemini import GeminiProvider
    from app.services import summarizer
    from unittest.mock import MagicMock

    captured = {"contents": None}

    def fake_retry(client, contents, max_retries=3):
        captured["contents"] = contents
        m = MagicMock()
        m.text = "raw model output"
        part = MagicMock()
        part.thought = False
        part.text = "raw model output"
        m.candidates = [MagicMock()]
        m.candidates[0].content.parts = [part]
        return m

    monkeypatch.setattr(summarizer, "_generate_with_retry", fake_retry)
    monkeypatch.setattr(summarizer, "_get_client", lambda: object())
    p = GeminiProvider()
    out = await p.generate_text("hello prompt")
    assert out == "raw model output"
    assert captured["contents"] == "hello prompt"


@pytest.mark.asyncio
async def test_gemini_provider_supports_audio_upload():
    from app.services.llm_providers.gemini import GeminiProvider
    p = GeminiProvider()
    assert p.supports_audio_upload is True
    assert p.supports_streaming is True
    assert p.name == "gemini"


# ── OpenRouterProvider ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openrouter_generate_text_builds_correct_request(monkeypatch):
    """OpenRouter sends a chat-completions POST with the right shape."""
    import httpx
    from app.services.llm_providers.openrouter import OpenRouterProvider
    from app.config import settings

    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)
    monkeypatch.setattr(settings, "openrouter_model", "test-model", raising=False)

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "model output"}}]}
        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OpenRouterProvider()
    out = await p.generate_text("hello")
    assert out == "model output"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_openrouter_audio_upload_raises_unsupported():
    from app.services.llm_providers.openrouter import OpenRouterProvider
    p = OpenRouterProvider()
    with pytest.raises(ProviderUnsupportedError):
        await p.upload_audio("/tmp/x.mp3")


@pytest.mark.asyncio
async def test_openrouter_classifies_401_as_auth_error(monkeypatch):
    import httpx
    from app.services.llm_providers.openrouter import OpenRouterProvider
    from app.config import settings

    monkeypatch.setattr(settings, "openrouter_api_key", "bad", raising=False)

    class FakeResp:
        status_code = 401
        text = "invalid api key"
        def raise_for_status(self):
            raise httpx.HTTPStatusError("401", request=None, response=self)
        def json(self):
            return {"error": {"message": "invalid api key"}}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **kw): return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OpenRouterProvider()
    with pytest.raises(ProviderAuthError):
        await p.generate_text("x")


# ── OllamaProvider ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ollama_generate_text_builds_correct_request(monkeypatch):
    import httpx
    from app.services.llm_providers.ollama import OllamaProvider
    from app.config import settings

    monkeypatch.setattr(settings, "ollama_base_url", "http://test-ollama:11434", raising=False)
    monkeypatch.setattr(settings, "ollama_model", "test-llama", raising=False)

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"response": "ollama output", "done": True}
        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OllamaProvider()
    out = await p.generate_text("hi")
    assert out == "ollama output"
    assert captured["url"] == "http://test-ollama:11434/api/generate"
    assert captured["json"]["model"] == "test-llama"
    assert captured["json"]["prompt"] == "hi"
    assert captured["json"]["stream"] is False


@pytest.mark.asyncio
async def test_ollama_audio_upload_raises_unsupported():
    from app.services.llm_providers.ollama import OllamaProvider
    p = OllamaProvider()
    with pytest.raises(ProviderUnsupportedError):
        await p.upload_audio("/tmp/x.mp3")


# ── summarizer.py provider dispatch ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_summarize_transcript_uses_openrouter_when_configured(monkeypatch):
    """When LLM_PROVIDER=openrouter, summarize_transcript goes via the provider."""
    from app.config import settings
    from app.services import summarizer
    from app.services.llm_providers import _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)
    _reset_provider_cache()

    # Ensure the Gemini path is NOT taken
    def fail_gemini_path(*a, **kw):
        raise AssertionError("Gemini path should not run when llm_provider=openrouter")
    monkeypatch.setattr(summarizer, "_get_client", fail_gemini_path)

    # Stub the provider's generate_text
    canned_json = json.dumps({
        "summary": "סיכום בדיקה",
        "chapters": [],
        "quiz": [],
        "language": "he",
    }, ensure_ascii=False)

    from app.services.llm_providers.openrouter import OpenRouterProvider
    async def fake_gen(self, prompt, **kw):
        return canned_json
    monkeypatch.setattr(OpenRouterProvider, "generate_text", fake_gen)

    # Disable critique to keep this test simple — full pipeline tested elsewhere
    monkeypatch.setattr(settings, "enable_exam_critique", False, raising=False)

    result = await summarizer.summarize_transcript("תמלול לדוגמה")
    assert result.summary == "סיכום בדיקה"


@pytest.mark.asyncio
async def test_summarize_audio_with_openrouter_raises_unsupported(monkeypatch):
    from app.config import settings
    from app.services import summarizer
    from app.services.llm_providers import _reset_provider_cache, ProviderUnsupportedError

    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)
    _reset_provider_cache()

    # No need to mock anything — the call should raise before any network I/O
    with pytest.raises(ProviderUnsupportedError):
        await summarizer.summarize_audio("/tmp/fake.mp3")
