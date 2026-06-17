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


async def test_gemini_provider_supports_audio_upload():
    from app.services.llm_providers.gemini import GeminiProvider
    p = GeminiProvider()
    assert p.supports_audio_upload is True
    assert p.supports_streaming is True
    assert p.name == "gemini"


# ── OpenRouterProvider ────────────────────────────────────────────────────────

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


async def test_openrouter_audio_upload_raises_unsupported():
    from app.services.llm_providers.openrouter import OpenRouterProvider
    p = OpenRouterProvider()
    with pytest.raises(ProviderUnsupportedError):
        await p.upload_audio("/tmp/x.mp3")


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
    # num_ctx must be sent so long transcripts aren't truncated by Ollama's default.
    assert captured["json"]["options"]["num_ctx"] == settings.ollama_num_ctx


async def test_ollama_audio_upload_raises_unsupported():
    from app.services.llm_providers.ollama import OllamaProvider
    p = OllamaProvider()
    with pytest.raises(ProviderUnsupportedError):
        await p.upload_audio("/tmp/x.mp3")


# ── summarizer.py provider dispatch ──────────────────────────────────────────

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


# ── json_mode wiring (constrained decoding) ───────────────────────────────────

async def test_ollama_json_mode_sets_format(monkeypatch):
    """json_mode=True must add Ollama's `format=json` and honour max_tokens."""
    import httpx
    from app.services.llm_providers.ollama import OllamaProvider

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"response": "{}", "done": True}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers=None):
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OllamaProvider()
    await p.generate_text("hi", json_mode=True, max_tokens=4096)
    assert captured["json"]["format"] == "json"
    assert captured["json"]["options"]["num_predict"] == 4096


async def test_ollama_omits_format_without_json_mode(monkeypatch):
    import httpx
    from app.services.llm_providers.ollama import OllamaProvider

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"response": "{}", "done": True}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers=None):
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OllamaProvider()
    await p.generate_text("hi")
    assert "format" not in captured["json"]


async def test_openrouter_json_mode_sets_response_format(monkeypatch):
    import httpx
    from app.services.llm_providers.openrouter import OpenRouterProvider
    from app.config import settings

    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)

    captured = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}
        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers):
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    p = OpenRouterProvider()
    await p.generate_text("x", json_mode=True)
    assert captured["json"]["response_format"] == {"type": "json_object"}


# ── provider transcript path: context budgeting + map-reduce ──────────────────

def test_provider_chunk_size_math(monkeypatch):
    """The size helpers reserve output headroom out of the total num_ctx budget."""
    from app.config import settings
    from app.services import summarizer

    monkeypatch.setattr(settings, "ollama_num_ctx", 24576, raising=False)
    out = summarizer._PROVIDER_FULL_OUTPUT_TOKENS
    partial = summarizer._PROVIDER_PARTIAL_OUTPUT_TOKENS
    framing = summarizer._PROVIDER_SYSTEM_FRAMING_TOKENS
    tpc = summarizer._PROVIDER_TOK_PER_CHAR
    # Each full-output reserve covers the output cap *and* the system-prompt framing
    # that rides on the same call's input; the map reserve only needs the small
    # partial-summary instruction.
    assert summarizer._provider_single_call_chars() == int((24576 - (out + framing)) / tpc)
    assert summarizer._provider_map_chunk_chars() == int((24576 - (partial + 500)) / tpc)
    assert summarizer._provider_merge_budget_chars() == int((24576 - (out + framing + 500)) / tpc)


def test_provider_chunk_size_floored_on_tiny_ctx(monkeypatch):
    """Even an absurdly small context never yields a non-positive budget."""
    from app.config import settings
    from app.services import summarizer

    monkeypatch.setattr(settings, "ollama_num_ctx", 4096, raising=False)
    assert summarizer._provider_single_call_chars() == int(4000 / 0.8)
    assert summarizer._provider_map_chunk_chars() == int(4000 / 0.8)
    assert summarizer._provider_merge_budget_chars() == int(2000 / 0.8)


def test_chunk_transcript_reconstructs_and_respects_size():
    from app.services import summarizer

    text = ("שורה ראשונה. שורה שנייה? שורה שלישית! " * 50).strip()
    chunks = summarizer._chunk_transcript(text, 80)
    assert "".join(chunks) == text            # contiguous — no content dropped
    assert all(len(c) <= 80 for c in chunks)  # never exceeds the cap
    assert len(chunks) > 1


async def test_provider_collapse_partials_reduces_to_fit():
    """Many partials that would overflow the merge prompt are collapsed to fit."""
    from app.services import summarizer

    partials = [f"partial-{i}-" + "x" * 90 for i in range(5)]  # ~500 chars joined

    class FakeProvider:
        async def generate_text(self, prompt, *, json_mode=False, max_tokens=65536, **kw):
            return json.dumps({"summary": "merged", "key_points": []}, ensure_ascii=False)

    out = await summarizer._provider_collapse_partials(
        partials, FakeProvider(), "", budget_chars=250
    )
    assert len("\n\n---\n\n".join(out)) <= 250
    assert len(out) < len(partials)


async def test_provider_summarize_short_transcript_is_single_call(monkeypatch):
    """A short transcript takes the single-call path with json_mode + sized output."""
    from app.config import settings
    from app.services import summarizer
    from app.services.llm_providers import _reset_provider_cache
    from app.services.llm_providers.ollama import OllamaProvider

    monkeypatch.setattr(settings, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(settings, "lecture_language", "he", raising=False)
    monkeypatch.setattr(settings, "enable_exam_critique", False, raising=False)
    _reset_provider_cache()

    calls = []

    async def fake_gen(self, prompt, *, json_mode=False, max_tokens=65536, **kw):
        calls.append({"json_mode": json_mode, "max_tokens": max_tokens})
        return json.dumps(
            {"summary": "ס", "chapters": [], "quiz": [], "language": "he"},
            ensure_ascii=False,
        )

    monkeypatch.setattr(OllamaProvider, "generate_text", fake_gen)

    result = await summarizer.summarize_transcript("תמלול קצר.")
    assert result.summary == "ס"
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True
    assert calls[0]["max_tokens"] == 8192  # sized to num_ctx, not the 65536 default


async def test_provider_summarize_map_reduces_long_transcript(monkeypatch):
    """A transcript over the single-call limit is chunked, summarized, then merged."""
    from app.config import settings
    from app.services import summarizer
    from app.services.llm_providers import _reset_provider_cache
    from app.services.llm_providers.ollama import OllamaProvider

    monkeypatch.setattr(settings, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(settings, "lecture_language", "he", raising=False)
    monkeypatch.setattr(settings, "enable_exam_critique", False, raising=False)
    _reset_provider_cache()

    # Force the map-reduce branch with tiny budgets; merge budget stays large so
    # the collapse step does not fire (its own behaviour is tested separately).
    monkeypatch.setattr(summarizer, "_provider_single_call_chars", lambda: 100)
    monkeypatch.setattr(summarizer, "_provider_map_chunk_chars", lambda: 50)
    monkeypatch.setattr(summarizer, "_provider_merge_budget_chars", lambda: 10_000)

    calls = []
    final_json = json.dumps(
        {"summary": "סיכום סופי", "chapters": [], "quiz": [], "language": "he"},
        ensure_ascii=False,
    )
    partial_json = json.dumps(
        {"summary": "חלק", "key_points": ["נקודה"]}, ensure_ascii=False
    )

    async def fake_gen(self, prompt, *, json_mode=False, max_tokens=65536, **kw):
        calls.append({"json_mode": json_mode, "max_tokens": max_tokens})
        # Only the per-chunk map prompt carries the partial-summary instruction.
        is_partial = "להלן חלק מתמלול" in prompt
        return partial_json if is_partial else final_json

    monkeypatch.setattr(OllamaProvider, "generate_text", fake_gen)

    transcript = "משפט לדוגמה. " * 40  # well over the 100-char single-call limit
    result = await summarizer.summarize_transcript(transcript)

    assert result.summary == "סיכום סופי"
    assert len(calls) >= 3                       # >=2 map calls + 1 merge call
    assert all(c["json_mode"] for c in calls)    # every provider call used json_mode
    assert calls[-1]["max_tokens"] == 8192       # merge call sized to num_ctx
