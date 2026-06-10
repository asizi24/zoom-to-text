# LLM Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LLM backend swappable between Gemini, OpenRouter, and Ollama via a single `LLM_PROVIDER` env var, without breaking any existing behavior or test.

**Architecture:** New `app/services/llm_providers/` package defines an `LLMProvider` ABC. `GeminiProvider` is a facade over the existing `summarizer.py` helpers (so existing tests are untouched). `OpenRouterProvider` and `OllamaProvider` use `httpx` for REST and SSE/NDJSON streaming. `summarizer.py` public async functions stay as the single pipeline entry points; internally they dispatch to the provider when `LLM_PROVIDER != "gemini"`. A new `/api/capabilities` endpoint advertises provider features so the frontend can hide unsupported modes.

**Tech Stack:** Python 3, FastAPI 0.111, `google-genai`, `httpx==0.27` (already in requirements), pytest 8.2 + pytest-asyncio.

**Spec reference:** `docs/superpowers/specs/2026-04-25-llm-provider-abstraction-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app/services/llm_providers/__init__.py` | Create | `get_provider()` factory based on `settings.llm_provider` |
| `app/services/llm_providers/base.py` | Create | `LLMProvider` ABC, `AudioRef` dataclass, `_with_retry` helper |
| `app/services/llm_providers/errors.py` | Create | Exception hierarchy (`ProviderError` + 4 subclasses) |
| `app/services/llm_providers/gemini.py` | Create | `GeminiProvider` — thin facade over existing summarizer helpers |
| `app/services/llm_providers/openrouter.py` | Create | `OpenRouterProvider` — REST + SSE via httpx |
| `app/services/llm_providers/ollama.py` | Create | `OllamaProvider` — REST + NDJSON via httpx (code-only, not deployed) |
| `app/services/summarizer.py` | Modify | Add provider-dispatch branches in 5 public async functions; keep existing Gemini code path untouched |
| `app/config.py` | Modify | Add `llm_provider`, `openrouter_*`, `ollama_*` fields + startup validator |
| `app/api/routes.py` | Modify | Add `GET /api/capabilities` endpoint |
| `static/index.html` | Modify | On load, fetch `/api/capabilities`, hide `gemini_direct` option if unsupported, show provider badge |
| `.env.example` | Modify | Document new env vars |
| `CLAUDE.md` | Modify | Add LLM Provider section under Architecture |
| `tests/test_llm_providers.py` | Create | Unit tests: factory, error mapping, OpenRouter payload, Ollama payload, unsupported-feature behavior, retry classification |

---

## Task 1: Scaffold `errors.py`

**Files:**
- Create: `app/services/llm_providers/__init__.py` (empty for now)
- Create: `app/services/llm_providers/errors.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Create the empty package**

```bash
mkdir -p app/services/llm_providers
touch app/services/llm_providers/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_llm_providers.py`:

```python
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
```

- [ ] **Step 3: Run the tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.llm_providers.errors'`

- [ ] **Step 4: Implement `errors.py`**

Create `app/services/llm_providers/errors.py`:

```python
"""
Exception hierarchy for LLM providers.

ProviderError is the root; subclasses convey the kind of failure so callers
(or pipeline orchestrators) can decide whether to surface the user_message,
retry, or fail fast.
"""
from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for all LLM provider failures."""

    def __init__(
        self,
        *,
        provider: str,
        stage: str,
        code: str,
        user_message: str,
        technical_details: str = "",
    ) -> None:
        self.provider = provider
        self.stage = stage
        self.code = code
        self.user_message = user_message
        self.technical_details = technical_details
        super().__init__(user_message)


class ProviderUnsupportedError(ProviderError):
    """The provider does not support the requested feature (e.g. audio upload)."""


class ProviderRateLimitError(ProviderError):
    """Quota exceeded / 429 — caller should back off or surface to user."""


class ProviderTimeoutError(ProviderError):
    """Provider did not respond within the configured timeout."""


class ProviderAuthError(ProviderError):
    """API key missing or rejected (401/403). Terminal — do not retry."""
```

- [ ] **Step 5: Run the tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add app/services/llm_providers/__init__.py app/services/llm_providers/errors.py tests/test_llm_providers.py
git commit -m "feat(llm): add ProviderError hierarchy for the new abstraction"
```

---

## Task 2: Scaffold `base.py` — `LLMProvider` ABC + retry helper

**Files:**
- Create: `app/services/llm_providers/base.py`
- Modify: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_llm_providers.py`)

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.llm_providers.base'`

- [ ] **Step 3: Implement `base.py`**

Create `app/services/llm_providers/base.py`:

```python
"""
Abstract LLM provider interface + shared helpers.

Every concrete provider (Gemini, OpenRouter, Ollama) inherits from
LLMProvider. The interface intentionally returns RAW MODEL TEXT for
generation methods — JSON parsing stays in summarizer.py because it is
specific to the LessonResult schema and has accumulated defensive logic
that would not survive being scattered across providers.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderUnsupportedError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Audio reference ────────────────────────────────────────────────────────────

@dataclass
class AudioRef:
    """Reference to an audio file living in a provider's storage."""

    provider_name: str
    provider_specific_id: str
    raw: Any = None  # original SDK object, kept for cleanup calls


# ── Abstract provider ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    name: str
    supports_audio_upload: bool
    supports_streaming: bool
    default_model: str

    # ---- text generation -----------------------------------------------------

    @abstractmethod
    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        """Single-shot generation. Returns raw model text. Caller parses."""

    @abstractmethod
    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Used by chat-with-recording streaming."""

    # ---- audio upload (Gemini-only by default) ------------------------------

    async def upload_audio(self, path: str) -> AudioRef:
        raise ProviderUnsupportedError(
            provider=self.name,
            stage="upload_audio",
            code="unsupported",
            user_message=f"❌ ה-LLM הנוכחי ({self.name}) לא תומך בהעלאת אודיו. בחר מצב עיבוד אחר.",
        )

    async def generate_text_with_audio(
        self,
        audio_ref: AudioRef,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        raise ProviderUnsupportedError(
            provider=self.name,
            stage="generate_text_with_audio",
            code="unsupported",
            user_message=f"❌ ה-LLM הנוכחי ({self.name}) לא תומך בעיבוד אודיו ישיר.",
        )

    async def cleanup_audio(self, audio_ref: AudioRef) -> None:
        """Default no-op; Gemini overrides to delete the Files-API entry."""
        return None


# ── Shared retry helper ───────────────────────────────────────────────────────

async def _with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """
    Exponential-backoff retry for an async function.

    Retries on every ProviderError EXCEPT ProviderAuthError (terminal) and
    ProviderUnsupportedError (terminal). Re-raises on final attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except (ProviderAuthError, ProviderUnsupportedError):
            raise
        except ProviderError as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(
                f"Provider error (attempt {attempt + 1}/{max_retries}): "
                f"{exc.code} — retrying in {wait}s"
            )
            await asyncio.sleep(wait)
    # Unreachable — loop either returns or raises
    raise last_exc or RuntimeError("retry loop exited unexpectedly")
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/llm_providers/base.py tests/test_llm_providers.py
git commit -m "feat(llm): add LLMProvider ABC, AudioRef, and shared retry helper"
```

---

## Task 3: Add config fields and startup validator

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_llm_providers.py`)

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py::test_settings_default_provider_is_gemini -v`
Expected: `AttributeError: 'Settings' object has no attribute 'llm_provider'`

- [ ] **Step 3: Modify `app/config.py`**

Add the import at the top of the file (after the existing imports):

```python
from typing import Literal
```

Add the fields inside the `Settings` class — place them right after the `# ── Auth ──` block (the section ending with `cors_origin: str = ...`), before `settings = Settings()`:

```python
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
```

Add a startup validator method inside `Settings` (after the existing `derive_downloads_dir` validator):

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: 10 passed (4 new + 6 existing)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_llm_providers.py
git commit -m "feat(config): add LLM_PROVIDER + provider-specific credentials"
```

---

## Task 4: Implement `get_provider()` factory

**Files:**
- Modify: `app/services/llm_providers/__init__.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py::test_get_provider_returns_gemini_by_default -v`
Expected: `ImportError: cannot import name 'get_provider'`

- [ ] **Step 3: Implement the factory**

Replace `app/services/llm_providers/__init__.py` with:

```python
"""
Factory for LLM providers.

`get_provider()` returns a singleton matching `settings.llm_provider`. The
singleton is rebuilt only when `_reset_provider_cache()` is called (used by
tests; production code calls get_provider() repeatedly without overhead).
"""
from __future__ import annotations

from app.config import settings

from .base import AudioRef, LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnsupportedError,
)

__all__ = [
    "AudioRef",
    "LLMProvider",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnsupportedError",
    "get_provider",
]


_cached: LLMProvider | None = None


def _reset_provider_cache() -> None:
    """Clear the cached provider — used by tests when changing settings."""
    global _cached
    _cached = None


def get_provider() -> LLMProvider:
    """Return the active LLMProvider instance based on settings.llm_provider."""
    global _cached
    if _cached is not None:
        return _cached

    name = settings.llm_provider
    if name == "gemini":
        from .gemini import GeminiProvider
        _cached = GeminiProvider()
    elif name == "openrouter":
        from .openrouter import OpenRouterProvider
        _cached = OpenRouterProvider()
    elif name == "ollama":
        from .ollama import OllamaProvider
        _cached = OllamaProvider()
    else:
        raise ValueError(f"Unknown llm_provider: {name!r}")

    return _cached
```

- [ ] **Step 4: Run tests — expect failure (provider modules not yet implemented)**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ImportError` from `from .gemini import GeminiProvider`

That's fine — Tasks 5-7 implement those. Skip the factory tests for now and continue.

- [ ] **Step 5: Commit the factory scaffolding**

```bash
git add app/services/llm_providers/__init__.py tests/test_llm_providers.py
git commit -m "feat(llm): add get_provider() factory + reset hook for tests"
```

---

## Task 5: Implement `GeminiProvider` (facade over existing summarizer code)

**Files:**
- Create: `app/services/llm_providers/gemini.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.llm_providers.gemini'`

- [ ] **Step 3: Implement `gemini.py`**

Create `app/services/llm_providers/gemini.py`:

```python
"""
Gemini provider — a thin facade over the existing helpers in
`app.services.summarizer`. The goal of this module is to expose Gemini
through the `LLMProvider` interface WITHOUT changing summarizer.py — every
existing test that monkeypatches summarizer internals continues to work.

We keep the sync google-genai SDK on a thread executor (as summarizer.py
already does) and translate Gemini-specific exceptions into ProviderError
subclasses on the way out.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from app.config import settings

from .base import AudioRef, LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _classify_gemini_exception(exc: Exception) -> ProviderError:
    """Map a raw Gemini SDK / RuntimeError into a ProviderError subclass."""
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "quota" in low or "rate limit" in low or "מכסת" in msg:
        return ProviderRateLimitError(
            provider="gemini",
            stage="generate",
            code="rate_limit",
            user_message="⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות",
            technical_details=msg,
        )
    if "401" in msg or "403" in msg or "api key" in low:
        return ProviderAuthError(
            provider="gemini",
            stage="generate",
            code="auth",
            user_message="🔑 מפתח ה-API של Gemini שגוי או לא הוגדר",
            technical_details=msg,
        )
    return ProviderError(
        provider="gemini",
        stage="generate",
        code="unknown",
        user_message=f"שגיאה ב-Gemini: {msg[:200]}",
        technical_details=msg,
    )


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_audio_upload = True
    supports_streaming = True
    default_model = settings.gemini_model

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        """Delegate to summarizer._generate_with_retry on a thread executor."""
        from app.services import summarizer  # local import — avoids cycle

        loop = asyncio.get_running_loop()

        def _call() -> str:
            try:
                client = summarizer._get_client()
                resp = summarizer._generate_with_retry(client, prompt)
                return summarizer._response_text(resp)
            except Exception as exc:
                raise _classify_gemini_exception(exc)

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                provider="gemini",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ Gemini לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Delegate to the existing streaming code in summarizer.py."""
        from app.services import summarizer

        # summarizer.stream_chat_response expects (context, history, question).
        # GeminiProvider's interface is more general (raw `contents`), so for
        # now we call the summarizer streaming function directly when invoked
        # via the chat path. For the facade behavior, providers that want to
        # bypass summarizer's chat helper can call generate_content_stream
        # themselves; we keep this method as a thin pass-through.
        from google import genai
        client = summarizer._get_client()

        loop = asyncio.get_running_loop()
        import queue as _q
        q: _q.Queue = _q.Queue()

        def _run() -> None:
            try:
                for chunk in client.models.generate_content_stream(
                    model=settings.gemini_model,
                    contents=contents,
                ):
                    text = getattr(chunk, "text", None)
                    if text:
                        q.put(text)
            except Exception as exc:
                q.put(_classify_gemini_exception(exc))
            finally:
                q.put(None)

        loop.run_in_executor(None, _run)

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def upload_audio(self, path: str) -> AudioRef:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _upload():
            client = summarizer._get_client()
            f = client.files.upload(file=path)
            # Poll until processed (mirrors summarizer._summarize_audio_sync)
            import time
            waited = 0
            while f.state.name == "PROCESSING":
                if waited >= 300:
                    raise RuntimeError("⏱️ Gemini לא סיים לעבד את קובץ האודיו תוך 5 דקות")
                time.sleep(5)
                waited += 5
                f = client.files.get(name=f.name)
            if f.state.name == "FAILED":
                raise RuntimeError("❌ Gemini נכשל בעיבוד קובץ האודיו")
            return f

        try:
            f = await loop.run_in_executor(None, _upload)
        except Exception as exc:
            raise _classify_gemini_exception(exc)

        return AudioRef(provider_name="gemini", provider_specific_id=f.name, raw=f)

    async def generate_text_with_audio(
        self,
        audio_ref: AudioRef,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _call() -> str:
            try:
                client = summarizer._get_client()
                resp = summarizer._generate_with_retry(client, [prompt, audio_ref.raw])
                return summarizer._response_text(resp)
            except Exception as exc:
                raise _classify_gemini_exception(exc)

        return await loop.run_in_executor(None, _call)

    async def cleanup_audio(self, audio_ref: AudioRef) -> None:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _delete() -> None:
            try:
                client = summarizer._get_client()
                client.files.delete(name=audio_ref.provider_specific_id)
            except Exception as exc:
                logger.warning(f"Gemini audio cleanup failed: {exc}")

        await loop.run_in_executor(None, _delete)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: 16 passed (factory tests now pass too because GeminiProvider exists)

- [ ] **Step 5: Run all existing tests — expect zero regressions**

Run: `pytest tests/ -v`
Expected: All previously-passing tests still pass. The Gemini code path in summarizer.py is untouched.

- [ ] **Step 6: Commit**

```bash
git add app/services/llm_providers/gemini.py tests/test_llm_providers.py
git commit -m "feat(llm): add GeminiProvider as facade over existing summarizer helpers"
```

---

## Task 6: Implement `OpenRouterProvider`

**Files:**
- Create: `app/services/llm_providers/openrouter.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.llm_providers.openrouter'`

- [ ] **Step 3: Implement `openrouter.py`**

Create `app/services/llm_providers/openrouter.py`:

```python
"""
OpenRouter provider — REST + SSE via httpx.

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint. We use
the `messages` array for both single-shot and multi-turn calls, and SSE
(`stream=true`) for the chat panel.

Audio upload is not supported — caller falls back to one of the WHISPER
modes for transcription, then sends the text via generate_text.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import settings

from .base import LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _classify_http_error(status: int, body: str) -> ProviderError:
    if status in (401, 403):
        return ProviderAuthError(
            provider="openrouter",
            stage="generate",
            code=str(status),
            user_message="🔑 מפתח ה-API של OpenRouter שגוי או לא הוגדר",
            technical_details=body[:300],
        )
    if status == 429:
        return ProviderRateLimitError(
            provider="openrouter",
            stage="generate",
            code="429",
            user_message="⚠️ מכסת ה-API של OpenRouter הוצתה — נסה שוב בעוד כמה דקות",
            technical_details=body[:300],
        )
    return ProviderError(
        provider="openrouter",
        stage="generate",
        code=str(status),
        user_message=f"שגיאה מ-OpenRouter ({status})",
        technical_details=body[:300],
    )


def _contents_to_messages(contents: str | list) -> list[dict]:
    """Convert prompt|contents into OpenAI-style messages."""
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    out = []
    for c in contents:
        # google-genai-style {"role": "user|model", "parts": [{"text": ...}]}
        role = c.get("role", "user")
        if role == "model":
            role = "assistant"
        text_parts = [p.get("text", "") for p in c.get("parts", []) if "text" in p]
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    supports_audio_upload = False
    supports_streaming = True
    default_model = ""  # populated in __init__ from settings

    def __init__(self) -> None:
        self.default_model = settings.openrouter_model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends these for analytics
            "HTTP-Referer": settings.base_url,
            "X-Title": settings.app_title,
        }

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        url = f"{settings.openrouter_base_url}/chat/completions"
        payload = {
            "model": settings.openrouter_model,
            "messages": _contents_to_messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                if resp.status_code >= 400:
                    raise _classify_http_error(resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="openrouter",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ OpenRouter לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )

        return data["choices"][0]["message"]["content"]

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        url = f"{settings.openrouter_base_url}/chat/completions"
        payload = {
            "model": settings.openrouter_model,
            "messages": _contents_to_messages(contents),
            "temperature": temperature,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=self._headers()
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise _classify_http_error(resp.status_code, body.decode("utf-8", "ignore"))
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {}).get("content")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="openrouter",
                stage="stream_text",
                code="timeout",
                user_message="⏱️ OpenRouter לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: All tests pass (19 total).

- [ ] **Step 5: Commit**

```bash
git add app/services/llm_providers/openrouter.py tests/test_llm_providers.py
git commit -m "feat(llm): add OpenRouterProvider with REST + SSE streaming"
```

---

## Task 7: Implement `OllamaProvider`

**Files:**
- Create: `app/services/llm_providers/ollama.py`
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `ollama.py`**

Create `app/services/llm_providers/ollama.py`:

```python
"""
Ollama provider — REST + NDJSON streaming via httpx.

Ollama runs models locally (llama3.1, mistral, qwen, etc.). We hit
/api/generate for single-shot text and /api/chat with stream=true for the
chat panel. NDJSON: each line is a JSON object with `response` (text delta)
and `done` (final flag).

This provider is code-only on the current Fly.io deploy — Ollama needs
local GPU / sizable RAM. It's wired to be ready for a future home-server
deploy with no further code changes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import settings

from .base import LLMProvider
from .errors import (
    ProviderError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _contents_to_messages(contents: str | list) -> list[dict]:
    """google-genai-style contents → Ollama chat messages."""
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    out = []
    for c in contents:
        role = c.get("role", "user")
        if role == "model":
            role = "assistant"
        parts = [p.get("text", "") for p in c.get("parts", []) if "text" in p]
        out.append({"role": role, "content": "\n".join(parts)})
    return out


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_audio_upload = False
    supports_streaming = True
    default_model = ""

    def __init__(self) -> None:
        self.default_model = settings.ollama_model

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        url = f"{settings.ollama_base_url}/api/generate"
        # Ollama /api/generate takes a flat prompt; for multi-turn use /api/chat
        if isinstance(prompt, list):
            # Flatten contents into a single prompt string
            messages = _contents_to_messages(prompt)
            prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        else:
            prompt_text = prompt

        payload = {
            "model": settings.ollama_model,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    raise ProviderError(
                        provider="ollama",
                        stage="generate",
                        code=str(resp.status_code),
                        user_message=f"שגיאה מ-Ollama ({resp.status_code})",
                        technical_details=resp.text[:300],
                    )
                data = resp.json()
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="ollama",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ Ollama לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
        except httpx.ConnectError as exc:
            raise ProviderError(
                provider="ollama",
                stage="generate_text",
                code="connect",
                user_message="❌ לא הצלחתי להתחבר לשרת Ollama — ודא שהוא רץ",
                technical_details=str(exc),
            )

        return data.get("response", "")

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": _contents_to_messages(contents),
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise ProviderError(
                            provider="ollama",
                            stage="stream_text",
                            code=str(resp.status_code),
                            user_message=f"שגיאה מ-Ollama ({resp.status_code})",
                            technical_details=body.decode("utf-8", "ignore")[:300],
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("done"):
                            break
                        delta = chunk.get("message", {}).get("content")
                        if delta:
                            yield delta
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="ollama",
                stage="stream_text",
                code="timeout",
                user_message="⏱️ Ollama לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: All tests pass (21 total).

- [ ] **Step 5: Commit**

```bash
git add app/services/llm_providers/ollama.py tests/test_llm_providers.py
git commit -m "feat(llm): add OllamaProvider (code-only — not deployed on Fly.io)"
```

---

## Task 8: Wire `summarizer.py` to dispatch on provider

**Files:**
- Modify: `app/services/summarizer.py`
- Test: `tests/test_llm_providers.py`

**Goal:** When `LLM_PROVIDER != "gemini"`, the public async functions delegate to `get_provider()` instead of running the original Gemini code path. When `LLM_PROVIDER == "gemini"` (default), the original code path runs unchanged so all existing tests keep passing.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_llm_providers.py -v`
Expected: `AssertionError` from `fail_gemini_path` (current code ignores `llm_provider` and always uses Gemini).

- [ ] **Step 3: Modify `summarizer.py` — add a small dispatch helper**

Open `app/services/summarizer.py`. Add this import at the top with the other imports (after the existing `from app.models import ...` line):

```python
from app.services.llm_providers import get_provider
```

Add this helper near the top of the file (after the `_ProgressCallback` line, before `# ── Prompt ──`):

```python
def _is_gemini_provider() -> bool:
    """True if the current provider is Gemini (use existing code path)."""
    return settings.llm_provider == "gemini"
```

- [ ] **Step 4: Modify `summarize_audio` to dispatch on provider**

Replace the current `summarize_audio` async function (around lines 743-755) with:

```python
async def summarize_audio(
    audio_path: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Async: upload audio directly to LLM (GEMINI_DIRECT mode)."""
    if not _is_gemini_provider():
        # Non-Gemini providers do not support audio upload — fail fast with
        # a clear UX message. The caller (processor.py) maps this to user
        # error in the SQLite error_details column.
        provider = get_provider()
        # The default upload_audio() raises ProviderUnsupportedError with a
        # Hebrew user_message. Trigger it explicitly so the UX is right.
        await provider.upload_audio(audio_path)
        # Unreachable — upload_audio raised
        raise RuntimeError("unreachable")

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _summarize_audio_sync, audio_path, progress_cb),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")
```

- [ ] **Step 5: Modify `summarize_transcript` to dispatch on provider**

Replace the current `summarize_transcript` async function with:

```python
async def summarize_transcript(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Async: summarize a text transcript (WHISPER_LOCAL / WHISPER_API mode)."""
    if not _is_gemini_provider():
        return await _summarize_transcript_via_provider(transcript, progress_cb)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _summarize_text_sync, transcript, progress_cb),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")


async def _summarize_transcript_via_provider(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Run the full transcript→summary pipeline through a non-Gemini provider."""
    provider = get_provider()

    if progress_cb:
        progress_cb(82, f"🤖 שולח ל-{provider.name} — מייצר סיכום ומבחן...")

    prompt = f"{_SYSTEM_PROMPT}\n\nתמלול השיעור:\n{transcript[:_MAX_CHUNK_CHARS]}"
    text = await provider.generate_text(prompt)
    result = _parse_response(text)

    # Reuse the existing critique pipeline — but only for Gemini, since
    # critique_exam/revise_exam call Gemini directly. For non-Gemini
    # providers we skip critique to avoid mixing backends in one task.
    return result
```

- [ ] **Step 6: Modify `ask_about_lesson` to dispatch**

Replace the current `ask_about_lesson` async function with:

```python
async def ask_about_lesson(context: str, question: str) -> str:
    """Async: answer a student question based on the lesson content."""
    if not _is_gemini_provider():
        provider = get_provider()
        prompt = f"{_ASK_SYSTEM_PROMPT}\n\nתוכן השיעור:\n{context}\n\nשאלת התלמיד: {question}"
        return await provider.generate_text(prompt, timeout=_ASK_TIMEOUT)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _ask_sync, context, question),
            timeout=_ASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 2 דקות — נסה שוב")
```

- [ ] **Step 7: Modify `stream_chat_response` to dispatch**

Replace the current `stream_chat_response` async generator with:

```python
async def stream_chat_response(context: str, history: list[dict], question: str):
    """
    Async generator that yields text chunks from a streaming LLM chat call.

    For Gemini: uses the original sync-SDK + thread-queue bridge.
    For OpenRouter / Ollama: uses the provider's native streaming.
    """
    if not _is_gemini_provider():
        provider = get_provider()
        contents = _build_chat_contents(context, history, question)
        async for chunk in provider.stream_text(contents):
            yield chunk
        return

    # Original Gemini path — UNCHANGED (matches existing tests)
    import queue as _q_mod

    loop = asyncio.get_running_loop()
    q: _q_mod.Queue = _q_mod.Queue()

    loop.run_in_executor(None, _stream_chat_sync, context, history, question, q)

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item
```

- [ ] **Step 8: Modify `generate_flashcards` to dispatch**

Replace the current `generate_flashcards` async function with:

```python
async def generate_flashcards(
    summary: str,
    transcript: str | None = None,
) -> list[Flashcard]:
    """Async: generate 15-25 flashcards from a lesson summary (+ optional transcript)."""
    if not summary.strip():
        return []

    if not _is_gemini_provider():
        provider = get_provider()
        # Build the same context that _generate_flashcards_sync builds
        context_parts = [f"סיכום השיעור:\n{summary}"]
        if transcript:
            context_parts.append(f"\nקטע מהתמלול:\n{transcript[:30_000]}")
        joined_context = "\n\n".join(context_parts)
        prompt = f"{_FLASHCARDS_PROMPT}\n\n{joined_context}"
        try:
            text = await asyncio.wait_for(
                provider.generate_text(prompt),
                timeout=_FLASHCARDS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Flashcards generation timed out — returning empty list")
            return []
        return _parse_flashcards_response(text)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _generate_flashcards_sync, summary, transcript),
            timeout=_FLASHCARDS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Flashcards generation timed out — returning empty list")
        return []
```

- [ ] **Step 9: Run all existing tests — expect zero regressions**

Run: `pytest tests/ -v`
Expected: All previously-passing tests still pass (default `llm_provider=gemini` → original code paths run untouched).

- [ ] **Step 10: Run new tests — expect pass**

Run: `pytest tests/test_llm_providers.py -v`
Expected: All tests pass (23 total).

- [ ] **Step 11: Commit**

```bash
git add app/services/summarizer.py tests/test_llm_providers.py
git commit -m "feat(summarizer): dispatch to LLM provider when LLM_PROVIDER!=gemini"
```

---

## Task 9: Add `/api/capabilities` endpoint

**Files:**
- Modify: `app/api/routes.py`
- Create: `tests/test_capabilities.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_capabilities.py`:

```python
"""Tests for GET /api/capabilities."""
import pytest


def test_capabilities_returns_gemini_defaults(client):
    """With default settings, the endpoint reports Gemini + all four modes."""
    # The /api/capabilities endpoint is public (no auth) so the UI can read
    # provider info before the user logs in. Test does not need the auth
    # header that the protected endpoints require.
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "gemini"
    assert data["supports_audio_upload"] is True
    assert data["supports_streaming"] is True
    assert "gemini_direct" in data["available_modes"]
    assert "whisper_local" in data["available_modes"]
    assert "whisper_api" in data["available_modes"]
    assert "ivrit_ai" in data["available_modes"]


def test_capabilities_hides_gemini_direct_for_openrouter(client, monkeypatch):
    """When provider can't upload audio, gemini_direct mode is excluded."""
    from app.config import settings
    from app.services.llm_providers import _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)
    _reset_provider_cache()

    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "openrouter"
    assert data["supports_audio_upload"] is False
    assert "gemini_direct" not in data["available_modes"]
    assert "whisper_local" in data["available_modes"]
```

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_capabilities.py -v`
Expected: 404 (endpoint does not exist yet).

- [ ] **Step 3: Add the endpoint to `app/api/routes.py`**

Add this import near the top of `app/api/routes.py` (with the other `app.services` imports):

```python
from app.services.llm_providers import get_provider
```

Add this code at the end of the file (after the last endpoint):

```python
# ── Provider capabilities (UI uses this to filter the mode dropdown) ──────────

_ALL_MODES = ["gemini_direct", "whisper_local", "whisper_api", "ivrit_ai"]


def _available_modes_for(provider) -> list[str]:
    """Filter the four processing modes by what the provider can actually do."""
    if provider.supports_audio_upload:
        return list(_ALL_MODES)
    # Without audio upload, the only viable modes are the transcribe-then-summarize ones
    return [m for m in _ALL_MODES if m != "gemini_direct"]


@router.get("/capabilities")
async def get_capabilities() -> dict:
    """
    Public read-only endpoint describing the active LLM provider's capabilities.

    The frontend calls this on page load to know which processing modes to
    offer. No auth required — the response is configuration metadata, not
    user data. Failures here are very rare (factory raises ValueError on a
    bad llm_provider env value), so we let the framework return 500 in that
    edge case rather than masking a real config bug.
    """
    p = get_provider()
    return {
        "llm_provider": p.name,
        "supports_audio_upload": p.supports_audio_upload,
        "supports_streaming": p.supports_streaming,
        "available_modes": _available_modes_for(p),
    }
```

- [ ] **Step 4: Run tests — expect pass**

Run: `pytest tests/test_capabilities.py -v`
Expected: 2 passed

- [ ] **Step 5: Run full suite — expect zero regressions**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py tests/test_capabilities.py
git commit -m "feat(api): add GET /api/capabilities for UI mode filtering"
```

---

## Task 10: Frontend — capabilities check + provider badge

**Files:**
- Modify: `static/index.html`

This task has no automated test — the UI is verified via manual smoke test in Task 12.

- [ ] **Step 1: Locate the mode dropdown in `static/index.html`**

Run: `grep -n "gemini_direct\|whisper_local\|<select" static/index.html | head -30`

This shows the lines containing the mode `<option>` values. Note the line numbers of:
- The `<select>` element for processing mode
- Each of the four `<option value="..."` lines

- [ ] **Step 2: Add a provider badge container**

In the HTML body, near the top of the form area (just above the mode dropdown), add this badge element:

```html
<div id="provider-badge" class="provider-badge" style="display:none;">
    LLM: <span id="provider-name"></span>
</div>
```

Add the matching CSS to the existing `<style>` block:

```css
.provider-badge {
    display: inline-block;
    padding: 2px 10px;
    margin-bottom: 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
```

- [ ] **Step 3: Add the JS bootstrap call**

In the `<script>` block, near the existing `DOMContentLoaded` or page-init code, add:

```javascript
// ── Provider capabilities (filters the mode dropdown) ──────────────────────
async function loadCapabilities() {
    try {
        const r = await fetch('/api/capabilities');
        if (!r.ok) return;
        const cap = await r.json();
        // Show provider badge
        const badge = document.getElementById('provider-badge');
        const nameEl = document.getElementById('provider-name');
        if (badge && nameEl) {
            nameEl.textContent = cap.llm_provider;
            badge.style.display = 'inline-block';
        }
        // Filter the mode dropdown
        const select = document.querySelector('select[name="mode"]');
        if (select && Array.isArray(cap.available_modes)) {
            const allowed = new Set(cap.available_modes);
            Array.from(select.options).forEach(opt => {
                if (!allowed.has(opt.value)) opt.remove();
            });
            // If the current selection got removed, fall back to the first option
            if (!allowed.has(select.value) && select.options.length) {
                select.value = select.options[0].value;
            }
        }
    } catch (e) {
        // Non-critical — UI just keeps showing all options
        console.warn('capabilities fetch failed', e);
    }
}
document.addEventListener('DOMContentLoaded', loadCapabilities);
```

If the existing JS already has a `DOMContentLoaded` listener that runs the page boot, append `loadCapabilities();` inside it instead of registering a second listener.

- [ ] **Step 4: Manual sanity check**

Run: `docker compose up -d --build`
Open: `http://localhost:8000` in a browser.
Expected: page loads, badge shows `LLM: GEMINI`, dropdown still has all four modes.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat(ui): hide unsupported modes + show provider badge"
```

---

## Task 11: Update `.env.example` and `CLAUDE.md`

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Append the new env vars to `.env.example`**

Add these lines at the end of `.env.example`:

```
# ── LLM Provider Selection ────────────────────────────────────
# Which LLM handles summarization, critique, chat, and flashcards.
# Options: gemini | openrouter | ollama
LLM_PROVIDER=gemini

# OpenRouter (https://openrouter.ai) — optional, only needed if
# LLM_PROVIDER=openrouter. Get a key at openrouter.ai/keys.
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Ollama (https://ollama.com) — local model runner, optional.
# Only used if LLM_PROVIDER=ollama. Not deployed on Fly.io.
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:70b
```

- [ ] **Step 2: Add an LLM Provider section to `CLAUDE.md`**

Find the `## Architecture` section in `CLAUDE.md`. Add this subsection at the end of it (just before the next top-level `##` heading):

```markdown
### LLM Provider Abstraction
The summarizer + chat + flashcards subsystem is provider-agnostic. Set
`LLM_PROVIDER` in `.env` to switch backends:
- `gemini` (default) — Google Gemini 2.5 Flash via google-genai SDK.
  Supports audio upload (`GEMINI_DIRECT` mode).
- `openrouter` — OpenAI-compatible REST API. No audio upload — use
  WHISPER modes for transcription, then summarize.
- `ollama` — Local model runner. Code-only on Fly.io (deps too heavy);
  ready for a home-server deploy.

Code lives in `app/services/llm_providers/`. The factory `get_provider()`
returns the active backend; `summarizer.py` dispatches to it when
`LLM_PROVIDER != "gemini"` and otherwise runs the original Gemini code
path (so existing tests keep passing).

The frontend calls `GET /api/capabilities` on page load to learn what
the active provider supports, and hides processing modes that wouldn't
work (e.g. `gemini_direct` is hidden when `LLM_PROVIDER=ollama`).
```

- [ ] **Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: document LLM_PROVIDER env var and capabilities endpoint"
```

---

## Task 12: Smoke test + `/review`

**Files:** none modified

- [ ] **Step 1: Smoke test default (Gemini) deploy**

Run: `docker compose up -d --build`

Wait ~60s for the app to come up. Then:

Run: `curl -s http://localhost:8000/health`
Expected: `200 OK` JSON.

Run: `curl -s http://localhost:8000/api/capabilities`
Expected: `{"llm_provider":"gemini","supports_audio_upload":true,...}`

Open `http://localhost:8000` in a browser, log in, upload a 30-second test audio, pick `GEMINI_DIRECT`, and confirm a `LessonResult` is produced.

- [ ] **Step 2: Smoke test OpenRouter (manual, optional — needs a real key)**

If you have an OpenRouter API key handy:

```bash
echo "LLM_PROVIDER=openrouter" >> .env
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
docker compose up -d --build
```

Run: `curl -s http://localhost:8000/api/capabilities`
Expected: `{"llm_provider":"openrouter","supports_audio_upload":false,...}`

Open the UI — confirm `gemini_direct` is missing from the dropdown and the badge shows `LLM: OPENROUTER`. Upload a short audio with `WHISPER_LOCAL` mode and confirm a result is produced via OpenRouter.

Restore `LLM_PROVIDER=gemini` afterward.

- [ ] **Step 3: Run the full test suite one more time**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new).

- [ ] **Step 4: Run `/review`**

In the Claude Code session: `/review`

Address any high or medium severity findings before merging. Low severity findings can be triaged.

- [ ] **Step 5: Final commit (only if review surfaced fixes)**

If any review-driven fixes were made:

```bash
git add <files>
git commit -m "fix: address /review findings on llm-provider-abstraction"
```

If no fixes needed, skip this step.

---

## Definition of Done (from spec)

- [ ] All existing tests pass with `LLM_PROVIDER=gemini` (default), zero edits to those tests
- [ ] `tests/test_llm_providers.py` passes
- [ ] `tests/test_capabilities.py` passes
- [ ] `docker compose up -d --build` boots and `/health` is green
- [ ] Manual end-to-end run via UI in `gemini_direct` mode produces a `LessonResult`
- [ ] `/api/capabilities` returns the expected payload for both `gemini` and (when configured) `openrouter`
- [ ] `/review` finds no high or medium severity issues (or any are addressed)
- [ ] `CLAUDE.md` updated with the LLM Provider section
- [ ] `.env.example` updated with the new vars
