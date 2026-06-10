# LLM Provider Abstraction — Design Spec

**Date:** 2026-04-25
**Cycle:** A (foundation for the larger upgrade plan)
**Status:** Approved by user (Asaf), pending implementation
**Source plan:** `UPGRADE_PROMPT.md` Task 1.3

---

## 1. Goal

Make the LLM backend swappable via a single environment variable
(`LLM_PROVIDER=gemini|openrouter|ollama`), with **zero changes** to any other
file in the project. All six existing call patterns currently coupled to
`google-genai` continue to work unchanged from a caller's perspective.

The six call patterns being abstracted:

1. **Direct audio summarization** — `summarize_audio()` uploads an audio file
   to Gemini's Files API and asks for a structured `LessonResult`.
2. **Transcript summarization** — `summarize_transcript()` sends text and
   chunk-merges if it exceeds 350k chars.
3. **Exam critique + revise** — `critique_exam()` and `revise_exam()` produce
   structured JSON for the quality pipeline.
4. **Single-shot Q&A** — `ask_about_lesson()`.
5. **Multi-turn streaming chat** — `stream_chat_response()` yields text deltas.
6. **Flashcards generation** — `generate_flashcards()` JSON.

## 2. Non-goals

Out of scope for Cycle A:

- Per-call provider override (e.g. chat=Claude, summary=Gemini).
- Cost tracking / billing per provider.
- Response caching layer.
- Embeddings provider abstraction.
- Migrating any data stored in SQLite.
- Changes to the Whisper / ivrit-ai transcription path (those are not LLM
  calls — they're ASR).

## 3. Architecture

```
app/services/llm_providers/
├── __init__.py        # get_provider() factory
├── base.py            # LLMProvider ABC, AudioRef dataclass, retry helper
├── errors.py          # ProviderError + subclasses
├── gemini.py          # GeminiProvider — wraps current google-genai code
├── openrouter.py      # OpenRouterProvider — text + streaming via httpx
└── ollama.py          # OllamaProvider — text + streaming via httpx (local)
```

`summarizer.py` continues to expose the same public async functions and is
the only file in the **processing pipeline** that talks to `get_provider()`.
The single other consumer is `routes.py`, which calls `get_provider()` from
the read-only `/api/capabilities` endpoint (§8) to advertise provider
capabilities to the UI. `processor.py` and the tests are untouched.

### 3.1 Module responsibilities

| Module | Owns | Does NOT own |
|---|---|---|
| `summarizer.py` | Prompts, JSON parsing for `LessonResult`/`Flashcard`, chunking logic, public async API, `_apply_critique_pipeline` orchestration | Network calls, retry, model names |
| `llm_providers/base.py` | Abstract interface, retry-with-backoff helper, `AudioRef` dataclass | Prompts, parsing |
| `llm_providers/gemini.py` | `google-genai` SDK calls, Files API, `generate_content_stream` | Anything else |
| `llm_providers/openrouter.py` | OpenRouter REST + SSE via httpx | Audio upload (raises `ProviderUnsupportedError`) |
| `llm_providers/ollama.py` | Ollama REST + streaming via httpx | Audio upload (raises `ProviderUnsupportedError`) |

## 4. Interface — `LLMProvider`

```python
# app/services/llm_providers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Any

@dataclass
class AudioRef:
    """Reference to an audio file uploaded to a provider's storage."""
    provider_name: str          # "gemini", etc.
    provider_specific_id: str   # e.g. Gemini files/abc123 name
    raw: Any = None             # Original SDK object (kept for cleanup calls)

class LLMProvider(ABC):
    name: str                       # "gemini" | "openrouter" | "ollama"
    supports_audio_upload: bool
    supports_streaming: bool
    default_model: str

    @abstractmethod
    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        """Single-shot text generation. Returns raw model text. Caller parses."""

    @abstractmethod
    async def stream_text(
        self,
        contents: list[dict],          # [{"role":"user"|"model","parts":[{"text":...}]}]
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Used by chat-with-recording panel."""

    async def upload_audio(self, path: str) -> AudioRef:
        """Default: raise ProviderUnsupportedError. Gemini overrides."""
        raise ProviderUnsupportedError(self.name, "audio_upload")

    async def generate_text_with_audio(
        self,
        audio_ref: AudioRef,
        prompt: str,
        **kwargs,
    ) -> str:
        """Default: raise. Gemini overrides for GEMINI_DIRECT pipeline."""
        raise ProviderUnsupportedError(self.name, "audio_with_text")

    async def cleanup_audio(self, audio_ref: AudioRef) -> None:
        """Default: no-op. Gemini overrides to delete the uploaded file."""
        return None
```

### 4.1 Why no `generate_json()` method

The current code does *not* use Gemini's structured-output / response-schema
feature — it asks for "raw JSON, no fences" in the prompt and parses with
defensive regex (`_sanitize_json_escapes`, `_response_text`). That parser
handles real-world quirks (thinking preamble, stray backslashes, code-fence
leakage). Moving JSON parsing into the provider would either duplicate that
logic per provider or push the parser down where it doesn't belong.

**Decision:** the provider returns raw text. `summarizer.py` parses. The
existing `_parse_response()` and `_parse_flashcards_response()` stay.

## 5. Provider behavior matrix

| Capability | Gemini | OpenRouter | Ollama |
|---|---|---|---|
| `generate_text` | ✅ `client.models.generate_content` | ✅ `POST /chat/completions` | ✅ `POST /api/generate` |
| `stream_text` | ✅ `generate_content_stream` | ✅ SSE on `/chat/completions` | ✅ NDJSON on `/api/chat` |
| `upload_audio` | ✅ Files API | ❌ `ProviderUnsupportedError` | ❌ `ProviderUnsupportedError` |
| `generate_text_with_audio` | ✅ | ❌ raises | ❌ raises |
| `cleanup_audio` | ✅ deletes Files-API file | no-op | no-op |
| Default model | `gemini-2.5-flash` | `anthropic/claude-3.5-sonnet` | `llama3.1:70b` |

When `LLM_PROVIDER=openrouter|ollama` and the user picks `GEMINI_DIRECT` mode,
the request fails fast at the API boundary (see §8 UX guard).

## 6. Configuration changes

### 6.1 `app/config.py` additions

```python
# ── LLM provider selection ─────────────────────────────────────────────────
llm_provider: Literal["gemini", "openrouter", "ollama"] = "gemini"

# OpenRouter
openrouter_api_key:  str = ""
openrouter_model:    str = "anthropic/claude-3.5-sonnet"
openrouter_base_url: str = "https://openrouter.ai/api/v1"

# Ollama (local — not deployed on Fly.io)
ollama_base_url: str = "http://localhost:11434"
ollama_model:    str = "llama3.1:70b"
```

### 6.2 Startup validator

`@model_validator(mode="after")` raises a clear error if:
- `llm_provider=openrouter` and `openrouter_api_key` is empty
- `llm_provider=gemini` and both `google_api_key` is empty AND
  `google_application_credentials` file does not exist (already implicitly
  enforced today; we make it explicit at startup not at first call).

### 6.3 `.env.example` updates

Add the new keys with sensible defaults and a one-line comment.

## 7. Wiring `summarizer.py` into the abstraction

The change is mechanical. Every place that today calls `_get_client()` and
then `client.models.generate_content(...)` becomes a call to
`provider.generate_text(...)`. Specifically:

| Current code | New code |
|---|---|
| `client = _get_client()` | `provider = get_provider()` |
| `response = _generate_with_retry(client, contents)` | `text = await provider.generate_text(contents)` |
| `text = _response_text(response)` | (removed — provider returns clean text) |
| `client.files.upload(file=...)` | `await provider.upload_audio(...)` |
| `client.files.delete(name=...)` | `await provider.cleanup_audio(ref)` |
| `client.models.generate_content_stream(...)` | `async for chunk in provider.stream_text(contents):` |

The retry loop currently in `_generate_with_retry()` moves into
`base.py:_with_retry()` and is reused by all providers. Each provider
classifies its own error codes (HTTP 429 vs Gemini quota strings, etc.) but
the retry policy is shared.

The thread-pool pattern (`run_in_executor`) used today to bridge sync SDK
calls into async stays inside the *Gemini* provider — it's a Gemini SDK
quirk. OpenRouter and Ollama use httpx natively and need no executor.

After refactor, `summarizer.py` is **provider-agnostic**: the prompts,
chunking, JSON parsing, and critique/revise orchestration are unchanged.

## 8. UX guard — `/api/capabilities` endpoint + frontend dropdown filter

### 8.1 Backend

Add a small read-only endpoint:

```python
# app/api/routes.py
@router.get("/capabilities")
async def get_capabilities() -> dict:
    p = get_provider()
    return {
        "llm_provider": p.name,
        "supports_audio_upload": p.supports_audio_upload,
        "supports_streaming": p.supports_streaming,
        "available_modes": _available_modes_for(p),
    }
```

`_available_modes_for(p)` returns:
- All four modes when `p.supports_audio_upload=True`
- `[WHISPER_LOCAL, WHISPER_API, IVRIT_AI]` when `False`

### 8.2 Frontend (`static/index.html`)

On page load, fetch `/api/capabilities` and hide the `gemini_direct` `<option>`
in the mode dropdown if the provider doesn't support audio upload. Display a
small badge near the dropdown showing the active provider name (purely
informational, ~10 lines of JS).

### 8.3 Why this matters

Without the guard, a user with `LLM_PROVIDER=ollama` could pick
`GEMINI_DIRECT`, wait, and get a confusing failure mid-pipeline. With it,
the option is silently absent and the failure mode is impossible.

## 9. Error handling

### 9.1 Hierarchy

```python
# app/services/llm_providers/errors.py
class ProviderError(RuntimeError):
    def __init__(self, *, provider: str, stage: str, code: str,
                 user_message: str, technical_details: str = ""):
        self.provider = provider
        self.stage = stage
        self.code = code
        self.user_message = user_message
        self.technical_details = technical_details
        super().__init__(user_message)

class ProviderUnsupportedError(ProviderError): ...
class ProviderRateLimitError(ProviderError): ...
class ProviderTimeoutError(ProviderError): ...
class ProviderAuthError(ProviderError): ...
```

### 9.2 Mapping

| Underlying error | Mapped to |
|---|---|
| Gemini 429 / quota string | `ProviderRateLimitError` |
| Gemini 5xx | retried, then raised as generic `ProviderError` |
| OpenRouter 401 | `ProviderAuthError` |
| OpenRouter 429 | `ProviderRateLimitError` |
| `httpx.ReadTimeout` | `ProviderTimeoutError` |
| `asyncio.TimeoutError` (outer) | `ProviderTimeoutError` |

### 9.3 User messages (Hebrew, surface in UI)

- `RateLimit` → `"⚠️ מכסת ה-API של {provider} הוצתה — נסה שוב בעוד כמה דקות"`
- `Auth` → `"🔑 מפתח ה-API של {provider} שגוי או לא הוגדר"`
- `Timeout` → `"⏱️ {provider} לא הגיב בזמן — נסה שוב"`
- `Unsupported` → `"❌ ה-LLM הנוכחי ({provider}) לא תומך ב-{feature}. בחר מצב עיבוד אחר."`

These messages match the existing error-message style in `summarizer.py` (the
emoji + Hebrew sentence pattern is already in use).

### 9.4 Note on coupling with Task 1.5

Task 1.5 of `UPGRADE_PROMPT.md` introduces a broader `ProcessingError` for
the whole pipeline. The `ProviderError` defined here is a strict subset and
can later be wrapped/lifted by Task 1.5 without breaking changes.

## 10. Testing strategy (TDD)

New file: `tests/test_llm_providers.py`

### 10.1 Unit tests (no network, fast)

1. `test_factory_returns_gemini_when_configured`
2. `test_factory_returns_openrouter_when_configured`
3. `test_factory_returns_ollama_when_configured`
4. `test_factory_raises_on_unknown_provider`
5. `test_openrouter_audio_upload_raises_unsupported`
6. `test_ollama_audio_upload_raises_unsupported`
7. `test_gemini_provider_calls_genai_client_correctly` — patch `genai.Client`,
   assert request body and parsed text match what current code produces.
8. `test_openrouter_builds_correct_payload` — patch `httpx.AsyncClient`,
   assert URL/headers/body for both `generate_text` and `stream_text`.
9. `test_ollama_builds_correct_payload` — same shape, Ollama endpoints.
10. `test_retry_on_429_then_succeeds` — mock returns 429 then 200, assert
    success and that retry happened.
11. `test_retry_classifies_auth_error_as_terminal` — 401 must NOT retry.

### 10.2 Backward-compat tests (must keep passing untouched)

These existing tests exercise summarizer.py's public API. They run with
`LLM_PROVIDER=gemini` (default) and a mocked Gemini client. **Zero source
changes** allowed:

- `tests/test_exam_critique.py`
- `tests/test_flashcards.py`
- `tests/test_code_switching_prompt.py`
- `tests/test_timestamp_player.py`
- `tests/test_ivrit_ai_wiring.py`
- `tests/test_auth.py`

If any of these break, the refactor regressed and we stop.

### 10.3 Integration tests (skipped when keys absent)

12. `test_gemini_real_short_summary` — `pytest.mark.skipif(not GOOGLE_API_KEY)`,
    summarize a 200-char transcript, assert structure.
13. `test_openrouter_real_short_summary` — `skipif(not OPENROUTER_API_KEY)`.

Ollama integration test is **omitted** from CI — no Ollama in the deploy env.
A manual smoke-test command goes in CLAUDE.md.

## 11. Migration plan (incremental, never red)

Order of work, each step independently verifiable:

1. **Step 1 — scaffold.** Create `app/services/llm_providers/{__init__,base,errors,gemini,openrouter,ollama}.py`. `base.py` defines the ABC. `gemini.py` is implemented and unit-tested. `openrouter.py` and `ollama.py` raise `NotImplementedError` for now. **summarizer.py untouched.**
2. **Step 2 — switch summarizer.py to Gemini provider.** Mechanical refactor: replace direct `genai` calls with `provider.*` calls. Run **all** existing tests — must pass.
3. **Step 3 — implement OpenRouter.** Fill in `openrouter.py`. Add unit tests.
4. **Step 4 — implement Ollama.** Fill in `ollama.py`. Add unit tests (Ollama is code-only, never selected at runtime on Fly.io).
5. **Step 5 — capabilities endpoint + UI guard.** Tiny backend endpoint + ~10 lines in `static/index.html`.
6. **Step 6 — config + .env.example + CLAUDE.md update.**
7. **Step 7 — `docker compose up -d --build` smoke test.** `/health` green, one end-to-end test from the UI with `LLM_PROVIDER=gemini`.
8. **Step 8 — `/review`** before commit/PR.

Each step is its own commit on a feature branch (`feature/llm-provider-abstraction`).

## 12. Risk callouts

- **Streaming chat behavior parity.** Gemini's `generate_content_stream` and OpenRouter's SSE `[DONE]` framing have different end-of-stream conventions. The `stream_text` interface returns clean strings; both providers translate their native frames into plain deltas before yielding. Tested by mocking the byte stream and asserting yielded chunks.
- **Token count differences across providers.** OpenRouter/Ollama may produce slightly different summaries than Gemini for the same prompt. This is not a regression — it's the design intent. The exam critique pipeline still gates quality.
- **`google-genai` stays in `requirements.txt`** even if a user runs Ollama-only. ~5MB, acceptable.
- **No automatic provider failover.** If Gemini errors, we surface the error; we don't silently retry on OpenRouter. A future "fallback chain" feature is explicitly out of scope.
- **Thread-pool executor staying in Gemini provider.** `google-genai` is sync; `httpx.AsyncClient` is async. Mixing them under one ABC is fine because `generate_text` is `async def` either way — the difference is internal.

## 13. File-level deliverables (concrete)

| File | Action | Approx LOC |
|---|---|---|
| `app/services/llm_providers/__init__.py` | new | ~20 |
| `app/services/llm_providers/base.py` | new | ~120 |
| `app/services/llm_providers/errors.py` | new | ~40 |
| `app/services/llm_providers/gemini.py` | new (extracts current logic) | ~250 |
| `app/services/llm_providers/openrouter.py` | new | ~180 |
| `app/services/llm_providers/ollama.py` | new | ~150 |
| `app/services/summarizer.py` | refactor: replace `genai` direct calls with provider | net delta ~−80 |
| `app/config.py` | add 5 fields + validator | ~25 |
| `app/api/routes.py` | add `/capabilities` endpoint | ~25 |
| `static/index.html` | fetch capabilities, filter dropdown, show provider badge | ~30 |
| `.env.example` | add new keys | ~8 |
| `CLAUDE.md` | document the new section in Architecture | ~15 |
| `tests/test_llm_providers.py` | new | ~300 |

Total roughly +1100 LOC, −80 LOC net change.

## 14. Definition of done

- All existing tests pass with `LLM_PROVIDER=gemini` (default) — zero edits to those tests.
- New `test_llm_providers.py` passes (unit + integration where keys present).
- `docker compose up -d --build` boots, `/health` is green, and a manual end-to-end run via the UI in `gemini_direct` mode produces a `LessonResult`.
- `LLM_PROVIDER=openrouter` with a valid key produces a `LessonResult` from a transcript-mode pipeline (manual smoke test).
- `/api/capabilities` returns the right payload for both providers.
- `/review` finds no high or medium severity issues (or any are addressed before merge).
- `CLAUDE.md` updated.
