# Zoom to Text — Project Context for Claude

## Status (2026-04-28)
- **Branch:** `main` (single active branch; cycle/feature branches merged via fast-forward)
- **Live URL:** https://zoom-to-text.fly.dev
- **Cycle A shipped:** Task 1.3 (LLM Provider Abstraction)
- **Cycle B shipped:** Task 1.1 (extraction-fields schema), Task 1.2 (Gemini text diarization), Task 1.4 (Obsidian export), Task 1.5 (ProcessingError + structured `error_details`)
- **Task 2.1 shipped (code-only, not deployed):** `desktop/` loopback capture POC — runs on the user's machine, posts to existing `/api/tasks/upload`
- **Test suite:** 187 passing (178 server + 9 desktop)

## What This Project Does
A FastAPI service that receives a Zoom recording (URL or file upload) and returns within minutes:
- A 3-5 paragraph summary
- Chapters organized by topic with key points
- An American-style exam with 8-10 questions and explanations

## Architecture
- **Backend**: FastAPI + uvicorn
- **Transcription**: 3 modes — `gemini_direct` (audio → Gemini), `whisper_local` (Faster-Whisper), `whisper_api` (OpenAI whisper-1)
- **Summarization**: Gemini 2.5 Flash via google-genai SDK, returns raw JSON (no markdown fences)
- **Download**: yt-dlp + ffmpeg, supports private recordings via Chrome cookie injection
- **DB**: SQLite with aiosqlite (async), stores full result_json per task
- **Frontend**: Single-file SPA at `static/index.html` (3 tabs: Link / Upload / History)
- **Infrastructure**: Docker + docker-compose locally; Fly.io in production
- **Auth**: Magic-link via Resend.com → `session_id` HttpOnly cookie (`app/api/auth.py`). Whitelist `ALLOWED_EMAILS` env var.
- **Chrome extension** (separate concern): exports Netscape-format Zoom cookies for `yt-dlp` so the server can download private recordings. Not used for app auth.

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

### Two-call pipeline (Tasks 1.1 + 1.2)
For every recording, `summarizer` runs **synthesis** (summary + chapters
+ exam at temperature 0.3) and **extraction** (action_items, decisions,
open_questions, sentiment_analysis, objections_tracked at temperature
0.2) in parallel via `asyncio.gather`. WHISPER paths run an additional
**diarization** call beforehand to label `Speaker A/B/...` and propose
real names when self-introductions are detected. Synthesis failure is
fatal; extraction and diarization failures degrade gracefully (fields
default to empty / None).

### Structured errors (Task 1.5)
Pipeline stages wrap exceptions through `app/errors.py`'s
`classify_exception()` into a `ProcessingError(stage, code,
user_message, technical_details)`. Persisted as JSON in the
`tasks.error_details` SQLite column. The History UI prepends the
stage label (`[הורדה]`, `[תמלול]`, `[סיכום]`...) to the failure
banner.

### Obsidian Markdown export (Task 1.4)
`app/services/exporters/markdown.py` is a pure function that renders a
completed `TaskResponse` to Obsidian-flavored Markdown — YAML
frontmatter (`date`, `source`, `content_type`, `language`,
`participants`, `tags`), action items as `- [ ]` checkboxes with
`#action/<owner>` tags, dedicated sections for decisions / open
questions / sentiment / objections, exam in a collapsible `<details>`
block, and the diarized transcript when present. Exposed via
`GET /api/tasks/{id}/export/obsidian` (cookie auth, returns
`text/markdown` attachment). The original client-side Markdown export
is unchanged.

### Desktop loopback capture POC (Task 2.1, code-only)
`desktop/capture.py` is a standalone Python tool that records the
user's machine (WASAPI loopback on Windows + mic, mixed to mono 48
kHz PCM_16 WAV) and uploads to `/api/tasks/upload` via cookie auth.
Stops automatically after 2 minutes of system-audio silence.
Dependencies live in `desktop/requirements.txt` (sounddevice, numpy,
soundfile, httpx) and are **never installed on Fly.io** — the server
stays small. Tests in `tests/desktop/` use mocked sounddevice +
`httpx.MockTransport`; the hardware-gated test is `skipif`'d off CI.

## Key Files
```
app/main.py                    # FastAPI app, lifespan, idle watcher
app/config.py                  # Settings from .env (pydantic-settings)
app/models.py                  # Pydantic schemas (incl. Task 1.1 extraction fields)
app/state.py                   # SQLite CRUD (incl. error_details column)
app/errors.py                  # ProcessingError + classify_exception (Task 1.5)
app/api/routes.py              # REST endpoints
app/api/auth.py                # Magic-link auth + session_id cookie
app/services/processor.py      # Main pipeline orchestrator (wraps stages with _run_stage)
app/services/transcriber.py    # Faster-Whisper + OpenAI API
app/services/audio_preprocessor.py  # ffmpeg chunking + silence removal
app/services/summarizer.py     # Synthesis ‖ extraction + diarization
app/services/zoom_downloader.py     # yt-dlp + cookie handling
app/services/llm_providers/    # Provider abstraction (gemini/openrouter/ollama)
app/services/exporters/markdown.py  # Obsidian-flavored markdown exporter (Task 1.4)
static/index.html              # Full UI (incl. 🟣 Obsidian button)
extension/                     # Chrome extension (Zoom-cookie helper for yt-dlp)
desktop/capture.py             # Standalone loopback capture POC (Task 2.1)
desktop/requirements.txt       # Desktop-only deps; never installed on Fly.io
tests/                         # 178 server tests
tests/desktop/                 # 9 desktop tests (mocked sounddevice + httpx)
```

## Processing Pipeline
```
GEMINI_DIRECT:  download → summarize_audio(audio_path) → (synthesis ‖ extraction) → merge → done
WHISPER_LOCAL:  download → Faster-Whisper transcribe → diarize → (synthesis ‖ extraction) → merge → done
WHISPER_API:    download → preprocess (13min chunks + silence removal) → OpenAI whisper-1 → diarize → (synthesis ‖ extraction) → merge → done
```
- **synthesis ‖ extraction** runs via `asyncio.gather` (Task 1.1).
- **diarize** runs only on WHISPER paths and only when `ENABLE_DIARIZATION=True`
  with a Gemini provider (Task 1.2).
- Each stage is wrapped by `processor._run_stage(stage, coro)`. Native exceptions
  are classified into `ProcessingError(stage, code, user_message,
  technical_details)` and persisted as JSON in `tasks.error_details` (Task 1.5).
- Progress updates: 5% → 40% → 50% → 80% → 100% stored in SQLite, polled every
  2 s by the UI.

## Important Technical Notes
1. **Whisper local model**: Lazy-loaded on first use, unloaded after 30 min idle
2. **Gemini**: Returns raw JSON (no markdown fences), has retry logic for 429/5xx
3. **SQLite**: Every task saved with full result_json — reloadable from History tab
4. **audio_preprocessor**: Uses ffmpeg only (no pydub), runs only in WHISPER_API mode
5. **Chrome extension**: Sends cookies in Netscape format + URL directly to the server
6. **Run locally**: `docker compose up -d` → http://localhost:8000

## GitHub
- Repo: https://github.com/asizi24/zoom-to-text
- Active branch: `main`

---

## How to Use Your Installed Tools

### 🧠 claude-mem — Persistent Memory
At the START of every session, run:
```
/mem load
```
At the END of every session, run:
```
/mem save
```
Use `/mem note` to capture important decisions or discoveries mid-session (e.g., "whisper model path changed", "Gemini prompt updated").

### 🔍 Code Review (PR Review Agent)
Before pushing any branch, run:
```
/review
```
This runs 5 parallel review agents (bugs, compliance, git context, history, comments). Only high-confidence findings are surfaced. Use it before every PR to `main`.

### 🔒 Security Review (security-guidance)
This plugin is **always-on** — it intercepts write operations and warns about:
- Command injection risks in `zoom_downloader.py` (yt-dlp subprocess calls)
- Unsafe exec() patterns
- XSS vectors in `index.html`
- Cookie/secret exposure in `config.py`

Take every warning seriously. Do NOT dismiss security warnings without understanding them.

### 🎨 Frontend Design
When improving `static/index.html`, invoke:
```
/frontend-design
```
This pushes Claude toward bold, distinctive UI — not generic AI aesthetics. Good for:
- Redesigning the progress UI
- Improving the exam/quiz display
- Making the History tab more useful

### ⚡ Superpowers — TDD & Systematic Debugging
Use these skills when working on backend code:

**For new features:**
```
/tdd [feature description]
```
Follow red-green-refactor: write failing test first, then implement.

**For bugs:**
```
/debug [symptom description]
```
Enforces 4-phase debugging: observe → hypothesize → test → fix. Do NOT write code until phase 3.

**For code quality:**
```
/brainstorm [problem]
```
Before implementing anything non-trivial, brainstorm approaches.

### 🏢 gstack — Virtual Engineering Team
Use gstack roles for different types of work:
- **`/pm`** — When prioritizing what to improve next
- **`/eng`** — When implementing backend features
- **`/designer`** — When working on UI/UX
- **`/qa`** — When writing tests or reviewing edge cases
- **`/security`** — Deep security review of specific components

---

## Improvement Priorities (work through these in order)

### 🔴 High Priority
1. **Task 2.2 — Pyannote diarization provider (code-only)** — implement
   `app/services/diarization/pyannote_provider.py` against the existing
   text-based diarization interface. Heavy deps (`pyannote.audio`) live in a
   new `requirements-heavy.txt`, **never installed on Fly.io**. Gate with
   `DIARIZATION_PROVIDER=gemini|pyannote`. Activates only on a future home
   server with GPU.
2. **Task 2.3 — WebSocket streaming endpoint** — `app/api/streaming.py` with
   `/ws/transcribe` for incremental transcript + speaker labels. Production
   stays at `ENABLE_STREAMING=False`; turned on only on the home server.
3. **Gemini prompt quality** — exam questions are sometimes too easy; tighten
   the prompt (existing item, still open).
4. **Task 3.\* — Test expansion + `/review` integration + this file's own
   updates** — every shipped task should land with new tests, a `/review`
   pass, and a CLAUDE.md refresh.

### 🟡 Medium Priority
4. **Whisper local memory leak** — verify the 30-min idle unload actually frees GPU memory
5. **Audio preprocessor robustness** — what happens with very short recordings (<5 min)?
6. **History tab UX** — add search/filter by date or topic

### 🟢 Nice to Have
7. **Export to PDF** — alongside current Markdown export
8. **Multi-language support** — Gemini prompt currently assumes Hebrew content
9. **Rate limiting** — protect the API from abuse

---

## Operating Principles

- **Test First, Code Second.** For any pipeline change — new LLM call,
  schema field, error path, exporter — write the failing test before
  the implementation. The superpowers `/tdd` skill enforces the
  red-green-refactor loop.
- **Ephemeral-Volume Hygiene.** Fly.io's persistent volume at `/data` is
  10 GB. Every audio file the pipeline creates must have a defined
  deletion path. The reference pattern is `Path.unlink()` after task
  delete (`app/api/routes.py:163`). Do not let temp artifacts pile up
  in `data/`.
- **Targeted Reading.** Prefer `@path/to/file` references and
  `Grep`/`Glob` over full-file reads when mapping functionality. Use
  `Read` only when the surrounding code actually matters for the task
  at hand.
- **Provider-Agnostic LLM Calls.** Any new LLM-using code must go
  through `app/services/llm_providers/get_provider()`. Never import
  `google-genai` directly outside that package.
- **Backward-Compatible Schema Additions.** SQLite stores
  `LessonResult` as a JSON blob. Every new field must be `Optional`
  with a default of `[]` / `None`, and accompanied by a
  `test_lesson_result_loads_old_json_*`-style test.

---

## Session Workflow

1. `/mem load` — load previous context
2. Pick ONE item from Improvement Priorities
3. `/brainstorm` or `/pm` to clarify approach
4. `/tdd` for backend, `/frontend-design` for UI
5. Security plugin monitors automatically — review any warnings
6. `/review` before committing
7. `/mem save` — persist what you learned

---
*This file is automatically read by Claude Code at session start.*
