# Zoom to Text — Project Context for Claude

## Status (2026-05-03)
- **Branch:** `main` (single active branch; cycle/feature branches merged via fast-forward)
- **Live URL:** https://zoom-to-text.fly.dev
- **Cycle A shipped:** Task 1.3 (LLM Provider Abstraction)
- **Cycle B shipped:** Task 1.1 (extraction-fields schema), Task 1.2 (Gemini text diarization), Task 1.4 (Obsidian export), Task 1.5 (ProcessingError + structured `error_details`)
- **Task 2.1 shipped (code-only, not deployed):** `desktop/` loopback capture POC — runs on the user's machine, posts to existing `/api/tasks/upload`
- **Task 2.2 shipped (code-only, not deployed):** Pyannote acoustic diarization — `app/services/diarization/pyannote_provider.py`. Heavy deps in `requirements-heavy.txt`. Gate: `DIARIZATION_PROVIDER=pyannote`.
- **Task 2.3 shipped (code-only, not deployed):** WebSocket streaming endpoint at `/ws/transcribe`. Gate: `ENABLE_STREAMING=false` (never enable on Fly.io). 8 tests in `tests/test_streaming.py`.
- **LTI 1.3 SSO shipped + reviewed:** institutional auth via Canvas/Moodle. 42 LTI tests + atomic `DELETE … RETURNING` anti-replay fix applied post-review. Awaiting LMS admin handoff to seed `/data/lti_platforms.json`.
- **Test suite:** 261 passing (252 server + 9 desktop)

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

## Why These Choices

Quick rationale for the load-bearing decisions — read before proposing
migrations, especially "let's switch to X for scale."

- **Fly.io over Cloud Run.** Cloud Run is ephemeral; SQLite files and
  in-flight `BackgroundTasks` would die on every cold start. Fly.io
  has a persistent machine + 10 GB volume — both audio files and the
  task DB survive restarts.
- **SQLite over Postgres.** ~5 concurrent users max. A second daemon
  with its own connection pool would buy nothing at this scale.
- **`BackgroundTasks` over Celery / ARQ.** Background work runs on the
  same uvicorn process. No Redis, no worker fleet, no ops surface.
  At 5 users this is the cheapest correct answer; revisit only if
  task starvation appears.
- **Two-call synthesis ‖ extraction.** Synthesis wants temperature 0.3
  (creative summary); extraction wants 0.2 (factual lists). Merging
  into one prompt loses precision; running serially doubles latency.
  `asyncio.gather` keeps wall-clock at `max(call_1, call_2)` while
  paying ~30 % more tokens.
- **Graceful skip vs fatal failure.** Synthesis is the user's primary
  product — its failure fails the task. Extraction and diarization
  are enrichments — their failure logs a warning and leaves fields
  empty.
- **Text-based diarization, not Pyannote.** Current Fly.io machine is
  512 MB shared-cpu-1x. Pyannote needs GPU and ~2 GB RAM for the
  pretrained pipeline. Text diarization (Gemini-driven) loses some
  acoustic cues but runs free on the existing infra. Pyannote stays
  ready as code (Task 2.2) for a future home-server deploy.
- **Magic-link + session cookie over JWT / OAuth.** Closed group
  (~6 emails). Magic link via Resend.com costs nothing on the free
  tier. No password to manage; no third-party identity provider to
  configure. Session is an opaque random key in SQLite — easy to
  revoke (`DELETE FROM sessions WHERE id = ?`).

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

## Operations

Concrete commands a fresh session needs.

### Tests
```bash
pytest tests/ -q                     # full server suite (178 tests)
pytest tests/desktop/ -v             # desktop POC (9 tests; needs desktop reqs)
pytest tests/test_obsidian_export.py # single file, verbose
```
**Do not run `pytest .` from the repo root** — it picks up
`_legacy_archive/` and errors out on `from app import app`. Always
restrict to `tests/`.

To run desktop tests:
```bash
pip install -r desktop/requirements.txt
```
Tests skip cleanly when desktop deps are missing
(`pytest.importorskip` guards them).

### Local dev
```bash
docker compose up -d                 # http://localhost:8000
# or, for hot reload:
uvicorn app.main:app --reload
```

### Deploy
```bash
fly deploy --remote-only             # build on Fly's depot builder
fly logs                             # tail server logs
fly ssh console                      # shell into the machine
fly status                           # machine state
```
- The depot builder occasionally returns a TLS handshake error
  (`x509: certificate is not valid for any names`). Re-run; it's
  a transient infra issue.
- After deploy, smoke-check via:
  `curl --ssl-no-revoke -s https://zoom-to-text.fly.dev/health`
  (Windows curl needs `--ssl-no-revoke`; the site itself is fine.)

### Secrets
```bash
fly secrets set GOOGLE_API_KEY=...
fly secrets set RESEND_API_KEY=... ALLOWED_EMAILS="a@x.com,b@y.com"
fly secrets list
```
**Windows gotcha**: paths starting with `/` get rewritten by Git Bash:
```bash
MSYS_NO_PATHCONV=1 fly secrets set DATA_DIR="/data" DOWNLOADS_DIR="/data/downloads"
```

### Rollback
```bash
git revert <bad_sha>..<head_sha>     # create revert commits
git push origin main
fly deploy --remote-only             # rolling deploy of the revert
```

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

1. ~~**Task 2.2 — Pyannote diarization provider (code-only)**~~ ✅ **Done (2026-05-03)** — `app/services/diarization/pyannote_provider.py` with 9 tests. `requirements-heavy.txt` created. Gate: `DIARIZATION_PROVIDER=pyannote`.
2. ~~**Task 2.3 — WebSocket streaming endpoint**~~ ✅ **Done (2026-05-03)** — `app/api/streaming.py` at `/ws/transcribe` with 8 tests. Gate: `ENABLE_STREAMING=false`.
3. **Gemini prompt quality** — exam questions are sometimes too easy; tighten
   the prompt (existing item, still open).
4. ~~**Task 3.\* — Test expansion + `/review` + CLAUDE.md refresh**~~ ✅ **Done
   (2026-05-03)** — LTI back-fill: 42 tests, atomic anti-replay fix, review
   pass, this file updated.

### 🟡 Medium Priority
5. **Whisper local memory leak** — `auto_shutdown_idle_minutes=30` exists in
   `config.py`; the model unload path is wired, but actual GPU/RAM release
   has not been measured under sustained load. Profile before claiming done.
6. **Audio preprocessor robustness** — what happens on recordings <5 min,
   silent recordings, single-chunk paths? No regression tests yet.
7. **History tab UX** — add search/filter by date or topic.

### 🟢 Nice to Have
8. **Export to PDF** — alongside current Markdown export.
9. **Multi-language support** — Gemini prompt currently assumes Hebrew content.
10. **Rate limiting** — protect the API from abuse.

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
- **`pytest.importorskip` for Optional Deps.** Tests that exercise
  optional/desktop-only dependencies (e.g. `sounddevice`,
  `soundfile`) must call `pytest.importorskip("name")` before any
  module import that pulls them. This way the suite stays green on
  machines without the optional install (`tests/desktop/test_capture.py`
  is the reference pattern).

---

## Footguns / Don'ts

Land mines that have bitten this project before. A fresh Claude
session should know about these before the first edit.

- **Don't upgrade `httpx` past 0.27.0.** `httpx==0.28+` breaks
  starlette's `TestClient` (`Client.__init__() got unexpected 'app'`).
  Both `requirements.txt` and `desktop/requirements.txt` pin
  `httpx==0.27.0`. Same pin keeps a single test run exercising both.
- **Don't run `pytest .` from the repo root.** It picks up
  `_legacy_archive/` and errors on `from app import app`. Always
  use `pytest tests/` or a specific file path.
- **Don't add deps to top-level `requirements.txt` casually.** The
  Fly.io image is 512 MB shared-cpu-1x. Any heavy dep (torch,
  pyannote, transformers...) must go in a separate
  `requirements-heavy.txt` and never be imported at server boot.
- **Don't bypass `get_provider()`.** Direct `import google.genai`
  outside `app/services/llm_providers/gemini.py` defeats the
  abstraction and will break OpenRouter / Ollama paths. Same for
  flashcards, chat, critique — all routed through the provider.
- **Don't break `/api/*` backward compatibility.** The Chrome
  extension and the History tab read JSON blobs from old tasks; new
  fields must be optional with safe defaults. See the
  `test_lesson_result_loads_old_json_*` family.
- **Don't put secrets in code.** Use `fly secrets set` for prod and
  `.env` (gitignored) for local. `.env.example` is the source of
  truth for what variables exist.
- **Don't deploy without green tests.** `pytest tests/` must be
  178+ passing before `fly deploy`. The deploy is rolling but the
  machine is single — a bad image takes the site down.
- **Don't change Pydantic field names with Python keywords without
  alias.** `from`, `to`, `class`, etc. need `Field(..., alias="from")`
  + `model_config = {"populate_by_name": True}`. See `ToneShift` in
  `app/models.py`.
- **Don't catch `BaseException` to swallow errors.** Wrap in
  `processor._run_stage(stage, coro)` so `ProcessingError` is
  raised with stage context. Bare `except: pass` loses the
  `error_details` payload the UI depends on.
- **Don't render user content via `innerHTML` in `static/index.html`.**
  XSS surface. Use DOM APIs (`textContent`, `appendChild`). Existing
  patterns in `chapters-list` rendering are the reference.

## Where to Find More

External-to-this-file context Claude should reach for when relevant:

- `UPGRADE_PROMPT.md` — the strategic upgrade roadmap. Source of
  Tasks 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.x. Read this
  before scoping a "next big thing."
- `docs/superpowers/specs/` — design specs written before
  implementation (e.g. `2026-04-26-schema-upgrade-design.md`,
  `2026-04-26-diarization-design.md`). Useful for understanding
  the *why* behind a Cycle B feature.
- `docs/superpowers/plans/` — implementation plans
  (e.g. `2026-04-25-llm-provider-abstraction.md`).
- `.claude/plans/` (gitignored, user-local) — short-lived plan
  files from prior sessions.
- `~/.claude/projects/.../memory/` (gitignored, user-local) —
  persistent cross-session memory: project-state, technical
  decisions, deployment details, user profile. Loaded only via
  `/mem load`; not visible to fresh CI runs.

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
