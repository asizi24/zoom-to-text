# Zoom to Text — Project Context for Claude

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
- **Infrastructure**: Docker + docker-compose
- **Auth**: Chrome extension sends Netscape-format cookies directly to the server

## Key Files
```
app/main.py                    # FastAPI app, lifespan, idle watcher
app/config.py                  # Settings from .env (pydantic-settings)
app/models.py                  # Pydantic schemas
app/state.py                   # SQLite CRUD
app/api/routes.py              # REST endpoints
app/services/processor.py      # Main pipeline orchestrator
app/services/transcriber.py    # Faster-Whisper + OpenAI API
app/services/audio_preprocessor.py  # ffmpeg chunking + silence removal
app/services/summarizer.py     # Gemini — summary + exam
app/services/zoom_downloader.py     # yt-dlp + cookie handling
static/index.html              # Full UI
extension/                     # Chrome extension
```

## Processing Pipeline
```
GEMINI_DIRECT:  download → summarize_audio(audio_path) → Gemini → done
WHISPER_LOCAL:  download → transcribe → Faster-Whisper → summarize_transcript → Gemini → done
WHISPER_API:    download → preprocess (13min chunks + silence removal) → OpenAI whisper-1 → summarize_transcript → Gemini → done
```
Progress updates: 5% → 40% → 50% → 80% → 100% stored in SQLite, polled every 2s by UI.

## Important Technical Notes
1. **Whisper local model**: Lazy-loaded on first use, unloaded after 30 min idle
2. **Gemini**: Returns raw JSON (no markdown fences), has retry logic for 429/5xx
3. **SQLite**: Every task saved with full result_json — reloadable from History tab
4. **audio_preprocessor**: Uses ffmpeg only (no pydub), runs only in WHISPER_API mode
5. **Chrome extension**: Sends cookies in Netscape format + URL directly to the server
6. **Run locally**: `docker compose up -d` → http://localhost:8000

## GitHub
- Repo: https://github.com/asizi24/zoom-to-text
- Active branch: `local-monolith`

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
1. **Error handling in processor.py** — tasks can silently fail; improve error messages stored in SQLite
2. **Gemini prompt quality** — exam questions are sometimes too easy; tighten the prompt
3. **UI progress feedback** — the progress bar feels slow/jumpy; improve estimated time logic

### 🟡 Medium Priority
4. **Whisper local memory leak** — verify the 30-min idle unload actually frees GPU memory
5. **Audio preprocessor robustness** — what happens with very short recordings (<5 min)?
6. **History tab UX** — add search/filter by date or topic

### 🟢 Nice to Have
7. **Export to PDF** — alongside current Markdown export
8. **Multi-language support** — Gemini prompt currently assumes Hebrew content
9. **Rate limiting** — protect the API from abuse

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
