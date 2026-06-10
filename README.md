# Zoom to Text — AI Study Assistant

FastAPI service that turns a Zoom or YouTube recording into a structured summary, chapters, an American-style exam, flashcards, action items, and a fully searchable transcript — in a few minutes.

**Live deployment:** https://zoom-to-text.fly.dev (closed beta — magic-link allowlist)

---

## Highlights

- **3 transcription modes** — Gemini Direct (audio → LLM), Whisper Local, Whisper API. Plus an **ivrit-ai** mode tuned for Hebrew.
- **3 LLM backends** — Gemini 2.5 Flash, OpenRouter, Ollama. Switch with one env var.
- **Two-call pipeline** — synthesis (summary + chapters + exam) ‖ extraction (action items, decisions, open questions, sentiment, objections) run in parallel.
- **Diarization** — text-based via Gemini (default) or acoustic via Pyannote (home-server only).
- **Study tools** — Anki / CSV flashcard export, SM-2 spaced repetition, Socratic AI tutor, transcript search, mind maps, smart highlights, cross-lecture glossary, topic-mastery dashboard.
- **Collaboration** — public share links, cohort per-task sharing, ask-across-lectures, weekly email digest, outgoing webhooks (Slack/Discord).
- **Exports** — Obsidian-flavored Markdown, PDF, ICS calendar, Anki `.apkg`, AI-generated podcast script.
- **Auth** — Magic-link via Resend.com (closed allowlist) **or** LTI 1.3 SSO (Canvas / Moodle).
- **PWA** — installable on desktop and mobile.

---

## Architecture

```
                       ┌─────────────────────┐
URL or upload ─────────┤ download (yt-dlp)   │
                       └──────────┬──────────┘
                                  │
                       ┌──────────▼──────────┐
                       │ transcribe          │   Gemini Direct │ Whisper Local
                       │                     │   Whisper API   │ ivrit-ai
                       └──────────┬──────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
        ┌─────────▼─────┐ ┌───────▼──────┐ ┌──────▼──────┐
        │ diarization   │ │ synthesis    │ │ extraction  │
        │ (Gemini /     │ │ summary +    │ │ actions,    │
        │  Pyannote)    │ │ chapters +   │ │ decisions,  │
        │               │ │ exam         │ │ sentiment   │
        └───────────────┘ └──────┬───────┘ └──────┬──────┘
                                 │                │
                                 └───────┬────────┘
                                         │
                                ┌────────▼────────┐
                                │ SQLite + UI     │
                                └─────────────────┘
```

Synthesis and extraction run in parallel via `asyncio.gather`. Synthesis failure fails the task; extraction and diarization degrade gracefully.

---

## Quick Start (Local Docker)

### 1. Clone

```bash
git clone https://github.com/asizi24/zoom-to-text.git
cd zoom-to-text
```

### 2. Configure

```bash
cp .env.example .env
```

At minimum set `GOOGLE_API_KEY` (free key from [Google AI Studio](https://aistudio.google.com/app/apikey)).

### 3. Run

```bash
docker compose up -d
```

First build takes 3–5 minutes; subsequent starts ~5 seconds.

### 4. Use

Open http://localhost:8000 and paste a Zoom / YouTube URL or upload a file.

---

## Authentication

Two modes are supported:

**Magic-link (default).** Set `ALLOWED_EMAILS=alice@x.com,bob@y.com` and `RESEND_API_KEY=...`. The user enters their email at `/login` and receives a one-time link. Session is an opaque cookie in SQLite — revoke with `DELETE FROM sessions WHERE id = ?`.

**LTI 1.3 SSO.** For institutions running Canvas / Moodle / similar. Seed `/data/lti_platforms.json` with the platform registration JSON; users launch from the LMS course. Endpoints: `GET /lti/jwks`, `POST /lti/login`, `POST /lti/launch`.

`ADMIN_EMAILS` bypasses per-user 24h quotas and per-IP rate limits.

---

## Processing Modes

| Mode | Speed (3 h lecture) | Privacy | Requires | Best for |
|---|---|---|---|---|
| **Gemini Direct** | ~3 min | Audio uploaded to Google | `GOOGLE_API_KEY` | Most cases |
| **Whisper Local** | ~15 min (CPU) | Stays on your machine | nothing | Sensitive content |
| **Whisper API** | ~5 min | Audio uploaded to OpenAI | `OPENAI_API_KEY` | Speed + accuracy |
| **ivrit-ai** ⭐ | ~25 min (CPU) | Stays on your machine | nothing | Hebrew lectures |

Whisper Local model size via `WHISPER_MODEL` (`tiny` / `base` / `small` / `medium` / `large-v3`).

Whisper API handles long audio by removing silence + chunking to ≤13-minute pieces before upload (the OpenAI API has a 25 MB limit).

---

## LLM Provider Abstraction

Summarization, chat, flashcards, critique, podcast generation — all route through `app/services/llm_providers/`. Switch backends with:

```env
LLM_PROVIDER=gemini      # default — supports audio upload (GEMINI_DIRECT mode)
LLM_PROVIDER=openrouter  # OPENROUTER_API_KEY + OPENROUTER_MODEL
LLM_PROVIDER=ollama      # OLLAMA_BASE_URL + OLLAMA_MODEL (local; not deployed on Fly.io)
```

The frontend calls `GET /api/capabilities` on load and hides options the active provider can't support (e.g. `gemini_direct` is hidden under `ollama`).

---

## Feature Tour

**Study**
- Summary + chapters + American-style exam (Bloom-tagged questions)
- Flashcards with **SM-2 spaced repetition** and Anki `.apkg` export
- **Socratic AI tutor** — chat that probes rather than answers
- **Cross-lecture glossary** + **topic-mastery dashboard**
- **Smart Highlights** + lazy-rendered **Mind Map**
- Transcript search, multi-lecture cram guide

**Collaboration**
- Public share links per lesson
- Cohort per-task sharing
- Ask-across-lectures
- Weekly email digest
- Outgoing webhooks (Slack / Discord)

**Live capture**
- WebSocket streaming endpoint at `/ws/transcribe` (home-server only)
- Desktop loopback capture POC (`desktop/capture.py` — records WASAPI loopback + mic, posts to the server)

**Exports**
- Obsidian-flavored Markdown (YAML frontmatter, `#action/<owner>` tags, collapsible exam)
- PDF
- ICS calendar (action items → calendar events)
- AI-generated podcast script
- Anki `.apkg` and CSV

**UX**
- PWA install on desktop & mobile
- Live tasks panel on home (cancel running jobs)
- Bulk delete + auto-cleanup
- Chat with recording (diarized)
- Timestamp player (click `[MM:SS]` → seek)
- Lesson recipes (saved processing presets)
- Slide alignment (sync slide deck to transcript)
- Multi-language synthesis (`LECTURE_LANGUAGE=auto|he|en|...`)

---

## Chrome Extension (private Zoom recordings)

If your recordings require Zoom auth, install the extension from `extension/`:

1. `chrome://extensions/` → Developer mode → Load unpacked → select `extension/`
2. Open the Zoom recording page (logged in) → click the extension → **Send to Transcriber**

The extension extracts Netscape-format cookies and posts the URL + cookies to the server, which feeds them to `yt-dlp`.

---

## Project Structure

```
zoom-to-text/
├── app/
│   ├── api/
│   │   ├── routes.py            # REST endpoints
│   │   ├── auth.py              # Magic-link + session cookie
│   │   ├── lti.py               # LTI 1.3 SSO
│   │   ├── streaming.py         # WebSocket /ws/transcribe
│   │   └── deps.py              # Shared dependencies
│   ├── services/
│   │   ├── processor.py         # Pipeline orchestrator
│   │   ├── transcriber.py       # Faster-Whisper + OpenAI Whisper
│   │   ├── summarizer.py        # Synthesis ‖ extraction + diarization
│   │   ├── audio_preprocessor.py
│   │   ├── zoom_downloader.py
│   │   ├── llm_providers/       # gemini / openrouter / ollama
│   │   ├── diarization/         # gemini text + pyannote acoustic
│   │   ├── exporters/markdown.py
│   │   ├── lti/                 # LTI 1.3 modules
│   │   ├── anki_export.py
│   │   ├── clip_extractor.py
│   │   ├── glossary.py
│   │   ├── sm2.py               # Spaced repetition
│   │   ├── podcast_script.py
│   │   ├── webhooks.py
│   │   ├── email_digest.py
│   │   └── text_extractor.py
│   ├── config.py
│   ├── models.py
│   ├── state.py
│   ├── errors.py                # ProcessingError + classifier
│   └── main.py
├── static/
│   ├── index.html               # SPA
│   └── ...
├── extension/                   # Chrome extension
├── desktop/                     # Loopback capture POC
├── tests/                       # 261 server + 9 desktop tests
├── docs/                        # Design docs, specs, plans
├── Dockerfile
├── docker-compose.yml
├── fly.toml
├── requirements.txt
├── requirements-heavy.txt       # torch + pyannote (home-server only)
└── .env.example
```

---

## API Reference

Swagger UI at http://localhost:8000/docs (or set `ENABLE_DOCS=false` for production).

**Tasks**

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/tasks` | Submit a URL |
| `POST` | `/api/tasks/upload` | Submit an uploaded file |
| `GET` | `/api/tasks` | List recent tasks |
| `GET` | `/api/tasks/{id}` | Poll status / get result |
| `GET` | `/api/tasks/{id}/events` | SSE progress stream |
| `POST` | `/api/tasks/{id}/cancel` | Cancel in-flight task |
| `POST` | `/api/tasks/{id}/retry` | Retry failed task |
| `DELETE` | `/api/tasks/{id}` | Delete |
| `POST` | `/api/tasks/bulk_delete` | Bulk delete |

**Study**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/tasks/{id}/transcript` | Full transcript |
| `GET` | `/api/tasks/{id}/search` | Search inside transcript |
| `POST` | `/api/tasks/{id}/ask` | Q&A on a single lesson |
| `POST` | `/api/ask` | Ask across all lessons |
| `POST` | `/api/tasks/{id}/chat` | Chat history |
| `POST` | `/api/tasks/{id}/tutor` | Socratic tutor turn |
| `POST` | `/api/tasks/{id}/mindmap` | Generate mind map |
| `GET` | `/api/tasks/{id}/flashcards` | Get flashcards |
| `POST` | `/api/tasks/{id}/flashcards/{idx}/review` | SM-2 review |
| `GET` | `/api/flashcards/due` | Cards due across all lessons |
| `GET` | `/api/glossary` | Cross-lecture glossary |
| `GET` | `/api/mastery` | Topic-mastery dashboard |
| `POST` | `/api/study-guide` | Multi-lesson cram guide |

**Exports & sharing**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/tasks/{id}/export/obsidian` | Obsidian Markdown |
| `GET` | `/api/tasks/{id}/export/pdf` | PDF |
| `GET` | `/api/tasks/{id}/export/ics` | ICS calendar |
| `GET` | `/api/tasks/{id}/flashcards/export.apkg` | Anki package |
| `GET` | `/api/tasks/{id}/flashcards/export.csv` | CSV |
| `GET` | `/api/tasks/{id}/podcast-script` | AI podcast script |
| `POST` | `/api/tasks/{id}/share` | Create public share link |
| `GET` | `/share/{token}` | Public share viewer |
| `POST` | `/api/tasks/{id}/shares` | Share with a cohort user |
| `GET` | `/api/shared-tasks` | Tasks shared with me |

**Integrations & auth**

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`/`POST`/`DELETE` | `/api/webhooks` | Outgoing webhooks CRUD |
| `GET` | `/api/capabilities` | What the active provider supports |
| `POST` | `/api/auth/request` | Send magic link |
| `GET` | `/api/auth/verify` | Verify magic link |
| `POST` | `/api/auth/logout` | Logout |
| `GET` | `/lti/jwks` | LTI 1.3 JWKS |
| `POST` | `/lti/login` `/lti/launch` | LTI handshake |
| `WS` | `/ws/transcribe` | Live transcription (home-server only) |
| `GET` | `/health` | Health check |

---

## Deployment (Fly.io)

```bash
fly deploy --remote-only          # build on Fly's depot builder
fly logs                          # tail server logs
fly status                        # machine state
fly ssh console                   # shell into the machine
```

Secrets:

```bash
fly secrets set GOOGLE_API_KEY=...
fly secrets set RESEND_API_KEY=... ALLOWED_EMAILS="a@x.com,b@y.com"
```

Windows + Git Bash: prefix path values with `MSYS_NO_PATHCONV=1` to stop path mangling, e.g. `MSYS_NO_PATHCONV=1 fly secrets set DATA_DIR="/data"`.

---

## Development

```bash
pytest tests/ -q                  # 261 server tests
pytest tests/desktop/ -v          # 9 desktop tests (needs desktop/requirements.txt)
uvicorn app.main:app --reload     # hot reload
```

**Don't run `pytest .` from repo root** — it picks up `_legacy_archive/` and errors.

Contributing context lives in [`CLAUDE.md`](CLAUDE.md). The strategic roadmap is in [`docs/UPGRADE_PROMPT.md`](docs/UPGRADE_PROMPT.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Port 8000 busy | Edit `docker-compose.yml`: `"8001:8000"` |
| Zoom download fails (403) | Use the Chrome extension to send cookies |
| Whisper OOM | Smaller model (`WHISPER_MODEL=small`) |
| "No space left" | `docker system prune -af` |
| Fly TLS cert error on deploy | Transient — re-run `fly deploy --remote-only` |

---

## License

MIT
