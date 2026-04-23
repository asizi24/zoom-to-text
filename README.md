# Zoom Transcriber — AI Study Assistant

Automatically transcribe and summarize Zoom lecture recordings using AI.  
Upload a recording URL or an audio/video file → get a structured summary, chapter breakdown, and a 10-question quiz — in under 5 minutes.

---

## How It Works

1. **Download** — The server downloads the Zoom recording audio via yt-dlp + ffmpeg
2. **Process** — Three modes available (see [Processing Modes](#processing-modes) below)
3. **Results** — Structured output: summary, chapters with key points, 8–10 multiple-choice questions

---

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A **Gemini API key** (free) from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Windows 10/11, macOS, or Linux

> **RAM:** Gemini Direct mode needs ~1 GB. Whisper Local (medium model) needs ~3 GB.

---

## Setup — Step by Step

### 1. Clone the repository

```bash
git clone https://github.com/asizi24/zoom-to-text.git
cd zoom-to-text
```

### 2. Create your `.env` file

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

Get a free key at: https://aistudio.google.com/app/apikey

**Optional — OpenAI Whisper API mode:**

```env
OPENAI_API_KEY=sk-...
```

### 3. Start the server

```bash
docker compose up -d
```

First run takes **3–5 minutes** (builds the Docker image and downloads dependencies).  
Subsequent starts take ~5 seconds.

### 4. Open the web interface

```
http://localhost:8000
```

Paste a Zoom recording URL or upload an audio/video file and click **Start**.

---

## Chrome Extension (for private recordings)

If your Zoom recordings require authentication (e.g. university portal), use the Chrome extension:

### Install

1. Open Chrome → go to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder from this project

### Use

1. Open the Zoom recording page in Chrome while logged in
2. Click the extension icon
3. Click **Send to Transcriber** — it automatically extracts session cookies and sends the URL to your local server

> The extension communicates with `http://localhost:8000` by default.  
> You can change the server URL in the extension settings panel.

---

## Useful Commands

```bash
# Start the server (background)
docker compose up -d

# Stop the server
docker compose down

# View live logs
docker logs -f zoom_transcriber

# Rebuild after code changes
docker compose up -d --build

# Force full rebuild (clears pip cache)
docker compose build --no-cache
docker compose up -d

# Check server health
curl http://localhost:8000/health
```

---

## Processing Modes

| Mode | Speed | Privacy | Requirements | Best For |
|------|-------|---------|--------------|----------|
| **Gemini Direct** | ~3 min for 3h lecture | Audio sent to Google | `GOOGLE_API_KEY` | Most use cases |
| **Whisper Local** | ~15 min for 3h lecture (CPU) | Audio stays on your machine | None | Sensitive content |
| **OpenAI Whisper** | ~5 min for 3h lecture | Audio sent to OpenAI | `OPENAI_API_KEY` | Fast + accurate transcription |
| **ivrit-ai** ⭐ | ~25 min for 3h lecture (CPU) | Audio stays on your machine | None (auto-downloads model) | **Hebrew content** (best accuracy) |

### ivrit-ai mode — Hebrew-tuned Whisper

[ivrit-ai](https://github.com/ivrit-ai/ivrit.ai) is a Hebrew-focused fine-tune of OpenAI's Whisper, trained on thousands of hours of spoken Hebrew. On Hebrew lectures (especially with medical/technical jargon or fast speech) it typically outperforms vanilla Whisper Local.

- First use downloads ~1.5 GB of CT2 model weights into the `whisper_model_cache` Docker volume (one-time).
- Runs entirely on your machine — same privacy guarantees as Whisper Local.
- Override the model via `IVRIT_AI_MODEL=ivrit-ai/whisper-v3-ct2` in `.env`.

### OpenAI Whisper mode — how it works

Because the OpenAI API has a 25 MB file size limit (~22 min of audio at 96 kbps), the audio is preprocessed automatically before sending:

1. **Silence removal** — strips dead air using ffmpeg's `silenceremove` filter (threshold: −40 dBFS, minimum 1 second)
2. **Chunking** — splits into ≤13-minute pieces that safely fit under the API limit
3. **Transcription** — each chunk is sent to `whisper-1` in sequence
4. **Merge** — transcripts are joined in order and sent to Gemini for summarization

---

## Whisper Model Sizes

Relevant only for **Whisper Local** mode. Set `WHISPER_MODEL` in your `.env`:

| Model | RAM | Speed | Accuracy |
|-------|-----|-------|----------|
| `tiny` | ~400 MB | Fastest | Low |
| `base` | ~600 MB | Fast | OK |
| `small` | ~1 GB | Moderate | Good |
| `medium` | ~2 GB | Slow | **Recommended** |
| `large-v3` | ~4 GB | Slowest | Best |

---

## Supported File Formats

Direct upload supports: `mp3`, `mp4`, `m4a`, `wav`, `mkv`, `webm`  
Maximum upload size: **600 MB**

---

## UI Features

- **Export Markdown** — download the full summary, chapters, and quiz as a `.md` file
- **Copy to clipboard** — copy the full result or individual sections
- **History panel** — browse and reload past results without reprocessing
- **Estimated time** — live countdown calculated from processing progress
- **Keyboard shortcuts** — `Enter` to submit, `Esc` to return to the home screen

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker: command not found` | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Port 8000 already in use | Change port in `docker-compose.yml`: `"8001:8000"` |
| Download fails (403 / auth error) | Use the Chrome extension to send cookies with the request |
| Recording not found (404) | The Zoom link may have expired |
| AI returns malformed JSON | Transient Gemini error — the system retries automatically (up to 3x) |
| Whisper not loading | Not enough RAM — switch to a smaller model in `.env` |
| OpenAI mode: "API key not configured" | Add `OPENAI_API_KEY=sk-...` to your `.env` file |
| OpenAI mode: file too large | This shouldn't happen — the preprocessor chunks to ≤13 min automatically |
| `docker compose up` very slow | First run downloads the image (~3 GB) — wait it out |
| No space left on device | Run `docker system prune -af` to clear unused images |

---

## Project Structure

```
zoom-to-text/
├── app/
│   ├── api/routes.py              # REST API endpoints
│   ├── services/
│   │   ├── transcriber.py         # Faster-Whisper (local) + OpenAI Whisper API
│   │   ├── audio_preprocessor.py  # Silence removal + chunking (used by OpenAI mode)
│   │   ├── summarizer.py          # Gemini AI summarization + quiz
│   │   ├── zoom_downloader.py     # yt-dlp audio extraction
│   │   └── processor.py           # Pipeline orchestrator
│   ├── config.py                  # Settings (loaded from .env)
│   ├── models.py                  # Pydantic schemas
│   ├── state.py                   # SQLite task state manager
│   └── main.py                    # FastAPI app + startup
├── static/
│   ├── index.html                 # Web UI
│   └── style.css
├── extension/                     # Chrome extension
├── data/                          # SQLite DB + temp downloads (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API Reference

The server exposes a REST API at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/tasks` | Start job from a Zoom URL |
| `POST` | `/api/tasks/upload` | Start job from an uploaded file |
| `GET` | `/api/tasks/{id}` | Poll job status and progress |
| `GET` | `/api/tasks` | List recent jobs |
| `DELETE` | `/api/tasks/{id}` | Delete a job record |
| `GET` | `/health` | Server health check |

---

## License

MIT
