"""
REST API endpoints.

POST /api/tasks          — start a job from a Zoom URL
POST /api/tasks/upload   — start a job from an uploaded audio/video file
GET  /api/tasks          — list recent jobs
GET  /api/tasks/{id}     — get job status + result
DELETE /api/tasks/{id}   — delete a job record
"""
import json
import re
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app import state
from app.api.deps import get_current_user
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse
from app.services import anki_export, processor, summarizer
from app.services.exporters.markdown import build_obsidian_markdown
from app.services.llm_providers import get_provider
from app.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed audio/video extensions for upload
_ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".mkv", ".webm", ".avi"}


# ── Rate-limit string helper ──────────────────────────────────────────────────────
# Returns a slowapi limit string like "10/minute". Read at request time so that
# tests can monkeypatch settings.rate_limit_per_minute without restarting.
def _task_rate_limit(request: Request) -> str:  # noqa: ARG001 — request required by slowapi
    n = settings.rate_limit_per_minute
    return f"{n}/minute"


# ── Start job from URL ────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202)
@limiter.limit(_task_rate_limit)
async def create_task(
    request: Request,
    task_in: TaskCreate,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    """
    Submit a Zoom recording URL for processing.

    - **url**: Zoom recording link (public or private)
    - **mode**: `gemini_direct` (fast) or `whisper_local` (private/offline)
    - **cookies**: Netscape-format cookie string from the Chrome extension
                   (required for institutional recordings like ORT)
    - **language**: Audio language hint — `he` (Hebrew), `en`, or `auto`
    """
    task_id = str(uuid.uuid4())
    task = await state.create_task(task_id, task_in.url, user_id=user_id)

    background_tasks.add_task(
        processor.run_pipeline,
        task_id=task_id,
        url=task_in.url,
        mode=task_in.mode,
        cookies=task_in.cookies,
        language=task_in.language,
    )
    return task


# ── Start job from uploaded file ──────────────────────────────────────────────────

@router.post("/tasks/upload", response_model=TaskResponse, status_code=202)
@limiter.limit(_task_rate_limit)
async def create_task_from_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: ProcessingMode = Form(ProcessingMode.GEMINI_DIRECT),
    language: str = Form("he"),
    user_id: str = Depends(get_current_user),
):
    """
    Upload an audio or video file directly for processing.
    Supported formats: mp3, mp4, m4a, wav, mkv, webm
    """
    task_id = str(uuid.uuid4())

    # Validate file type
    safe_name = Path(file.filename).name if file.filename else "upload"
    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # Stream file to disk in 1 MB chunks — avoids loading a 3-hour recording into RAM
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.downloads_dir / f"{task_id}_{safe_name}"

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {settings.max_upload_bytes // 1024 // 1024} MB.",
                    )
                await f.write(chunk)
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise

    task = await state.create_task(task_id, f"upload:{safe_name}", user_id=user_id)

    background_tasks.add_task(
        processor.run_pipeline_from_file,
        task_id=task_id,
        file_path=str(file_path),
        mode=mode,
        language=language,
    )
    return task


# ── Query tasks ───────────────────────────────────────────────────────────────────

@router.get("/tasks", response_model=list)
async def list_tasks(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, max_length=200),
    user_id: str = Depends(get_current_user),
):
    """Return recent processing jobs (newest first) with optional search and pagination."""
    return await state.list_tasks(limit=limit, user_id=user_id, search=search, offset=offset)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, user_id: str = Depends(get_current_user)):
    """
    Poll this endpoint to check job progress.
    Frontend polls every 2 seconds until status is `completed` or `failed`.
    Returns 404 if not found or owned by a different user (prevents enumeration).
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks/{task_id}/cancel", status_code=200)
async def cancel_task(task_id: str, user_id: str = Depends(get_current_user)):
    """
    Cancel an in-progress task. Only works on tasks that are still running
    (pending / downloading / transcribing / summarizing).
    Returns 400 if the task is already finished or not found.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    cancelled = await state.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Task is already finished and cannot be cancelled")
    return {"status": "cancelled"}


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, user_id: str = Depends(get_current_user)):
    """
    Delete a task record from the database (only the owning user may delete).
    Also removes the persistent audio file on disk (Feature 7).
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # Remove the audio file first — even if the DB delete fails, we've freed the disk
    audio_path = await state.get_audio_path(task_id)
    if audio_path:
        try:
            p = Path(audio_path)
            if p.exists() and _path_under_audio_root(p):
                p.unlink()
        except Exception as exc:
            logger.warning(f"Could not remove audio for {task_id}: {exc}")
    await state.delete_task(task_id)


# ── Live transcript preview ───────────────────────────────────────────────────────

@router.get("/tasks/{task_id}/transcript")
async def get_partial_transcript(
    task_id: str,
    offset: int = Query(0, ge=0, description="Character offset — return only text after this position"),
    user_id: str = Depends(get_current_user),
):
    """
    Return the live partial transcript delta for WHISPER-mode tasks.

    Poll this endpoint while status == 'transcribing' to stream transcript
    text as it is produced. Use ?offset=N to get only the new characters
    since the last poll — the frontend tracks the offset locally and sends
    it on each request so only the delta is transferred.

    Response: {"text": "<new chars>", "total": <total length so far>}
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    delta, total = await state.get_partial_transcript(task_id, from_offset=offset)
    return {"text": delta, "total": total}


# ── Chat with transcript ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., description="Question about the lesson content")


@router.post("/tasks/{task_id}/ask")
async def ask_question(task_id: str, body: AskRequest, user_id: str = Depends(get_current_user)):
    """
    Ask a question about a completed lesson.
    Uses the stored summary + chapters as context for a Gemini-powered answer.
    Returns 404 if not found or owned by a different user.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet — wait for processing to complete")

    # Build context from the stored result
    context_parts = []
    if task.result.summary:
        context_parts.append(f"סיכום:\n{task.result.summary}")
    for ch in task.result.chapters:
        context_parts.append(f"\nפרק: {ch.title}\n{ch.content}")
        if ch.key_points:
            context_parts.append("נקודות מרכזיות: " + ", ".join(ch.key_points))
    if task.result.transcript:
        # Include up to 30k chars of transcript for richer answers
        context_parts.append(f"\nתמלול:\n{task.result.transcript[:30000]}")

    context = "\n".join(context_parts)

    try:
        answer = await summarizer.ask_about_lesson(context, body.question)
        return {"answer": answer}
    except Exception as exc:
        logger.error(f"Ask failed for task {task_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chat with recording (multi-turn, streaming) ───────────────────────────────────

def _build_lesson_context(result) -> str:
    """Build a rich context string from a completed lesson result."""
    parts = []
    if result.summary:
        parts.append(f"סיכום:\n{result.summary}")
    for ch in result.chapters:
        parts.append(f"\nפרק: {ch.title}\n{ch.content}")
        if ch.key_points:
            parts.append("נקודות מרכזיות: " + ", ".join(ch.key_points))
    if result.transcript:
        # Include up to 30 k chars of transcript for richer answers
        parts.append(f"\nתמלול:\n{result.transcript[:30_000]}")
    return "\n".join(parts)


@router.post("/tasks/{task_id}/chat")
async def chat_with_recording(
    task_id: str,
    body: AskRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Multi-turn streaming chat about a completed lesson.

    Returns a Server-Sent Events stream where each event carries a JSON payload:
      {"text": "<chunk>"}   — partial model response
      {"done": true}        — stream finished (no more events)
      {"error": "<msg>"}    — error occurred

    The user message and final model response are stored in SQLite so the
    history survives page reloads. History is capped at 40 messages (state.py).
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(
            status_code=400, detail="Task has no result yet — wait for processing to complete"
        )

    context = _build_lesson_context(task.result)
    history = await state.get_chat_history(task_id)

    # Persist the user message before streaming starts
    await state.append_chat_message(task_id, "user", body.question)

    async def generate():
        full_response: list[str] = []
        try:
            async for chunk in summarizer.stream_chat_response(context, history, body.question):
                full_response.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"Chat stream failed for task {task_id}: {exc}")
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            if full_response:
                await state.append_chat_message(task_id, "model", "".join(full_response))
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Cache-Control": "no-cache",
        },
    )


@router.get("/tasks/{task_id}/chat")
async def get_chat_history(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Return the stored chat history for a completed task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    history = await state.get_chat_history(task_id)
    return {"history": history}


@router.delete("/tasks/{task_id}/chat", status_code=204)
async def clear_chat_history(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Clear the chat history for a task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await state.clear_chat_history(task_id)


# ── Audio streaming (Feature 7) ──────────────────────────────────────────────────

# Persistent audio root — files live here after processing so the UI player can
# seek back into them. Matches processor._audio_root() exactly.
_AUDIO_ROOT = (settings.data_dir / "audio").resolve()


def _path_under_audio_root(p: Path) -> bool:
    """
    Return True iff p resolves to a location strictly inside _AUDIO_ROOT.

    Prevents directory traversal — even if somebody managed to inject a
    symlink or a ../../-laden path into the DB, we refuse to serve it.
    """
    try:
        return p.resolve().is_relative_to(_AUDIO_ROOT)
    except Exception:
        return False


# Content-type map: keep small, known-safe list (no arbitrary mimetypes.guess_type)
_AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def _parse_range(header: str | None, file_size: int) -> tuple[int, int] | None:
    """
    Parse a single-range 'Range: bytes=a-b' request.
    Returns (start, end) inclusive, or None if the header is missing/invalid.
    Multi-range requests are not supported — we serve the first range only.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].split(",", 1)[0].strip()
    try:
        start_s, end_s = spec.split("-", 1)
        if start_s == "":
            # suffix: "bytes=-500" → last 500 bytes
            length = int(end_s)
            if length <= 0:
                return None
            start = max(file_size - length, 0)
            end   = file_size - 1
        else:
            start = int(start_s)
            end   = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start < 0 or end >= file_size or start > end:
        return None
    return start, end


@router.get("/tasks/{task_id}/audio")
async def stream_audio(
    task_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """
    Stream the persistent audio file for a task with HTTP Range support.

    Range support is required for HTML5 <audio> seek to work — without it,
    the browser can only play from byte 0 and seekbar drags are no-ops.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    audio_path = await state.get_audio_path(task_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="No audio stored for this task")

    p = Path(audio_path)
    if not p.exists() or not _path_under_audio_root(p):
        # Either the file was wiped or the path is suspicious — both 404
        raise HTTPException(status_code=404, detail="Audio file unavailable")

    file_size = p.stat().st_size
    media_type = _AUDIO_TYPES.get(p.suffix.lower(), "application/octet-stream")

    range_header = request.headers.get("range")
    rng = _parse_range(range_header, file_size)

    async def _iter(start: int, length: int, chunk: int = 64 * 1024):
        # aiofiles for async I/O; small per-chunk read keeps memory bounded
        async with aiofiles.open(p, "rb") as f:
            await f.seek(start)
            remaining = length
            while remaining > 0:
                data = await f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if rng is None:
        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline",
        }
        return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)

    start, end = rng
    length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": "inline",
    }
    return StreamingResponse(
        _iter(start, length), status_code=206, media_type=media_type, headers=headers
    )


# ── Flashcards ────────────────────────────────────────────────────────────────────

def _sanitize_deck_name(task_id: str, url: str | None) -> str:
    """Build a readable deck name from the task. Stripped of filesystem-nasty chars."""
    base = (url or "").strip() or task_id
    # Collapse to a short label — Anki truncates long deck names awkwardly
    base = re.sub(r"https?://", "", base)[:60]
    base = re.sub(r"[^\w\-֐-׿ .]", " ", base).strip()
    return f"ZoomToText — {base or task_id[:8]}"


@router.get("/tasks/{task_id}/flashcards")
async def get_flashcards(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Return the generated flashcards for a completed task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet")
    return {"flashcards": [c.model_dump() for c in task.result.flashcards]}


@router.get("/tasks/{task_id}/flashcards/export.apkg")
async def export_flashcards_apkg(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Download the task's flashcards as an Anki .apkg package."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None or not task.result.flashcards:
        raise HTTPException(status_code=400, detail="No flashcards to export")

    deck_name = _sanitize_deck_name(task_id, task.url)
    data = anki_export.create_apkg(task.result.flashcards, deck_name, task_id)
    filename = f"flashcards-{task_id[:8]}.apkg"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/tasks/{task_id}/flashcards/export.csv")
async def export_flashcards_csv(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Download the task's flashcards as UTF-8 CSV (for users who don't use Anki)."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None or not task.result.flashcards:
        raise HTTPException(status_code=400, detail="No flashcards to export")

    data = anki_export.create_csv(task.result.flashcards)
    filename = f"flashcards-{task_id[:8]}.csv"
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Obsidian-flavored markdown export (Task 1.4) ─────────────────────────────

@router.get("/tasks/{task_id}/export/obsidian")
async def export_obsidian_markdown(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Download the task's lesson as Obsidian-flavored Markdown.

    Includes YAML frontmatter, action items as `- [ ]` checkboxes with
    `#action/<owner>` tags, decisions / open questions / sentiment /
    objections sections, chapters, and the exam in a `<details>` block.
    The original client-side Markdown export remains unchanged.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet")

    md = build_obsidian_markdown(task)
    filename = f"obsidian-{task_id[:8]}.md"
    return Response(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── PDF export (Task 5) ───────────────────────────────────────────────────────

@router.get("/tasks/{task_id}/export/pdf")
async def export_pdf(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Download the task's lesson as a PDF document.

    Converts Obsidian Markdown → HTML → PDF via WeasyPrint.
    Requires system libs (Pango, Cairo, fonts-dejavu-core) installed in Docker.
    Returns 503 if WeasyPrint is unavailable on the current host.

    RAM note: WeasyPrint uses ~80-150 MB during rendering. On the 512 MB
    Fly.io machine this is safe when Whisper is idle, but concurrent PDF
    generation + transcription may OOM. Acceptable for ≤6 users.
    """
    import asyncio as _asyncio

    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet")

    try:
        from app.services.exporters.pdf import build_pdf
        # Run in thread pool — WeasyPrint's Pango layout is CPU-bound
        pdf_bytes = await _asyncio.to_thread(build_pdf, task)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"PDF export not available: {exc}")

    filename = f"lesson-{task_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Share links ───────────────────────────────────────────────────────────────

@router.post("/tasks/{task_id}/share")
async def create_share_link(
    task_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """
    Generate (or return existing) a permanent public share token for a completed task.
    The token is embedded in a share URL the caller can give to anyone.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task is not complete yet")

    token = await state.create_share_token(task_id)
    base = str(request.base_url).rstrip("/")
    return {"share_url": f"{base}/share/{token}"}


@router.get("/share/{token}")
async def get_shared_task(token: str):
    """
    Public endpoint — no auth required.
    Returns the lesson result for a share token so the frontend can render it.
    """
    task = await state.get_task_by_share_token(token)
    if task is None:
        raise HTTPException(status_code=404, detail="Share link not found or task incomplete")
    return task


# ── Provider capabilities (UI uses this to filter the mode dropdown) ──────────

_ALL_MODES = ["gemini_direct", "whisper_local", "whisper_api", "ivrit_ai"]


def _available_modes_for(provider) -> list[str]:
    """Filter the four processing modes by what the provider can actually do."""
    if provider.supports_audio_upload:
        return list(_ALL_MODES)
    return [m for m in _ALL_MODES if m != "gemini_direct"]


@router.get("/capabilities")
async def get_capabilities() -> dict:
    """
    Public read-only endpoint describing the active LLM provider's capabilities.
    Frontend calls this on page load to know which processing modes to offer.
    """
    p = get_provider()
    return {
        "llm_provider": p.name,
        "supports_audio_upload": p.supports_audio_upload,
        "supports_streaming": p.supports_streaming,
        "available_modes": _available_modes_for(p),
    }
