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
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app import state
from app.api.deps import get_current_user
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse
from app.services import anki_export, processor, summarizer

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed audio/video extensions for upload
_ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".mkv", ".webm", ".avi"}


# ── Start job from URL ────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202)
async def create_task(
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
async def create_task_from_upload(
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
async def list_tasks(limit: int = 20, user_id: str = Depends(get_current_user)):
    """Return the most recent N processing jobs (newest first)."""
    return await state.list_tasks(limit=limit, user_id=user_id)


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


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, user_id: str = Depends(get_current_user)):
    """Delete a task record from the database (only the owning user may delete)."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
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
