"""
REST API endpoints.

POST /api/tasks          — start a job from a Zoom URL
POST /api/tasks/upload   — start a job from an uploaded audio/video file
GET  /api/tasks          — list recent jobs
GET  /api/tasks/{id}     — get job status + result
DELETE /api/tasks/{id}   — delete a job record
"""
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app import state
from app.api.deps import get_current_user
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse
from app.services import processor, summarizer

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
    """
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str, user_id: str = Depends(get_current_user)):
    """Delete a task record from the database."""
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await state.delete_task(task_id)


# ── Chat with transcript ───────────────────────────────────────────────────────────

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., description="Question about the lesson content")


@router.post("/tasks/{task_id}/ask")
async def ask_question(task_id: str, body: AskRequest, user_id: str = Depends(get_current_user)):
    """
    Ask a question about a completed lesson.
    Uses the stored summary + chapters as context for a Gemini-powered answer.
    """
    task = await state.get_task(task_id)
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
