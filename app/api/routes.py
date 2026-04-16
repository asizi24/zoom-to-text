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
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app import state
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse
from app.services import processor

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Start job from URL ────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202)
async def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks):
    """
    Submit a Zoom recording URL for processing.

    - **url**: Zoom recording link (public or private)
    - **mode**: `gemini_direct` (fast) or `whisper_local` (private/offline)
    - **cookies**: Netscape-format cookie string from the Chrome extension
                   (required for institutional recordings like ORT)
    - **language**: Audio language hint — `he` (Hebrew), `en`, or `auto`
    """
    task_id = str(uuid.uuid4())
    task = await state.create_task(task_id, task_in.url)

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
):
    """
    Upload an audio or video file directly for processing.
    Supported formats: mp3, mp4, m4a, wav, mkv, webm
    """
    task_id = str(uuid.uuid4())

    # Stream file to disk in 1 MB chunks — avoids loading a 3-hour recording into RAM
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name if file.filename else "upload"
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

    task = await state.create_task(task_id, f"upload:{safe_name}")

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
async def list_tasks(limit: int = 20):
    """Return the most recent N processing jobs (newest first)."""
    return await state.list_tasks(limit=limit)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Poll this endpoint to check job progress.
    Frontend polls every 2 seconds until status is `completed` or `failed`.
    """
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task(task_id: str):
    """Delete a task record from the database."""
    task = await state.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await state.delete_task(task_id)
