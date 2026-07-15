"""
Task lifecycle endpoints.

POST   /tasks               — start a job from a Zoom URL
POST   /tasks/upload        — start a job from an uploaded audio/video file
GET    /tasks               — list recent jobs
GET    /tasks/{id}          — get job status + result
DELETE /tasks/{id}          — delete a job record (+ its media on disk)
POST   /tasks/{id}/cancel   — cooperative cancellation
POST   /tasks/{id}/retry    — re-queue a failed/cancelled task
"""
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import cancellation, state
from app.api.deps import get_current_user
from app.api.routers.audio import _path_under_audio_root
from app.config import settings
from app.models import ProcessingMode, TaskCreate, TaskResponse, TaskStatus
from app.ratelimit import rate_limit
from app.services import worker
from app.services.zoom_downloader import ZoomDownloadError, ensure_url_allowed

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed audio/video extensions for upload
_ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".mkv", ".webm", ".avi"}

# Task submissions queue heavy pipeline work — shared budget across the three
# entry points (URL, upload, retry), keyed per client IP.
_tasks_rate_limit = Depends(rate_limit("tasks", "rate_limit_tasks_per_minute"))


# ── Start job from URL ────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse, status_code=202,
             dependencies=[_tasks_rate_limit])
async def create_task(
    task_in: TaskCreate,
    user_id: str = Depends(get_current_user),
):
    """
    Submit a Zoom recording URL for processing.

    - **url**: Zoom recording link (public or private)
    - **mode**: `gemini_direct` (fast) or `whisper_local` (private/offline)
    - **cookies**: Netscape-format cookie string from the Chrome extension
                   (required for institutional recordings like ORT)
    - **language**: Audio language hint — `he` (Hebrew), `en`, or `auto`

    Returns 202; the job runs on the worker queue (at most
    settings.pipeline_concurrency at a time) and survives server restarts.
    """
    # Vet the URL now so a blocked target is an immediate 400 instead of a
    # queued task that fails later (the downloader re-checks post-normalization).
    try:
        await ensure_url_allowed(task_in.url)
    except ZoomDownloadError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc

    task_id = str(uuid.uuid4())
    task = await state.create_task(task_id, task_in.url, user_id=user_id)
    await state.set_job_payload(task_id, {
        "url": task_in.url,
        "mode": task_in.mode.value,
        "cookies": task_in.cookies,
        "language": task_in.language,
    })
    worker.enqueue(task_id)
    return task


# ── Start job from uploaded file ──────────────────────────────────────────────────

@router.post("/tasks/upload", response_model=TaskResponse, status_code=202,
             dependencies=[_tasks_rate_limit])
async def create_task_from_upload(
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
    except Exception:
        # Any write failure (413 size limit, client disconnect, disk full) must
        # not leave a partial file behind — nothing references it yet.
        file_path.unlink(missing_ok=True)
        raise

    try:
        task = await state.create_task(task_id, f"upload:{safe_name}", user_id=user_id)
        await state.set_job_payload(task_id, {
            "file_path": str(file_path),
            "mode": mode.value,
            "language": language,
        })
        worker.enqueue(task_id)
    except Exception:
        # Task row/payload/queue failed after the bytes hit disk — without
        # this, the file is an orphan nothing references (the retention sweep
        # only reclaims payload-referenced media).
        file_path.unlink(missing_ok=True)
        raise
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
    # Also reclaim the retained upload source (kept for /retry on failed or
    # cancelled tasks) — deleting the task removes the only reference to it.
    payload = await state.get_job_payload(task_id)
    source_path = (payload or {}).get("file_path")
    if source_path:
        try:
            p = Path(source_path)
            if p.exists() and p.resolve().is_relative_to(settings.downloads_dir.resolve()):
                p.unlink()
        except Exception as exc:
            logger.warning(f"Could not remove upload source for {task_id}: {exc}")
    await state.delete_task(task_id)


# ── Cancel / retry ────────────────────────────────────────────────────────────────

# Statuses from which a task can still be cancelled (i.e. it is queued or running).
_CANCELLABLE = {
    TaskStatus.PENDING,
    TaskStatus.DOWNLOADING,
    TaskStatus.TRANSCRIBING,
    TaskStatus.SUMMARIZING,
}
# Terminal statuses a task can be retried from.
_RETRYABLE = {TaskStatus.FAILED, TaskStatus.CANCELLED}


@router.post("/tasks/{task_id}/cancel", status_code=200)
async def cancel_task(task_id: str, user_id: str = Depends(get_current_user)):
    """
    Request cancellation of a queued or running task.

    Cancellation is cooperative: this flips the DB status to CANCELLED and sets
    an in-process flag; the worker/transcriber notice it at the next checkpoint
    (between pipeline steps, or once per transcription segment) and unwind —
    releasing the GPU/model slot and any temp audio. Returns 409 if the task
    has already reached a terminal state.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in _CANCELLABLE:
        raise HTTPException(
            status_code=409,
            detail=f"Task is not running (status: {task.status.value})",
        )
    cancellation.request_cancel(task_id)
    if not await state.cancel_task(task_id):
        # The task reached a terminal state between our check above and the
        # guarded UPDATE — the finished state wins; undo the in-process flag.
        cancellation.clear(task_id)
        raise HTTPException(
            status_code=409, detail="Task finished before it could be cancelled"
        )
    return {"status": TaskStatus.CANCELLED.value}


@router.post("/tasks/{task_id}/retry", response_model=TaskResponse, status_code=202,
             dependencies=[_tasks_rate_limit])
async def retry_task(task_id: str, user_id: str = Depends(get_current_user)):
    """
    Re-queue a FAILED or CANCELLED task for a fresh run.

    Replays the stored job payload (URL or uploaded file path). For uploads the
    original file must still be on disk — it is retained on failure/cancel for
    exactly this, but the retention sweep reclaims it after
    settings.media_retention_days, after which retry returns 409.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in _RETRYABLE:
        raise HTTPException(
            status_code=409,
            detail=f"Only failed or cancelled tasks can be retried (status: {task.status.value})",
        )

    payload = await state.get_job_payload(task_id)
    if payload is None:
        raise HTTPException(
            status_code=409, detail="Cannot retry — original job parameters are no longer available"
        )
    file_path = payload.get("file_path")
    if file_path and not Path(file_path).exists():
        raise HTTPException(
            status_code=409, detail="Cannot retry — the uploaded file is no longer on disk"
        )

    # Clear any stale cancellation flag, reset the row to PENDING, re-enqueue.
    cancellation.clear(task_id)
    await state.requeue_task(task_id)
    worker.enqueue(task_id)
    return await state.get_task_for_user(task_id, user_id)
