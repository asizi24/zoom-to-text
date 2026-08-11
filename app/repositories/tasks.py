"""
Task-row repository: CRUD, sticky terminal states, live transcript preview,
audio-path tracking, and the retention queries.

Terminal statuses are STICKY: update_task/complete_task/fail_task refuse to
overwrite them (guarded UPDATE ... WHERE status NOT IN terminal). This closes
the cancel races in one place — a progress callback still in flight from the
transcription thread, or a complete/fail landing moments after the user hit
cancel, can no longer resurrect or overwrite a CANCELLED row. The only paths
out of a terminal state are the deliberate ones: requeue_task (retry) and
jobs.reset_interrupted_tasks (restart recovery), which target statuses
explicitly.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from app import state
from app.events import hub as _events
from app.models import LessonResult, TaskResponse, TaskStatus
from app.state import get_write_lock

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = (
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
)
_NOT_TERMINAL_GUARD = (
    f"status NOT IN ({','.join('?' * len(_TERMINAL_STATUSES))})"
)


# ── CRUD ──────────────────────────────────────────────────────────────────────────

async def create_task(task_id: str, url: str, user_id: Optional[str] = None) -> TaskResponse:
    async with get_write_lock():
        now = datetime.now(timezone.utc).isoformat()
        db = await state._get_db()
        await db.execute(
            "INSERT INTO tasks (id, status, progress, message, created_at, url, user_id) VALUES (?,?,?,?,?,?,?)",
            [task_id, TaskStatus.PENDING.value, 0, "Task queued", now, url, user_id],
        )
        await db.commit()
        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0,
            message="Task queued",
            created_at=now,
            url=url,
        )


async def update_task(task_id: str, status: TaskStatus, progress: int = 0, message: str = ""):
    async with get_write_lock():
        db = await state._get_db()
        await db.execute(
            "UPDATE tasks SET status=?, progress=?, message=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            [status.value, progress, message, task_id]
        )
        await db.commit()


async def append_partial_transcript(task_id: str, text: str):
    async with get_write_lock():
        db = await state._get_db()
        await db.execute(
            "INSERT INTO partial_transcripts (task_id, transcript_text) VALUES (?, ?)",
            [task_id, text]
        )
        await db.commit()


async def complete_task(task_id: str, result: LessonResult) -> bool:
    """Mark a task completed. Returns False (and stores nothing) if the task
    already reached a terminal state — e.g. a cancel won the race in the final
    moments of the pipeline. The SSE `done` event is only published on a real
    transition; the losing side already published its own."""
    async with get_write_lock():
        db = await state._get_db()
        # Clear the live-preview column on completion — the full transcript is
        # stored in result_json, so partial_transcript is no longer needed and
        # would only waste space in the DB.
        cursor = await db.execute(
            f"UPDATE tasks SET status=?, progress=100, message=?, result_json=?, "
            f"partial_transcript=NULL WHERE id=? AND {_NOT_TERMINAL_GUARD}",
            [TaskStatus.COMPLETED.value, "Processing complete ✅",
             result.model_dump_json(), task_id, *_TERMINAL_STATUSES],
        )
        await db.commit()
        if cursor.rowcount == 0:
            return False
        _events.publish(task_id, {"type": "done", "status": TaskStatus.COMPLETED.value})
        return True


async def fail_task(task_id: str, error: str, detail: str = "") -> bool:
    """Mark a task failed. `error` is the user-facing message; `detail` keeps
    the technical cause in error_detail so failures are debuggable later.
    Returns False if the task was already terminal (e.g. cancelled) — the
    existing terminal state wins and no event is published."""
    # Truncate long error messages so they fit cleanly in the DB
    short_error = error[:500] if len(error) > 500 else error
    async with get_write_lock():
        db = await state._get_db()
        cursor = await db.execute(
            f"UPDATE tasks SET status=?, message=?, error=?, error_detail=? "
            f"WHERE id=? AND {_NOT_TERMINAL_GUARD}",
            [TaskStatus.FAILED.value, f"Failed: {short_error}", short_error,
             detail[:2000], task_id, *_TERMINAL_STATUSES],
        )
        await db.commit()
        if cursor.rowcount == 0:
            return False
        _events.publish(task_id, {"type": "done", "status": TaskStatus.FAILED.value})
        return True


async def cancel_task(task_id: str) -> bool:
    """Mark a task cancelled (user-requested abort).

    The actual stop is cooperative — the worker/transcriber notice the flag in
    app.cancellation and unwind — but the DB status flips to CANCELLED now so
    the UI reflects the click immediately. `done` closes any open SSE stream.
    Returns False if the task finished (completed/failed) between the caller's
    status check and this write — the finished state wins."""
    async with get_write_lock():
        db = await state._get_db()
        cursor = await db.execute(
            f"UPDATE tasks SET status=?, message=? WHERE id=? AND {_NOT_TERMINAL_GUARD}",
            [TaskStatus.CANCELLED.value, "בוטל על ידי המשתמש", task_id, *_TERMINAL_STATUSES],
        )
        await db.commit()
        if cursor.rowcount == 0:
            return False
        _events.publish(task_id, {"type": "done", "status": TaskStatus.CANCELLED.value})
        return True


async def requeue_task(task_id: str) -> None:
    """Reset a FAILED/CANCELLED task back to PENDING for a fresh run.

    Clears the previous error so the retry starts clean. The partial transcript,
    payload_json (final_transcript), and audio_path are deliberately left intact
    so that _process_audio can detect a mid-transcription resume point or a
    cached full transcript and avoid redundant work.
    """
    async with get_write_lock():
        db = await state._get_db()
    msg = "ממתין בתור (ניסיון חוזר)"
    await db.execute(
        "UPDATE tasks SET status=?, progress=0, message=?, error=NULL, "
        "error_detail=NULL WHERE id=?",
        [TaskStatus.PENDING.value, msg, task_id],
    )
    await db.commit()
    _events.publish(task_id, {
        "type": "status",
        "status": TaskStatus.PENDING.value,
        "progress": 0,
        "message": msg,
    })

    # Keep any cancellation flag from a previous run from affecting the retry.
    # This is intentionally done here so the retry endpoint remains idempotent
    # and the new run starts cleanly.
    cancellation.clear(task_id)


def _row_to_task_response(row: aiosqlite.Row) -> TaskResponse:
    """Convert a tasks row into the API response model — single source of truth
    for the mapping (get_task and get_task_for_user must never drift apart)."""
    result = None
    if row["result_json"]:
        result = LessonResult.model_validate_json(row["result_json"])

    audio_path = row["audio_path"] if "audio_path" in row.keys() else None
    has_audio = bool(audio_path) and Path(audio_path).exists()

    return TaskResponse(
        task_id=row["id"],
        status=TaskStatus(row["status"]),
        progress=row["progress"],
        message=row["message"],
        created_at=row["created_at"],
        url=row["url"],
        result=result,
        error=row["error"],
        has_audio=has_audio,
    )


async def get_task(task_id: str) -> Optional[TaskResponse]:
    db = await state._get_db()
    async with db.execute("SELECT * FROM tasks WHERE id=?", [task_id]) as cursor:
        row = await cursor.fetchone()
    return _row_to_task_response(row) if row is not None else None


async def get_task_for_user(task_id: str, user_id: str) -> Optional[TaskResponse]:
    """
    Return a task only if it belongs to user_id.
    Returns None if not found OR if owned by a different user — both look like 404
    to prevent task-id enumeration across users.

    Ownerless legacy rows are NOT matched: they are adopted at startup by
    backfill_task_owners(), so an unowned row here means an anomaly, not a
    grant-to-everyone.
    """
    db = await state._get_db()
    async with db.execute(
        "SELECT * FROM tasks WHERE id=? AND user_id=?",
        [task_id, user_id],
    ) as cursor:
        row = await cursor.fetchone()
    return _row_to_task_response(row) if row is not None else None


async def backfill_task_owners(owner_user_id: str) -> int:
    """One-time adoption of legacy rows created before per-user ownership.

    Called at startup (app/main.py) with the first whitelisted user. Without
    this, dropping the old `OR user_id IS NULL` access rule would strand those
    rows; with the old rule, every logged-in user could read AND delete them.
    Returns the number of rows adopted.
    """
    async with get_write_lock():
        db = await state._get_db()
        cursor = await db.execute(
            "UPDATE tasks SET user_id=? WHERE user_id IS NULL", [owner_user_id]
        )
        await db.commit()
        return cursor.rowcount


async def list_tasks(limit: int = 50, user_id: Optional[str] = None) -> list[dict]:
    db = await state._get_db()
    if user_id:
        async with db.execute(
            "SELECT id, status, progress, message, created_at, url FROM tasks "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            [user_id, limit],
        ) as cursor:
            rows = await cursor.fetchall()
    else:
        async with db.execute(
            "SELECT id, status, progress, message, created_at, url FROM tasks "
            "ORDER BY created_at DESC LIMIT ?",
            [limit],
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def delete_task(task_id: str):
    db = await state._get_db()
    await db.execute("DELETE FROM tasks WHERE id=?", [task_id])
    await db.commit()


# ── Live transcript preview ───────────────────────────────────────────────────────

# Maximum characters stored in partial_transcript (~500 KB of Hebrew text).
# Prevents runaway growth on very long recordings; the final transcript in
# result_json has no such limit — this only caps the live preview column.
_MAX_PARTIAL_TRANSCRIPT_CHARS = 300_000


async def append_partial_transcript(task_id: str, text: str) -> None:
    """
    Append a chunk of text to the task's live transcript column.
    Called from transcriber.py as segments arrive (WHISPER modes only).
    Uses SQLite's native || string concatenation — safe for concurrent WAL writers.
    Silently drops writes once the column reaches _MAX_PARTIAL_TRANSCRIPT_CHARS.
    """
    if not text:
        return
    db = await state._get_db()
    # Guard: skip write if we are already at or above the size cap to prevent
    # the column from growing unboundedly on multi-hour recordings.
    # RETURNING gives the post-append length so the SSE event carries the
    # authoritative total — the client uses it to detect missed deltas.
    async with db.execute(
        """
        UPDATE tasks
           SET partial_transcript = COALESCE(partial_transcript, '') || ?
         WHERE id = ?
           AND LENGTH(COALESCE(partial_transcript, '')) < ?
     RETURNING LENGTH(partial_transcript)
        """,
        [text, task_id, _MAX_PARTIAL_TRANSCRIPT_CHARS],
    ) as cursor:
        row = await cursor.fetchone()
    await db.commit()
    if row is not None:
        _events.publish(task_id, {"type": "transcript", "text": text, "total": row[0]})


async def get_partial_transcript(task_id: str, from_offset: int = 0) -> tuple[str, int]:
    """
    Return new transcript text since from_offset, plus the current total length.
    Used by the GET /api/tasks/{id}/transcript endpoint so the frontend only
    fetches the delta on each poll rather than the entire growing string.

    Returns (delta_text, total_length).
    """
    db = await state._get_db()
    async with db.execute(
        "SELECT partial_transcript FROM tasks WHERE id=?", [task_id]
    ) as cursor:
        row = await cursor.fetchone()

    if row is None or row["partial_transcript"] is None:
        return "", 0

    full: str = row["partial_transcript"]
    total = len(full)
    delta = full[from_offset:] if from_offset < total else ""
    return delta, total


# ── Audio path tracking (Feature 7) ───────────────────────────────────────────────

async def set_audio_path(task_id: str, audio_path: str) -> None:
    """Store the persistent audio file path so the UI can stream it back."""
    db = await state._get_db()
    await db.execute(
        "UPDATE tasks SET audio_path=? WHERE id=?", [audio_path, task_id]
    )
    await db.commit()


async def get_audio_path(task_id: str) -> Optional[str]:
    """Return the stored audio file path for a task, or None if not set."""
    db = await state._get_db()
    async with db.execute(
        "SELECT audio_path FROM tasks WHERE id=?", [task_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return row["audio_path"]


async def clear_audio_path(task_id: str) -> None:
    """Null the audio_path column after the media file has been reclaimed on
    disk, so has_audio reflects reality and the retention sweep won't re-scan it."""
    db = await state._get_db()
    await db.execute("UPDATE tasks SET audio_path=NULL WHERE id=?", [task_id])
    await db.commit()


# ── Storage retention ──────────────────────────────────────────────────────────────

async def list_reclaimable_media(older_than_iso: str) -> list[dict]:
    """
    Return terminal tasks created before `older_than_iso` that still reference
    media on disk — either the persisted playback audio (audio_path, set for
    completed tasks) or the retained upload source (payload file_path, kept for
    failed/cancelled tasks so they stay retryable).

    The DB row and its transcript/summary in result_json are NOT touched; only
    the heavy audio bytes are candidates for deletion by the caller.

    Returns [{"id", "audio_path", "payload_file_path"}].
    """
    db = await state._get_db()
    marks = ",".join("?" * len(_TERMINAL_STATUSES))
    async with db.execute(
        f"SELECT id, audio_path, payload_json FROM tasks "
        f"WHERE created_at < ? AND status IN ({marks}) "
        f"AND (audio_path IS NOT NULL OR payload_json IS NOT NULL)",
        [older_than_iso, *_TERMINAL_STATUSES],
    ) as cursor:
        rows = await cursor.fetchall()

    out: list[dict] = []
    for row in rows:
        file_path = None
        if row["payload_json"]:
            try:
                file_path = json.loads(row["payload_json"]).get("file_path")
            except ValueError:
                file_path = None
        if row["audio_path"] or file_path:
            out.append({
                "id": row["id"],
                "audio_path": row["audio_path"],
                "payload_file_path": file_path,
            })
    return out
