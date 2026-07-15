"""
Durable job-queue repository.

The tasks table doubles as a durable job queue: payload_json holds everything
needed to (re)run the pipeline. It is scrubbed of secrets when the task
finishes because it may contain the user's Zoom session cookies.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from app import state
from app.models import TaskStatus

logger = logging.getLogger(__name__)

_IN_FLIGHT_STATUSES = [
    TaskStatus.PENDING.value,
    TaskStatus.DOWNLOADING.value,
    TaskStatus.TRANSCRIBING.value,
    TaskStatus.SUMMARIZING.value,
]


async def set_job_payload(task_id: str, payload: dict) -> None:
    db = await state._get_db()
    await db.execute(
        "UPDATE tasks SET payload_json=? WHERE id=?",
        [json.dumps(payload, ensure_ascii=False), task_id],
    )
    await db.commit()


async def get_job_payload(task_id: str) -> Optional[dict]:
    db = await state._get_db()
    async with db.execute(
        "SELECT payload_json FROM tasks WHERE id=?", [task_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or not row["payload_json"]:
        return None
    return json.loads(row["payload_json"])


async def clear_job_payload(task_id: str) -> None:
    """Wipe the job payload entirely."""
    db = await state._get_db()
    await db.execute("UPDATE tasks SET payload_json=NULL WHERE id=?", [task_id])
    await db.commit()


async def finalize_job_payload(task_id: str) -> None:
    """Called when a task reaches a terminal state.

    Strips the secret (Zoom session cookies) from the stored payload but KEEPS
    the rest (url/file_path/mode/language) so the task can be retried. Cookies
    expire fast and must never linger at rest; everything else is harmless
    metadata and is what POST /tasks/{id}/retry replays.

    A private-recording retry will therefore run without cookies and fail with
    a clear auth error — acceptable, since a stale cookie would have failed the
    download anyway; the user re-submits via the Chrome extension to refresh it.

    One guarded UPDATE (json_set) instead of read-modify-write, so it can't
    race a concurrent payload writer and resurrect the cookies it just scrubbed.
    """
    db = await state._get_db()
    # json('null') rather than SQL NULL: older SQLite (Docker's Debian build)
    # returned NULL from json_set on an SQL NULL value — wiping the payload.
    await db.execute(
        "UPDATE tasks SET payload_json = json_set(payload_json, '$.cookies', json('null')) "
        "WHERE id=? AND payload_json IS NOT NULL AND json_valid(payload_json)",
        [task_id],
    )
    await db.commit()


async def reset_interrupted_tasks() -> list[str]:
    """
    Called on startup by the worker. Tasks that were mid-flight when the server
    died are re-queued if their job payload is intact (and, for uploads, the
    source file still exists on disk); the rest are marked failed.

    Returns the list of task ids to re-enqueue, oldest first.
    """
    db = await state._get_db()
    placeholders = ",".join("?" * len(_IN_FLIGHT_STATUSES))
    async with db.execute(
        f"SELECT id, payload_json FROM tasks WHERE status IN ({placeholders}) "
        "ORDER BY created_at ASC",
        _IN_FLIGHT_STATUSES,
    ) as cursor:
        rows = await cursor.fetchall()

    resumable: list[str] = []
    dead: list[str] = []
    for row in rows:
        payload = None
        if row["payload_json"]:
            try:
                payload = json.loads(row["payload_json"])
            except ValueError:
                payload = None
        file_path = (payload or {}).get("file_path")
        if payload and (file_path is None or Path(file_path).exists()):
            resumable.append(row["id"])
        else:
            dead.append(row["id"])

    if resumable:
        marks = ",".join("?" * len(resumable))
        await db.execute(
            f"UPDATE tasks SET status=?, progress=0, message=? WHERE id IN ({marks})",
            [TaskStatus.PENDING.value, "ממתין בתור (חודש אחרי הפעלה מחדש)"] + resumable,
        )
        logger.warning(f"Re-queued {len(resumable)} interrupted task(s) on startup")
    if dead:
        marks = ",".join("?" * len(dead))
        await db.execute(
            f"UPDATE tasks SET status=?, progress=0, message=?, error=? WHERE id IN ({marks})",
            [
                TaskStatus.FAILED.value,
                "השרת הופעל מחדש — המשימה הופסקה",
                "השרת הופעל מחדש — נסה שוב",
            ] + dead,
        )
        logger.warning(f"Marked {len(dead)} interrupted task(s) as failed on startup")
    await db.commit()
    return resumable


async def list_payload_file_paths() -> list[str]:
    """Every file_path referenced by any stored job payload.

    The retention sweep's orphan pass must never delete a file some task still
    plans to replay — this is the authoritative referenced-set.
    """
    db = await state._get_db()
    async with db.execute(
        "SELECT payload_json FROM tasks WHERE payload_json IS NOT NULL"
    ) as cursor:
        rows = await cursor.fetchall()
    paths: list[str] = []
    for row in rows:
        try:
            file_path = json.loads(row["payload_json"]).get("file_path")
        except ValueError:
            continue
        if file_path:
            paths.append(file_path)
    return paths
