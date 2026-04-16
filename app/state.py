"""
Persistent task state manager backed by SQLite (via aiosqlite).

Why SQLite instead of an in-memory dict?
  - Tasks survive server restarts (the original bug: "Server restarted during processing")
  - Safe for concurrent async reads/writes
  - Zero infrastructure — no Redis, no Celery, no extra containers
  - The DB file lives in data/tasks.db which is mounted as a Docker volume
"""
import aiosqlite
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models import TaskStatus, TaskResponse, LessonResult

logger = logging.getLogger(__name__)

DB_PATH = settings.data_dir / "tasks.db"

# ── Schema ───────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    status      TEXT    NOT NULL DEFAULT 'pending',
    progress    INTEGER NOT NULL DEFAULT 0,
    message     TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    url         TEXT,
    result_json TEXT,
    error       TEXT
)
"""


# ── Lifecycle ─────────────────────────────────────────────────────────────────────

async def init_db():
    """Create the DB and table on startup. Mark interrupted tasks as failed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute(CREATE_TABLE_SQL)
        await db.commit()

    await _mark_interrupted_tasks_failed()
    logger.info(f"Database ready: {DB_PATH}")


async def _mark_interrupted_tasks_failed():
    """
    On startup, any task that was mid-flight when the server died is marked failed.
    This replaces the old behavior of silently leaving tasks stuck in 'downloading'
    state forever after a crash.
    """
    in_flight = [
        TaskStatus.PENDING.value,
        TaskStatus.DOWNLOADING.value,
        TaskStatus.TRANSCRIBING.value,
        TaskStatus.SUMMARIZING.value,
    ]
    placeholders = ",".join("?" * len(in_flight))
    async with aiosqlite.connect(DB_PATH) as db:
        result = await db.execute(
            f"UPDATE tasks SET status=?, progress=0, message=? WHERE status IN ({placeholders})",
            [TaskStatus.FAILED.value, "Server restarted — task was interrupted"] + in_flight,
        )
        if result.rowcount:
            logger.warning(f"Marked {result.rowcount} interrupted task(s) as failed on startup")
        await db.commit()


# ── CRUD ──────────────────────────────────────────────────────────────────────────

async def create_task(task_id: str, url: str) -> TaskResponse:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tasks (id, status, progress, message, created_at, url) VALUES (?,?,?,?,?,?)",
            [task_id, TaskStatus.PENDING.value, 0, "Task queued", now, url],
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


async def update_task(task_id: str, status: TaskStatus, progress: int, message: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET status=?, progress=?, message=? WHERE id=?",
            [status.value, progress, message, task_id],
        )
        await db.commit()


async def complete_task(task_id: str, result: LessonResult):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET status=?, progress=100, message=?, result_json=? WHERE id=?",
            [TaskStatus.COMPLETED.value, "Processing complete ✅", result.model_dump_json(), task_id],
        )
        await db.commit()


async def fail_task(task_id: str, error: str):
    # Truncate long error messages so they fit cleanly in the DB
    short_error = error[:500] if len(error) > 500 else error
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tasks SET status=?, message=?, error=? WHERE id=?",
            [TaskStatus.FAILED.value, f"Failed: {short_error}", short_error, task_id],
        )
        await db.commit()


async def get_task(task_id: str) -> Optional[TaskResponse]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM tasks WHERE id=?", [task_id]) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None

    result = None
    if row["result_json"]:
        result = LessonResult.model_validate_json(row["result_json"])

    return TaskResponse(
        task_id=row["id"],
        status=TaskStatus(row["status"]),
        progress=row["progress"],
        message=row["message"],
        created_at=row["created_at"],
        url=row["url"],
        result=result,
        error=row["error"],
    )


async def list_tasks(limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, status, progress, message, created_at, url FROM tasks "
            "ORDER BY created_at DESC LIMIT ?",
            [limit],
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def delete_task(task_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM tasks WHERE id=?", [task_id])
        await db.commit()
