"""
Persistent task state manager backed by SQLite (via aiosqlite).

Why SQLite instead of an in-memory dict?
  - Tasks survive server restarts (the original bug: "Server restarted during processing")
  - Safe for concurrent async reads/writes
  - Zero infrastructure — no Redis, no Celery, no extra containers
  - The DB file lives in data/tasks.db which is mounted as a Docker volume

Connection strategy:
  A single cached aiosqlite connection is reused for all operations.
  This avoids the overhead of opening/closing on every request (the old pattern),
  while staying safe for async code via WAL mode.
"""
import asyncio
import uuid
import aiosqlite
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models import TaskStatus, TaskResponse, LessonResult

logger = logging.getLogger(__name__)

DB_PATH = settings.data_dir / "tasks.db"

# ── Shared connection ────────────────────────────────────────────────────────
_db: aiosqlite.Connection | None = None
_db_lock = asyncio.Lock()

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

CREATE_TASKS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks (user_id)
"""

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    email      TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
)
"""

CREATE_MAGIC_TOKENS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS magic_tokens (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
)
"""

CREATE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
)
"""


# ── Lifecycle ─────────────────────────────────────────────────────────────────────

async def _get_db() -> aiosqlite.Connection:
    """Return the cached DB connection, creating it on first use."""
    global _db
    if _db is not None:
        return _db
    async with _db_lock:
        if _db is not None:
            return _db
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(DB_PATH)
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA synchronous=NORMAL")
        # Set row_factory once on the shared connection so all cursors return
        # aiosqlite.Row objects — avoids repeated mutation of the shared connection.
        _db.row_factory = aiosqlite.Row
        logger.info(f"SQLite connection opened: {DB_PATH}")
    return _db


async def close_db():
    """Close the shared connection (called during shutdown)."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("SQLite connection closed")


async def init_db():
    """Create all tables on startup. Migrate tasks table. Mark interrupted tasks as failed."""
    db = await _get_db()
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_USERS_TABLE_SQL)
    await db.execute(CREATE_MAGIC_TOKENS_TABLE_SQL)
    await db.execute(CREATE_SESSIONS_TABLE_SQL)
    await db.commit()

    # Add index on user_id for fast per-user task listings
    # (must run after user_id column exists, so after migration below)


    # Migrate: add missing columns to tasks table if needed
    async with db.execute("PRAGMA table_info(tasks)") as cursor:
        cols = [row[1] for row in await cursor.fetchall()]
    if "user_id" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN user_id TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added user_id column")
    if "partial_transcript" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN partial_transcript TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added partial_transcript column")
    if "chat_history" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN chat_history TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added chat_history column")

    # Create index now that user_id column is guaranteed to exist
    await db.execute(CREATE_TASKS_INDEX_SQL)
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
    db = await _get_db()
    result = await db.execute(
        f"UPDATE tasks SET status=?, progress=0, message=?, error=? WHERE status IN ({placeholders})",
        [TaskStatus.FAILED.value, "השרת הופעל מחדש — המשימה הופסקה", "השרת הופעל מחדש — נסה שוב"] + in_flight,
    )
    if result.rowcount:
        logger.warning(f"Marked {result.rowcount} interrupted task(s) as failed on startup")
    await db.commit()


# ── CRUD ──────────────────────────────────────────────────────────────────────────

async def create_task(task_id: str, url: str, user_id: Optional[str] = None) -> TaskResponse:
    now = datetime.now(timezone.utc).isoformat()
    db = await _get_db()
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


async def update_task(task_id: str, status: TaskStatus, progress: int, message: str):
    db = await _get_db()
    await db.execute(
        "UPDATE tasks SET status=?, progress=?, message=? WHERE id=?",
        [status.value, progress, message, task_id],
    )
    await db.commit()


async def complete_task(task_id: str, result: LessonResult):
    db = await _get_db()
    # Clear the live-preview column on completion — the full transcript is
    # stored in result_json, so partial_transcript is no longer needed and
    # would only waste space in the DB.
    await db.execute(
        "UPDATE tasks SET status=?, progress=100, message=?, result_json=?, partial_transcript=NULL WHERE id=?",
        [TaskStatus.COMPLETED.value, "Processing complete ✅", result.model_dump_json(), task_id],
    )
    await db.commit()


async def fail_task(task_id: str, error: str):
    # Truncate long error messages so they fit cleanly in the DB
    short_error = error[:500] if len(error) > 500 else error
    db = await _get_db()
    await db.execute(
        "UPDATE tasks SET status=?, message=?, error=? WHERE id=?",
        [TaskStatus.FAILED.value, f"Failed: {short_error}", short_error, task_id],
    )
    await db.commit()


async def get_task(task_id: str) -> Optional[TaskResponse]:
    db = await _get_db()
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


async def get_task_for_user(task_id: str, user_id: str) -> Optional[TaskResponse]:
    """
    Return a task only if it belongs to user_id.
    Returns None if not found OR if owned by a different user — both look like 404
    to prevent task-id enumeration across users.
    """
    db = await _get_db()
    async with db.execute(
        "SELECT * FROM tasks WHERE id=? AND (user_id=? OR user_id IS NULL)",
        [task_id, user_id],
    ) as cursor:
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


async def list_tasks(limit: int = 50, user_id: Optional[str] = None) -> list[dict]:
    db = await _get_db()
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
    db = await _get_db()
    await db.execute("DELETE FROM tasks WHERE id=?", [task_id])
    await db.commit()


# ── Auth CRUD ─────────────────────────────────────────────────────────────────────

async def get_or_create_user(email: str) -> str:
    """Return user_id for the email, creating the user row if this is their first login."""
    db = await _get_db()
    async with db.execute("SELECT id FROM users WHERE email=?", [email.lower()]) as cursor:
        row = await cursor.fetchone()
    if row:
        return row["id"]
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO users (id, email, created_at) VALUES (?,?,?)",
        [user_id, email.lower(), now],
    )
    await db.commit()
    return user_id


async def create_magic_token(user_id: str) -> str:
    """Create a 15-minute single-use token. Returns the token string."""
    from datetime import timedelta
    token = str(uuid.uuid4())
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    db = await _get_db()
    await db.execute(
        "INSERT INTO magic_tokens (token, user_id, expires_at) VALUES (?,?,?)",
        [token, user_id, expires_at],
    )
    await db.commit()
    return token


async def consume_magic_token(token: str) -> Optional[str]:
    """
    Validate and consume a magic token.
    Returns user_id if valid, None if expired/used/unknown.
    Marks the token used=1 on success.
    """
    db = await _get_db()
    async with db.execute(
        "SELECT user_id, expires_at, used FROM magic_tokens WHERE token=?", [token]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or row["used"]:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return None
    await db.execute("UPDATE magic_tokens SET used=1 WHERE token=?", [token])
    await db.commit()
    return row["user_id"]


async def create_session(user_id: str) -> str:
    """Create a 30-day session. Returns the session_id (stored in cookie)."""
    from datetime import timedelta
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=30)).isoformat()
    db = await _get_db()
    await db.execute(
        "INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        [session_id, user_id, now.isoformat(), expires_at],
    )
    await db.commit()
    return session_id


async def get_session_user(session_id: str) -> Optional[str]:
    """Return user_id if session exists and has not expired. None otherwise."""
    db = await _get_db()
    async with db.execute(
        "SELECT user_id, expires_at FROM sessions WHERE id=?", [session_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return None
    return row["user_id"]


async def delete_session(session_id: str):
    """Delete a session (logout)."""
    db = await _get_db()
    await db.execute("DELETE FROM sessions WHERE id=?", [session_id])
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
    db = await _get_db()
    # Guard: skip write if we are already at or above the size cap to prevent
    # the column from growing unboundedly on multi-hour recordings.
    await db.execute(
        """
        UPDATE tasks
           SET partial_transcript = COALESCE(partial_transcript, '') || ?
         WHERE id = ?
           AND LENGTH(COALESCE(partial_transcript, '')) < ?
        """,
        [text, task_id, _MAX_PARTIAL_TRANSCRIPT_CHARS],
    )
    await db.commit()


async def get_partial_transcript(task_id: str, from_offset: int = 0) -> tuple[str, int]:
    """
    Return new transcript text since from_offset, plus the current total length.
    Used by the GET /api/tasks/{id}/transcript endpoint so the frontend only
    fetches the delta on each poll rather than the entire growing string.

    Returns (delta_text, total_length).
    """
    db = await _get_db()
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


# ── Chat history ──────────────────────────────────────────────────────────────────

# Keep at most this many messages in the stored history (user + model turns combined).
# Older messages are trimmed from the front so the most recent context is preserved.
_MAX_CHAT_MESSAGES = 40


async def get_chat_history(task_id: str) -> list[dict]:
    """
    Return the stored chat history for a task as a list of
    {"role": "user"|"model", "content": "..."} dicts.
    Returns an empty list if no history yet.
    """
    import json as _json
    db = await _get_db()
    async with db.execute(
        "SELECT chat_history FROM tasks WHERE id=?", [task_id]
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or not row["chat_history"]:
        return []
    try:
        return _json.loads(row["chat_history"])
    except (ValueError, TypeError):
        return []


async def append_chat_message(task_id: str, role: str, content: str) -> None:
    """
    Append one message to the task's chat history and persist it.
    Trims the history to _MAX_CHAT_MESSAGES (oldest messages dropped first).
    """
    import json as _json
    history = await get_chat_history(task_id)
    history.append({"role": role, "content": content})
    # Trim from front to keep within the message cap
    if len(history) > _MAX_CHAT_MESSAGES:
        history = history[-_MAX_CHAT_MESSAGES:]
    db = await _get_db()
    await db.execute(
        "UPDATE tasks SET chat_history=? WHERE id=?",
        [_json.dumps(history, ensure_ascii=False), task_id],
    )
    await db.commit()


async def clear_chat_history(task_id: str) -> None:
    """Delete the chat history for a task (user-initiated reset)."""
    db = await _get_db()
    await db.execute(
        "UPDATE tasks SET chat_history=NULL WHERE id=?", [task_id]
    )
    await db.commit()
