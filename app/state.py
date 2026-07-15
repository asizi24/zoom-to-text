"""
SQLite connection lifecycle + data-access facade.

Why SQLite instead of an in-memory dict?
  - Tasks survive server restarts (the original bug: "Server restarted during processing")
  - Safe for concurrent async reads/writes
  - Zero infrastructure — no Redis, no Celery, no extra containers
  - The DB file lives in data/tasks.db which is mounted as a Docker volume

Connection strategy:
  A single cached aiosqlite connection is reused for all operations.
  This avoids the overhead of opening/closing on every request (the old pattern),
  while staying safe for async code via WAL mode.

Layout (since the repository split):
  This module owns the connection (DB_PATH/_get_db/close_db), the schema +
  migrations (init_db), and the readiness ping. The domain queries live in
  app/repositories/{tasks,jobs,auth,chat}.py and are re-exported here, so
  `state.<fn>` remains the stable seam for callers and for the test suite's
  monkeypatching (state.DB_PATH, state._db, state.consume_magic_token, …).
  New code may import from the specific repository directly.
"""
import asyncio
import logging

import aiosqlite

from app.config import settings

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
        # Wait up to 5s on a locked DB instead of failing immediately —
        # protects against transient contention from external readers
        # (sqlite3 CLI, backup scripts) hitting the same file.
        await _db.execute("PRAGMA busy_timeout=5000")
        # SQLite leaves REFERENCES clauses unenforced unless this is on.
        await _db.execute("PRAGMA foreign_keys=ON")
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


async def ping() -> bool:
    """Readiness check: True iff the DB connection answers a trivial query."""
    try:
        db = await _get_db()
        async with db.execute("SELECT 1") as cursor:
            await cursor.fetchone()
        return True
    except Exception as exc:
        logger.warning(f"DB ping failed: {exc}")
        return False


async def init_db():
    """Create all tables on startup. Migrate tasks table. Mark interrupted tasks as failed."""
    db = await _get_db()
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_USERS_TABLE_SQL)
    await db.execute(CREATE_MAGIC_TOKENS_TABLE_SQL)
    await db.execute(CREATE_SESSIONS_TABLE_SQL)
    await db.commit()

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
    if "audio_path" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN audio_path TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added audio_path column")
    if "payload_json" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN payload_json TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added payload_json column")
    if "error_detail" not in cols:
        await db.execute("ALTER TABLE tasks ADD COLUMN error_detail TEXT")
        await db.commit()
        logger.info("Migrated tasks table: added error_detail column")

    # Create indexes now that all columns are guaranteed to exist.
    # (user_id, created_at DESC) serves the history listing exactly
    # (WHERE user_id=? ORDER BY created_at DESC LIMIT n) with no sort step;
    # (status) serves reset_interrupted_tasks' startup scan.
    await db.execute(CREATE_TASKS_INDEX_SQL)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_user_created "
        "ON tasks (user_id, created_at DESC)"
    )
    await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)")
    await db.commit()
    logger.info(f"Database ready: {DB_PATH}")


# ── Facade re-exports ─────────────────────────────────────────────────────────────
# Imported at the bottom so the repositories (which call state._get_db() at
# runtime) can `from app import state` while this module is still initializing.

from app.repositories.tasks import (   # noqa: E402
    _MAX_PARTIAL_TRANSCRIPT_CHARS,
    _NOT_TERMINAL_GUARD,
    _TERMINAL_STATUSES,
    _row_to_task_response,
    append_partial_transcript,
    backfill_task_owners,
    cancel_task,
    clear_audio_path,
    complete_task,
    create_task,
    delete_task,
    fail_task,
    get_audio_path,
    get_partial_transcript,
    get_task,
    get_task_for_user,
    list_reclaimable_media,
    list_tasks,
    requeue_task,
    set_audio_path,
    update_task,
)
from app.repositories.jobs import (   # noqa: E402
    _IN_FLIGHT_STATUSES,
    clear_job_payload,
    finalize_job_payload,
    get_job_payload,
    list_payload_file_paths,
    reset_interrupted_tasks,
    set_job_payload,
)
from app.repositories.auth import (   # noqa: E402
    consume_magic_token,
    create_magic_token,
    create_session,
    delete_session,
    get_or_create_user,
    get_session_user,
    purge_expired_auth,
)
from app.repositories.chat import (   # noqa: E402
    _MAX_CHAT_MESSAGES,
    append_chat_message,
    clear_chat_history,
    get_chat_history,
)
