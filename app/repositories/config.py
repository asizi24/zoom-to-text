"""
Runtime configuration repository (the app_config kv table).

Settings in app/config.py are import-time and immutable. The first-boot Setup
Wizard needs to persist a handful of choices — which Smart Summary backend to
use, the local model name, an optional Gemini API key, and a setup_complete
flag — that take effect *without* a container restart. Those key/value pairs
live here.

Values are always stored as TEXT; typed access (bool, etc.) is layered on top
in app/services/runtime_config.py. Like every repository, all access goes
through state._get_db() so the test suite's monkeypatching of the shared
connection keeps working.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from app import state
from app.state import get_write_lock

logger = logging.getLogger(__name__)


async def get_config(key: str) -> Optional[str]:
    """Return the stored string value for `key`, or None if unset."""
    db = await state._get_db()
    async with db.execute(
        "SELECT value FROM app_config WHERE key=?", [key]
    ) as cursor:
        row = await cursor.fetchone()
    return row["value"] if row is not None else None


async def set_config(key: str, value: str) -> None:
    """Upsert a config value. One atomic statement (no read-modify-write)."""
    async with get_write_lock():
        db = await state._get_db()
        now = datetime.now(timezone.utc).isoformat()
        await db.execute(
            "INSERT INTO app_config (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            [key, value, now],
        )
        await db.commit()


async def delete_config(key: str) -> None:
    """Remove a config key (e.g. clearing a stored API key)."""
    async with get_write_lock():
        db = await state._get_db()
        await db.execute("DELETE FROM app_config WHERE key=?", [key])
        await db.commit()


async def get_all_config() -> dict[str, str]:
    """Return the whole kv store as a plain dict."""
    db = await state._get_db()
    async with db.execute("SELECT key, value FROM app_config") as cursor:
        rows = await cursor.fetchall()
    return {row["key"]: row["value"] for row in rows}
