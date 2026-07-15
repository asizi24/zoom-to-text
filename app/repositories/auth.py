"""
Auth repository: users, single-use magic tokens, and 30-day sessions.
"""
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app import state

logger = logging.getLogger(__name__)


async def get_or_create_user(email: str) -> str:
    """Return user_id for the email, creating the user row if this is their first login.

    Upsert (ON CONFLICT DO NOTHING) instead of check-then-insert: two
    concurrent first logins for the same email interleave at the awaits, and
    the old pattern let both pass the SELECT and one INSERT blow up on the
    UNIQUE constraint. The conflict-tolerant INSERT + fresh SELECT is race-free.
    """
    db = await state._get_db()
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        "INSERT INTO users (id, email, created_at) VALUES (?,?,?) "
        "ON CONFLICT(email) DO NOTHING",
        [str(uuid.uuid4()), email.lower(), now],
    )
    await db.commit()
    async with db.execute("SELECT id FROM users WHERE email=?", [email.lower()]) as cursor:
        row = await cursor.fetchone()
    return row["id"]


async def create_magic_token(user_id: str) -> str:
    """Create a 15-minute single-use token. Returns the token string."""
    # token_urlsafe over uuid4: a credential should come from the CSPRNG API
    # meant for secrets, with more entropy (256 bits vs uuid4's 122).
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    db = await state._get_db()
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

    The burn is a single atomic UPDATE (used=0 → used=1) so two concurrent
    verify requests with the same token can never both create a session — the
    old check-then-update pattern had exactly that window. An expired token is
    also burned by the UPDATE, which is fine: it was unusable either way.
    """
    db = await state._get_db()
    async with db.execute(
        "UPDATE magic_tokens SET used=1 WHERE token=? AND used=0 "
        "RETURNING user_id, expires_at",
        [token],
    ) as cursor:
        row = await cursor.fetchone()
    await db.commit()
    if row is None:
        return None
    if datetime.now(timezone.utc) > datetime.fromisoformat(row["expires_at"]):
        return None
    return row["user_id"]


async def create_session(user_id: str) -> str:
    """Create a 30-day session. Returns the session_id (stored in cookie)."""
    session_id = secrets.token_urlsafe(32)  # credential — CSPRNG, not uuid4
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(days=30)).isoformat()
    db = await state._get_db()
    await db.execute(
        "INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        [session_id, user_id, now.isoformat(), expires_at],
    )
    await db.commit()
    return session_id


async def get_session_user(session_id: str) -> Optional[str]:
    """Return user_id if session exists and has not expired. None otherwise."""
    db = await state._get_db()
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
    db = await state._get_db()
    await db.execute("DELETE FROM sessions WHERE id=?", [session_id])
    await db.commit()


async def purge_expired_auth() -> tuple[int, int]:
    """Delete expired sessions and dead (expired or used) magic tokens.

    Nothing reads these rows once past expiry — get_session_user and
    consume_magic_token both re-check the timestamp — but without a purge they
    accumulate forever. Called by the retention watcher (main.py) twice a day.
    Returns (sessions_deleted, tokens_deleted).
    """
    now = datetime.now(timezone.utc).isoformat()
    db = await state._get_db()
    cur_sessions = await db.execute(
        "DELETE FROM sessions WHERE expires_at < ?", [now]
    )
    cur_tokens = await db.execute(
        "DELETE FROM magic_tokens WHERE expires_at < ? OR used=1", [now]
    )
    await db.commit()
    return cur_sessions.rowcount, cur_tokens.rowcount
