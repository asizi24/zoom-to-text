"""
Regression tests for the architectural-review hardening:

  - Terminal task statuses are STICKY: a late progress update, completion, or
    failure can no longer overwrite/resurrect a CANCELLED (or other terminal)
    row. Deliberate exits (requeue for retry) still work.
  - Magic tokens are burned atomically — two concurrent verifies can never
    both create a session.
  - The retention sweep purges expired sessions / dead magic tokens and drops
    job payloads that can no longer be replayed.
  - TaskCreate rejects non-web URL schemes before they ever reach yt-dlp.
"""
import asyncio

import pytest
from pydantic import ValidationError

from app import state
from app.config import settings
from app.models import LessonResult, TaskCreate, TaskStatus


# ── Isolated-DB scenario helper (same pattern as test_task_management) ────────

def _run_with_db(tmp_path, monkeypatch, coro_factory):
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "guards.db")
        monkeypatch.setattr(state, "_db", None, raising=False)
        await state.init_db()
        try:
            await coro_factory()
        finally:
            await state.close_db()

    try:
        asyncio.run(wrapper())
    finally:
        # asyncio.run() leaves the thread with set_event_loop(None); restore a
        # usable current loop for sibling tests that still use the deprecated
        # asyncio.get_event_loop() (see test_task_management for details).
        asyncio.set_event_loop(asyncio.new_event_loop())


_RESULT = LessonResult(summary="s", language="he")


# ── Sticky terminal states ─────────────────────────────────────────────────────

def test_progress_update_cannot_resurrect_cancelled(tmp_path, monkeypatch):
    """The transcription thread's progress callback races the cancel endpoint;
    if it landed after the cancel it used to flip the row back to TRANSCRIBING
    forever (no terminal state, restart would even re-run the task)."""
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        assert await state.cancel_task("t1") is True
        await state.update_task("t1", TaskStatus.TRANSCRIBING, 60, "late callback")
        t = await state.get_task("t1")
        assert t.status == TaskStatus.CANCELLED
        assert t.progress != 60

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_complete_after_cancel_keeps_cancelled(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.cancel_task("t1")
        assert await state.complete_task("t1", _RESULT) is False
        t = await state.get_task("t1")
        assert t.status == TaskStatus.CANCELLED
        assert t.result is None          # losing result is discarded, not stored

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_fail_after_cancel_keeps_cancelled(tmp_path, monkeypatch):
    """E.g. the user cancels mid-download and yt-dlp then errors out — the
    task must stay CANCELLED, not flip to FAILED with a confusing message."""
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.cancel_task("t1")
        assert await state.fail_task("t1", "boom", detail="d") is False
        t = await state.get_task("t1")
        assert t.status == TaskStatus.CANCELLED
        assert t.error is None

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_cancel_after_complete_returns_false(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        assert await state.complete_task("t1", _RESULT) is True
        assert await state.cancel_task("t1") is False
        assert (await state.get_task("t1")).status == TaskStatus.COMPLETED

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_requeue_still_exits_terminal_state(tmp_path, monkeypatch):
    """The stickiness guard applies to pipeline writes only — retry's
    deliberate CANCELLED→PENDING transition must keep working."""
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.cancel_task("t1")
        await state.requeue_task("t1")
        assert (await state.get_task("t1")).status == TaskStatus.PENDING

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Atomic magic-token consume ─────────────────────────────────────────────────

def test_magic_token_is_single_use(tmp_path, monkeypatch):
    async def scenario():
        user_id = await state.get_or_create_user("a@example.com")
        token = await state.create_magic_token(user_id)
        assert await state.consume_magic_token(token) == user_id
        assert await state.consume_magic_token(token) is None    # already burned
        assert await state.consume_magic_token("no-such-token") is None

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_expired_magic_token_rejected(tmp_path, monkeypatch):
    async def scenario():
        user_id = await state.get_or_create_user("a@example.com")
        token = await state.create_magic_token(user_id)
        db = await state._get_db()
        await db.execute(
            "UPDATE magic_tokens SET expires_at=? WHERE token=?",
            ["2000-01-01T00:00:00+00:00", token],
        )
        await db.commit()
        assert await state.consume_magic_token(token) is None

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Auth purge ─────────────────────────────────────────────────────────────────

def test_purge_expired_auth(tmp_path, monkeypatch):
    async def scenario():
        user_id = await state.get_or_create_user("a@example.com")
        db = await state._get_db()

        live_session = await state.create_session(user_id)
        dead_session = await state.create_session(user_id)
        await db.execute(
            "UPDATE sessions SET expires_at=? WHERE id=?",
            ["2000-01-01T00:00:00+00:00", dead_session],
        )

        live_token = await state.create_magic_token(user_id)
        used_token = await state.create_magic_token(user_id)
        await db.commit()
        await state.consume_magic_token(used_token)   # marks used=1

        sessions, tokens = await state.purge_expired_auth()
        assert sessions == 1 and tokens == 1

        assert await state.get_session_user(live_session) == user_id
        assert await state.get_session_user(dead_session) is None
        # The unexpired, unused token still works after the purge
        assert await state.consume_magic_token(live_token) == user_id

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Retention sweep drops unreplayable payloads ────────────────────────────────

def test_retention_sweep_clears_dead_payload(tmp_path, monkeypatch):
    import app.main as main_module

    src = tmp_path / "upload.mp3"
    src.write_bytes(b"x" * 128)

    async def scenario():
        monkeypatch.setattr(settings, "media_retention_days", 7)
        monkeypatch.setattr(main_module, "_DATA_ROOT", tmp_path.resolve())

        await state.create_task("old_f", "upload:x", user_id="u1")
        await state.set_job_payload("old_f", {"file_path": str(src), "mode": "gemini_direct"})
        await state.fail_task("old_f", "boom")
        db = await state._get_db()
        await db.execute(
            "UPDATE tasks SET created_at=? WHERE id=?",
            ["2000-01-01T00:00:00+00:00", "old_f"],
        )
        await db.commit()

        await main_module._retention_sweep()

        assert not src.exists()                                # media reclaimed
        assert await state.get_job_payload("old_f") is None    # payload dropped
        # …so the row no longer surfaces in future sweeps
        assert await state.list_reclaimable_media("2020-01-01T00:00:00+00:00") == []

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── URL scheme validation ──────────────────────────────────────────────────────

def test_task_create_rejects_non_web_schemes():
    for bad in ("file:///etc/passwd", "javascript:alert(1)", "ftp://x/y", "zoom.us/rec/123"):
        with pytest.raises(ValidationError):
            TaskCreate(url=bad)


def test_task_create_accepts_and_trims_web_urls():
    assert TaskCreate(url="  https://zoom.us/rec/share/abc  ").url == "https://zoom.us/rec/share/abc"
    assert TaskCreate(url="http://zoom.us/rec/1").url == "http://zoom.us/rec/1"
