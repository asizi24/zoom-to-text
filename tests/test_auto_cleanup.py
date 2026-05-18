"""
Tests for the failed-task auto-cleanup mechanism.

state.cleanup_stale_failed_tasks(threshold_hours):
  - Removes failed rows whose failed_at is older than threshold.
  - Returns [{"id", "audio_path"}] so caller can unlink files.
  - Leaves recent-failed and non-failed tasks alone.
"""
from datetime import datetime, timedelta, timezone

import pytest


async def _set_failed_at_directly(task_id: str, iso_value: str) -> None:
    """
    Bypass fail_task() to set failed_at to an arbitrary point in time.
    Required for testing the 24h boundary.
    """
    from app import state
    db = await state._get_db()
    await db.execute(
        "UPDATE tasks SET status='failed', failed_at=? WHERE id=?",
        [iso_value, task_id],
    )
    await db.commit()


async def test_cleanup_removes_old_failed_tasks(client):
    """A failed task older than the threshold is deleted."""
    from app import state
    await state.create_task("old1", "https://zoom.us/old", user_id="u")
    old_iso = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await _set_failed_at_directly("old1", old_iso)

    removed = await state.cleanup_stale_failed_tasks(threshold_hours=24)

    ids = [r["id"] for r in removed]
    assert "old1" in ids
    assert await state.get_task_for_user("old1", "u") is None


async def test_cleanup_keeps_recent_failures(client):
    """A task that failed only an hour ago survives the 24h cleanup."""
    from app import state
    await state.create_task("recent1", "https://zoom.us/recent", user_id="u")
    recent_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    await _set_failed_at_directly("recent1", recent_iso)

    removed = await state.cleanup_stale_failed_tasks(threshold_hours=24)

    ids = [r["id"] for r in removed]
    assert "recent1" not in ids
    assert await state.get_task_for_user("recent1", "u") is not None


async def test_cleanup_ignores_non_failed_tasks(client):
    """Pending/completed tasks are never auto-cleaned even with old created_at."""
    from app import state
    from app.models import LessonResult

    await state.create_task("pend1", "https://zoom.us/pending", user_id="u")
    await state.create_task("done1", "https://zoom.us/done", user_id="u")
    await state.complete_task("done1", LessonResult(summary="ok"))

    # Force an old created_at AND failed_at, but status is NOT 'failed'.
    db = await state._get_db()
    old_iso = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    await db.execute(
        "UPDATE tasks SET created_at=?, failed_at=? WHERE id IN ('pend1','done1')",
        [old_iso, old_iso],
    )
    await db.commit()

    removed = await state.cleanup_stale_failed_tasks(threshold_hours=24)

    ids = [r["id"] for r in removed]
    assert "pend1" not in ids
    assert "done1" not in ids


async def test_cleanup_returns_audio_paths(client):
    """The return list carries audio_path so the caller can unlink files."""
    from app import state
    await state.create_task("aud1", "https://zoom.us/withaudio", user_id="u")
    await state.set_audio_path("aud1", "/data/downloads/aud1.m4a")
    old_iso = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await _set_failed_at_directly("aud1", old_iso)

    removed = await state.cleanup_stale_failed_tasks(threshold_hours=24)

    paths = [r["audio_path"] for r in removed if r["id"] == "aud1"]
    assert paths == ["/data/downloads/aud1.m4a"]


async def test_cleanup_skips_legacy_failed_without_failed_at(client):
    """
    Legacy rows that failed before failed_at existed have NULL failed_at.
    They MUST NOT be auto-deleted — only manually.
    """
    from app import state
    await state.create_task("legacy1", "https://zoom.us/legacy", user_id="u")
    db = await state._get_db()
    await db.execute(
        "UPDATE tasks SET status='failed', failed_at=NULL WHERE id='legacy1'"
    )
    await db.commit()

    removed = await state.cleanup_stale_failed_tasks(threshold_hours=24)

    ids = [r["id"] for r in removed]
    assert "legacy1" not in ids
    assert await state.get_task_for_user("legacy1", "u") is not None


async def test_fail_task_sets_failed_at(client):
    """fail_task() must populate failed_at automatically (without it cleanup is impossible)."""
    from app import state
    await state.create_task("ft1", "https://zoom.us/ft", user_id="u")
    await state.fail_task("ft1", "boom")
    task = await state.get_task_for_user("ft1", "u")
    assert task is not None
    assert task.failed_at is not None
    # Should be parseable as ISO 8601
    datetime.fromisoformat(task.failed_at)
