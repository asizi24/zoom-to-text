"""
Tests for POST /api/tasks/{task_id}/retry.

Behavior:
  - Only failed tasks can be retried (404/400 otherwise).
  - Tasks whose URL begins with "upload:" cannot be retried — file is gone.
  - Successful retry creates a NEW task with the same URL, deletes the old row,
    removes the old audio file, and schedules processor.run_pipeline as a
    BackgroundTask (we patch processor.run_pipeline to avoid real network IO).
"""
from unittest.mock import AsyncMock, patch

import pytest


def _login(client, monkeypatch) -> str:
    import app.api.auth as auth_module
    captured: list[str] = []

    async def fake_send(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def test_retry_requires_auth(client):
    r = client.post("/api/tasks/nonexistent/retry", json={})
    assert r.status_code == 401


async def test_retry_404_for_unknown_task(client, monkeypatch):
    sid = _login(client, monkeypatch)
    r = client.post(
        "/api/tasks/missing-id/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 404


async def test_retry_400_for_non_failed_task(client, monkeypatch):
    """Tasks in 'pending' status cannot be retried."""
    from app import state
    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)
    await state.create_task("pend-r1", "https://zoom.us/p1", user_id=user_id)

    r = client.post(
        "/api/tasks/pend-r1/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 400


async def test_retry_rejects_upload_source(client, monkeypatch):
    """Uploaded-file tasks have no URL to re-fetch — must return 400."""
    from app import state
    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)
    await state.create_task("up1", "upload:lecture.mp4", user_id=user_id)
    await state.fail_task("up1", "transcription failed")

    r = client.post(
        "/api/tasks/up1/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 400


async def test_retry_creates_new_task_and_removes_old(client, monkeypatch):
    """
    A successful retry:
      - schedules processor.run_pipeline (we patch it to no-op)
      - returns a TaskResponse for a NEW task_id (not the old one)
      - the old task is deleted from the DB
    """
    from app import state
    import app.api.routes as routes_module

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("old-r1", "https://zoom.us/retry-me", user_id=user_id)
    await state.fail_task("old-r1", "boom")

    pipeline_calls = []

    async def fake_pipeline(**kwargs):
        pipeline_calls.append(kwargs)

    monkeypatch.setattr(routes_module.processor, "run_pipeline", fake_pipeline)

    r = client.post(
        "/api/tasks/old-r1/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 202, r.text
    body = r.json()
    new_id = body["task_id"]
    assert new_id != "old-r1"
    assert body["status"] == "pending"
    assert body["url"] == "https://zoom.us/retry-me"

    # old row gone
    assert await state.get_task_for_user("old-r1", user_id) is None
    # new row present, owned by same user
    assert await state.get_task_for_user(new_id, user_id) is not None
    # background pipeline was scheduled with new id
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0]["task_id"] == new_id
    assert pipeline_calls[0]["url"] == "https://zoom.us/retry-me"


async def test_retry_unlinks_old_audio_file(client, monkeypatch, tmp_path):
    """The audio file from the old failed task is removed from disk."""
    from app import state
    import app.api.routes as routes_module

    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    monkeypatch.setattr(routes_module, "_AUDIO_ROOT", audio_root.resolve())

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    audio = audio_root / "to_unlink.m4a"
    audio.write_bytes(b"fake")
    assert audio.exists()

    await state.create_task("au-r1", "https://zoom.us/unlink", user_id=user_id)
    await state.set_audio_path("au-r1", str(audio))
    await state.fail_task("au-r1", "boom")

    async def fake_pipeline(**kwargs):
        pass

    monkeypatch.setattr(routes_module.processor, "run_pipeline", fake_pipeline)

    r = client.post(
        "/api/tasks/au-r1/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 202
    assert not audio.exists(), "old audio file should be unlinked on retry"


async def test_retry_rejects_foreign_owner(client, monkeypatch):
    """A user cannot retry another user's failed task — 404 (anti-enumeration)."""
    from app import state
    sid = _login(client, monkeypatch)
    await state.create_task("foreign1", "https://zoom.us/foreign", user_id="other")
    await state.fail_task("foreign1", "boom")

    r = client.post(
        "/api/tasks/foreign1/retry",
        json={},
        cookies={"session_id": sid},
    )
    assert r.status_code == 404
