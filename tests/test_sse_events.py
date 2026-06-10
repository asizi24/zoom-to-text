"""
Tests for the SSE task-progress endpoint: GET /api/tasks/{id}/events

Covers:
  1. Completed task → immediate terminal SSE event (done=true, status=completed)
  2. Failed task    → immediate terminal SSE event (done=true, status=failed)
  3. Missing task   → HTTP 404 before stream starts
  4. Response has Content-Type: text/event-stream

The TestClient consumes the full body because each terminal task's generator
yields exactly one event then closes — no infinite poll needed.
"""
import asyncio
import json

import pytest

from app.models import LessonResult, TaskStatus


# ── Auth helper ───────────────────────────────────────────────────────────────

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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sse_completed_task_sends_terminal_event(client, monkeypatch):
    """A completed task immediately emits one event with done=true and status=completed."""
    from app import state

    async def _seed():
        await state.create_task("sse-done", "https://zoom.us/rec/x", user_id="test-user")
        r = LessonResult(summary="שיעור טוב")
        await state.complete_task("sse-done", r)

    asyncio.run(_seed())

    from app.api import deps
    from app.main import app

    async def _fake_user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _fake_user
    try:
        resp = client.get("/api/tasks/sse-done/events")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        # SSE format: one or more "data: {...}\n\n" lines
        assert "data:" in body
        # Parse the last data line that contains done=true
        events = [
            json.loads(line[len("data:"):].strip())
            for line in body.splitlines()
            if line.startswith("data:")
        ]
        terminal = next((e for e in events if e.get("done")), None)
        assert terminal is not None, "Expected a terminal event with done=true"
        assert terminal["status"] == TaskStatus.COMPLETED
        assert terminal["progress"] == 100
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)


def test_sse_failed_task_sends_terminal_event(client, monkeypatch):
    """A failed task immediately emits one event with done=true and status=failed."""
    from app import state

    async def _seed():
        await state.create_task("sse-fail", "https://zoom.us/rec/y", user_id="test-user")
        await state.fail_task("sse-fail", "שגיאה", {"stage": "download", "code": "err"})

    asyncio.run(_seed())

    from app.api import deps
    from app.main import app

    async def _fake_user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _fake_user
    try:
        resp = client.get("/api/tasks/sse-fail/events")
        assert resp.status_code == 200
        body = resp.text
        events = [
            json.loads(line[len("data:"):].strip())
            for line in body.splitlines()
            if line.startswith("data:")
        ]
        terminal = next((e for e in events if e.get("done")), None)
        assert terminal is not None
        assert terminal["status"] == TaskStatus.FAILED
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)


def test_sse_missing_task_returns_404(client):
    """Requesting events for a non-existent task must return 404."""
    from app.api import deps
    from app.main import app

    async def _fake_user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _fake_user
    try:
        resp = client.get("/api/tasks/does-not-exist/events")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)


def test_sse_response_has_event_stream_content_type(client):
    """The endpoint must always advertise text/event-stream content type."""
    from app import state

    async def _seed():
        await state.create_task("sse-ct", "https://zoom.us/rec/z", user_id="test-user")
        await state.complete_task("sse-ct", LessonResult(summary="x"))

    asyncio.run(_seed())

    from app.api import deps
    from app.main import app

    async def _fake_user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _fake_user
    try:
        resp = client.get("/api/tasks/sse-ct/events")
        assert "text/event-stream" in resp.headers["content-type"]
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)
