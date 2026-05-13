"""Tests for the B5 ICS calendar export.

Verifies:
  • the pure builder emits a valid one-event VCALENDAR
  • text fields are RFC 5545-escaped (commas, newlines)
  • the endpoint returns text/calendar with the correct filename
  • 404 for foreign tasks, 400 for pending tasks
  • collaborators with shared access can download the ICS too
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Chapter, LessonResult
from app.services.exporters.ics import build_ics


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_completed(task_id: str, user_id: str, *, summary: str = "תוכן ההרצאה."):
    await state.create_task(task_id, "https://x/ics", user_id=user_id)
    await state.complete_task(
        task_id,
        LessonResult(summary=summary, chapters=[Chapter(title="פתיחה", content="x")]),
    )
    return await state.get_task_for_user(task_id, user_id)


# ── Pure builder ─────────────────────────────────────────────────────────────


async def test_build_ics_has_required_lines(client):
    task = await _seed_completed("ics-task-1", "ics-user-1")
    out = build_ics(task)
    assert "BEGIN:VCALENDAR" in out
    assert "VERSION:2.0" in out
    assert "BEGIN:VEVENT" in out
    assert "END:VEVENT" in out
    assert "END:VCALENDAR" in out
    assert "UID:task-ics-task-1@zoom-to-text" in out
    # CRLF line endings per RFC 5545 §3.1
    assert "\r\n" in out


async def test_build_ics_escapes_special_chars(client):
    task = await _seed_completed(
        "ics-task-2",
        "ics-user-2",
        summary="שורה אחת, עם פסיק; ושורה\nשנייה",
    )
    out = build_ics(task)
    # Per RFC 5545 §3.3.11, comma → \,  semicolon → \;  newline → \n
    assert "\\," in out
    assert "\\;" in out
    assert "\\n" in out


# ── Endpoint ─────────────────────────────────────────────────────────────────


async def test_endpoint_returns_calendar_attachment(client):
    await _seed_completed("ics-task-3", "ics-user-3")
    _override_user("ics-user-3")
    try:
        resp = client.get("/api/tasks/ics-task-3/export/ics")
        assert resp.status_code == 200
        assert "text/calendar" in resp.headers["content-type"]
        assert "lesson-ics-task" in resp.headers["content-disposition"]
        assert "BEGIN:VCALENDAR" in resp.text
    finally:
        _clear_override()


async def test_endpoint_404_for_other_user(client):
    await _seed_completed("ics-task-4", "ics-user-A")
    _override_user("ics-user-B")
    try:
        resp = client.get("/api/tasks/ics-task-4/export/ics")
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_endpoint_400_when_pending(client):
    await state.create_task("ics-task-5", "https://x/ics", user_id="ics-user-5")
    _override_user("ics-user-5")
    try:
        resp = client.get("/api/tasks/ics-task-5/export/ics")
        assert resp.status_code == 400
    finally:
        _clear_override()


async def test_endpoint_allows_collaborator(client):
    # Owner = ics-user-O, collaborator = ics-user-C
    await _seed_completed("ics-task-6", "ics-user-O")
    # Bootstrap the collaborator's user row
    from app import state as _s
    db = await _s._get_db()
    await db.execute(
        "INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?,?,?)",
        ["ics-user-C", "collab@example.com", "2026-01-01T00:00:00+00:00"],
    )
    await db.execute(
        "INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?,?,?)",
        ["ics-user-O", "owner@example.com", "2026-01-01T00:00:00+00:00"],
    )
    await db.commit()
    await state.share_task("ics-task-6", "ics-user-C", "ics-user-O")

    _override_user("ics-user-C")
    try:
        resp = client.get("/api/tasks/ics-task-6/export/ics")
        assert resp.status_code == 200
        assert "BEGIN:VCALENDAR" in resp.text
    finally:
        _clear_override()
