"""Tests for the Live Notepad (Batch B2).

Notes are a single free-form string column on tasks. The PUT endpoint replaces
the whole string; GETting the task returns it on `TaskResponse.notes`.

Async functions are used (asyncio_mode=auto) for robust event-loop handling
under Python 3.13 — the older `asyncio.get_event_loop().run_until_complete()`
pattern is brittle in full-suite runs.
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import LessonResult


def _override_user(uid: str = "user-1") -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_task(task_id: str, user_id: str, *, with_result: bool = True) -> None:
    await state.create_task(task_id, "https://example.com/x", user_id=user_id)
    if with_result:
        await state.complete_task(task_id, LessonResult(summary="hello"))


async def test_new_task_has_empty_notes(client):
    _override_user("user-1")
    try:
        await _seed_task("notepad-task-1", "user-1")
        resp = client.get("/api/tasks/notepad-task-1")
        assert resp.status_code == 200
        assert resp.json()["notes"] == ""
    finally:
        _clear_override()


async def test_put_notes_persists_and_round_trips(client):
    _override_user("user-1")
    try:
        await _seed_task("notepad-task-2", "user-1")

        put = client.put(
            "/api/tasks/notepad-task-2/notes",
            json={"notes": "זה הפנקס שלי\nשורה שנייה"},
        )
        assert put.status_code == 200
        assert put.json() == {"ok": True}

        got = client.get("/api/tasks/notepad-task-2")
        assert got.status_code == 200
        assert got.json()["notes"] == "זה הפנקס שלי\nשורה שנייה"
    finally:
        _clear_override()


async def test_put_notes_can_clear(client):
    _override_user("user-1")
    try:
        await _seed_task("notepad-task-3", "user-1")
        client.put("/api/tasks/notepad-task-3/notes", json={"notes": "draft"})
        client.put("/api/tasks/notepad-task-3/notes", json={"notes": ""})

        got = client.get("/api/tasks/notepad-task-3").json()
        assert got["notes"] == ""
    finally:
        _clear_override()


def test_put_notes_404_on_unknown_task(client):
    _override_user("user-1")
    try:
        resp = client.put("/api/tasks/does-not-exist/notes", json={"notes": "x"})
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_put_notes_cross_user_isolation(client):
    await state.create_task("notepad-task-4", "https://example.com/x", user_id="user-A")
    await state.complete_task("notepad-task-4", LessonResult(summary="x"))

    _override_user("user-B")
    try:
        resp = client.put("/api/tasks/notepad-task-4/notes", json={"notes": "evil"})
        assert resp.status_code == 404
    finally:
        _clear_override()

    _override_user("user-A")
    try:
        got = client.get("/api/tasks/notepad-task-4").json()
        assert got["notes"] == ""
    finally:
        _clear_override()


def test_lesson_response_default_notes_empty_string():
    """If the DB lacks the column (legacy row), notes defaults to ''."""
    from app.models import TaskResponse, TaskStatus

    t = TaskResponse(
        task_id="x",
        status=TaskStatus.COMPLETED,
        progress=100,
        message="ok",
        created_at="2026-05-13T10:00:00+00:00",
    )
    assert t.notes == ""
