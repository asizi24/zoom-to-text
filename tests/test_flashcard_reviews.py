"""Tests for SM-2 review endpoints (Batch B3).

End-to-end via TestClient: seeds a completed task with flashcards, exercises
GET /api/flashcards/due and POST /api/tasks/{id}/flashcards/{idx}/review.

Tests are `async def` so pytest-asyncio (asyncio_mode=auto) gives us a stable
event loop. The sync TestClient is safe to call from inside an async test —
it dispatches to a worker thread internally.
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Flashcard, LessonResult


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_task_with_cards(task_id: str, user_id: str, n_cards: int = 3):
    await state.create_task(task_id, f"https://x/{task_id}", user_id=user_id)
    cards = [
        Flashcard(front=f"שאלה {i}", back=f"תשובה {i}", tags=["test"])
        for i in range(n_cards)
    ]
    await state.complete_task(
        task_id, LessonResult(summary="סיכום קצר", flashcards=cards)
    )


async def test_due_endpoint_lists_new_cards(client):
    await _seed_task_with_cards("rev-task-1", "user-1", n_cards=3)
    _override_user("user-1")
    try:
        resp = client.get("/api/flashcards/due")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["counts"]["new"] == 3
        assert body["counts"]["review"] == 0
        assert len(body["due"]) == 3
        statuses = {c["status"] for c in body["due"]}
        assert statuses == {"new"}
    finally:
        _clear_override()


async def test_review_endpoint_records_state_and_marks_seen(client):
    await _seed_task_with_cards("rev-task-2", "user-2", n_cards=2)
    _override_user("user-2")
    try:
        resp = client.post(
            "/api/tasks/rev-task-2/flashcards/0/review",
            json={"grade": "good"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["state"]["interval"] == 1
        assert body["state"]["repetitions"] == 1
        resp2 = client.get("/api/flashcards/due")
        body2 = resp2.json()
        assert body2["counts"]["new"] == 1
        assert body2["counts"]["review"] == 0
    finally:
        _clear_override()


async def test_review_endpoint_rejects_unknown_grade(client):
    await _seed_task_with_cards("rev-task-3", "user-3", n_cards=1)
    _override_user("user-3")
    try:
        resp = client.post(
            "/api/tasks/rev-task-3/flashcards/0/review",
            json={"grade": "meh"},
        )
        assert resp.status_code == 422
    finally:
        _clear_override()


async def test_review_endpoint_404_on_card_out_of_range(client):
    await _seed_task_with_cards("rev-task-4", "user-4", n_cards=2)
    _override_user("user-4")
    try:
        resp = client.post(
            "/api/tasks/rev-task-4/flashcards/99/review",
            json={"grade": "good"},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_review_endpoint_isolates_users(client):
    await _seed_task_with_cards("rev-task-5", "user-A", n_cards=1)
    _override_user("user-B")
    try:
        resp = client.post(
            "/api/tasks/rev-task-5/flashcards/0/review",
            json={"grade": "good"},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_again_resets_repetitions(client):
    await _seed_task_with_cards("rev-task-6", "user-6", n_cards=1)
    _override_user("user-6")
    try:
        client.post("/api/tasks/rev-task-6/flashcards/0/review", json={"grade": "good"})
        client.post("/api/tasks/rev-task-6/flashcards/0/review", json={"grade": "good"})
        resp = client.post(
            "/api/tasks/rev-task-6/flashcards/0/review",
            json={"grade": "again"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"]["repetitions"] == 0
        assert body["state"]["interval"] == 1
    finally:
        _clear_override()
