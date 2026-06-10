"""Tests for the Socratic AI Tutor endpoint (Batch B3).

We mock summarizer.tutor_about_lesson so no LLM is hit. Verifies:
  • the endpoint requires a real, completed task
  • the route forwards context + question and returns the model's answer
  • cross-user access is blocked
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Chapter, LessonResult


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def test_tutor_returns_answer(client, monkeypatch):
    await state.create_task("tutor-task-1", "https://x/1", user_id="user-1")
    await state.complete_task(
        "tutor-task-1",
        LessonResult(
            summary="הקצב הסיני התחיל ב-1949",
            chapters=[Chapter(title="פתיחה", content="רקע היסטורי")],
            transcript="זה היה לפני 75 שנה",
        ),
    )

    captured = {}

    async def _fake_tutor(context, question):
        captured["context"] = context
        captured["question"] = question
        return "מה אתה חושב — מתי בדיוק זה היה?"

    monkeypatch.setattr(
        "app.services.summarizer.tutor_about_lesson", _fake_tutor
    )

    _override_user("user-1")
    try:
        resp = client.post(
            "/api/tasks/tutor-task-1/tutor",
            json={"question": "מתי הקצב הסיני קרה?"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "מה אתה חושב" in body["answer"]
        assert "1949" in captured["context"]
        assert "רקע היסטורי" in captured["context"]
        assert captured["question"] == "מתי הקצב הסיני קרה?"
    finally:
        _clear_override()


async def test_tutor_blocks_other_users(client, monkeypatch):
    await state.create_task("tutor-task-2", "https://x/2", user_id="user-A")
    await state.complete_task("tutor-task-2", LessonResult(summary="x"))

    async def _fake_tutor(context, question):
        return "should not be called"

    monkeypatch.setattr(
        "app.services.summarizer.tutor_about_lesson", _fake_tutor
    )

    _override_user("user-B")
    try:
        resp = client.post(
            "/api/tasks/tutor-task-2/tutor",
            json={"question": "?"},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()


def test_tutor_rejects_empty_question(client):
    _override_user("user-1")
    try:
        resp = client.post(
            "/api/tasks/some-task/tutor",
            json={"question": ""},
        )
        assert resp.status_code == 422
    finally:
        _clear_override()
