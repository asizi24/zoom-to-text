"""Tests for Ask-Across-Lectures (Batch B2).

The summarizer.answer_across_lectures call is monkeypatched so we don't hit a
real LLM. We're verifying:
  • the endpoint requires auth (handled by dependency override fixture)
  • only the calling user's completed tasks are sent into the prompt
  • source [src:<prefix>] citations are mapped back to full task_ids + metadata
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import LessonResult, Chapter


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_completed(task_id: str, user_id: str, summary: str = "hello",
                          url: str = "https://x.com/a"):
    await state.create_task(task_id, url, user_id=user_id)
    await state.complete_task(
        task_id,
        LessonResult(summary=summary, chapters=[Chapter(title="פרק 1", content="x")]),
    )


async def test_ask_endpoint_returns_answer_and_sources(client, monkeypatch):
    """Happy path: one completed task, model cites it."""
    await _seed_completed("ask-task-1234aaaa", "user-1", summary="באומגה 3 יש יתרון בריאותי")

    async def _fake_answer(question, tasks):
        # `answer_across_lectures` is responsible for mapping [src:prefix]
        # citations back to full task IDs — the route consumes them as full IDs.
        return {
            "answer": "תשובה מתוך אומגה 3 [src:ask-task].",
            "sources": ["ask-task-1234aaaa"],
        }

    monkeypatch.setattr(
        "app.services.summarizer.answer_across_lectures", _fake_answer
    )

    _override_user("user-1")
    try:
        resp = client.post("/api/ask", json={"question": "מהו אומגה 3?"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "אומגה 3" in body["answer"]
        assert body["considered"] == 1
        # Sources are full task IDs after mapping
        assert body["sources"] == ["ask-task-1234aaaa"]
        # source_meta enriches each source with url/title for the UI
        assert len(body["source_meta"]) == 1
        assert body["source_meta"][0]["task_id"] == "ask-task-1234aaaa"
    finally:
        _clear_override()


async def test_ask_endpoint_isolates_users(client, monkeypatch):
    """User-B must not see User-A's tasks in the prompt."""
    await _seed_completed("user-a-task-aa", "user-A", summary="private to A")
    await _seed_completed("user-b-task-bb", "user-B", summary="public to B")

    captured: dict = {}

    async def _fake_answer(question, tasks):
        captured["task_ids"] = [t.task_id for t in tasks]
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(
        "app.services.summarizer.answer_across_lectures", _fake_answer
    )

    _override_user("user-B")
    try:
        resp = client.post("/api/ask", json={"question": "x"})
        assert resp.status_code == 200
    finally:
        _clear_override()

    assert "user-b-task-bb" in captured["task_ids"]
    assert "user-a-task-aa" not in captured["task_ids"]


def test_ask_endpoint_validates_question_length(client):
    _override_user("user-1")
    try:
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422
    finally:
        _clear_override()


def test_ask_endpoint_no_lectures_returns_friendly_answer(client, monkeypatch):
    """Empty library: the helper should produce a friendly fallback."""
    async def _fake_answer(question, tasks):
        # Mimic the real fallback when there are no usable lectures
        if not tasks:
            return {"answer": "אין הקלטות מסוכמות עדיין — סכם הקלטה ראשונה ואחר כך נסה שוב.", "sources": []}
        return {"answer": "x", "sources": []}

    monkeypatch.setattr(
        "app.services.summarizer.answer_across_lectures", _fake_answer
    )

    _override_user("user-empty")
    try:
        resp = client.post("/api/ask", json={"question": "מה הקור הזה?"})
        assert resp.status_code == 200
        assert "אין הקלטות מסוכמות" in resp.json()["answer"]
        assert resp.json()["considered"] == 0
    finally:
        _clear_override()


def test_extract_sources_parses_and_dedupes():
    from app.services.summarizer import _extract_sources_from_answer

    answer = "א [src:abc12345] ב [src:deadbeef] ג [src:abc12345] ד"
    assert _extract_sources_from_answer(answer) == ["abc12345", "deadbeef"]


def test_extract_sources_empty_when_no_citations():
    from app.services.summarizer import _extract_sources_from_answer

    assert _extract_sources_from_answer("plain answer, no citations") == []
