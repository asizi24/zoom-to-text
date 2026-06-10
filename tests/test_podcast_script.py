"""Tests for B6.4 AI podcast script generator.

We monkeypatch `get_provider()` to return a deterministic stub that doesn't
talk to a real LLM. Two flavors of stub:
  • happy path returns a valid JSON two-host script
  • failure path raises an exception → endpoint falls back to a local script
"""
import json

import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Chapter, LessonResult
from app.services import podcast_script as ps_module


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


class _StubProviderOk:
    name = "stub-ok"

    async def generate_text(self, prompt, *, temperature=0.3, **kwargs):
        return json.dumps(
            {
                "turns": [
                    {"speaker": "host_a", "text": "מה הנושא היום?"},
                    {"speaker": "host_b", "text": "מערכות הפעלה."},
                    {"speaker": "host_a", "text": "מה הכי חשוב לדעת?"},
                    {"speaker": "host_b", "text": "תהליכים וזיכרון."},
                    {"speaker": "host_a", "text": "ומה השלב הבא?"},
                    {"speaker": "host_b", "text": "ללמוד על מערכת קבצים."},
                ]
            },
            ensure_ascii=False,
        )


class _StubProviderBad:
    name = "stub-bad"

    async def generate_text(self, prompt, *, temperature=0.3, **kwargs):
        raise RuntimeError("provider down")


class _StubProviderGarbage:
    name = "stub-garbage"

    async def generate_text(self, prompt, *, temperature=0.3, **kwargs):
        return "this is not json at all"


async def _seed_completed(task_id: str, user_id: str):
    await state.create_task(task_id, "https://x/podcast", user_id=user_id)
    await state.complete_task(
        task_id,
        LessonResult(
            summary="הרצאה על מבני נתונים: מערכים, רשימות, עצים.",
            chapters=[
                Chapter(title="מערכים", content="מערך הוא רצף איברים"),
                Chapter(title="רשימות מקושרות", content="רשימה מקושרת מצביעים"),
                Chapter(title="עצים", content="עצים בינאריים חיפוש"),
            ],
        ),
    )


async def test_endpoint_returns_llm_script_on_happy_path(client, monkeypatch):
    await _seed_completed("pod-task-1", "pod-user-1")
    monkeypatch.setattr(ps_module, "get_provider", lambda: _StubProviderOk())

    _override_user("pod-user-1")
    try:
        resp = client.get("/api/tasks/pod-task-1/podcast-script")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["task_id"] == "pod-task-1"
        assert body["model"] == "stub-ok"
        assert len(body["turns"]) >= 4
        speakers = {t["speaker"] for t in body["turns"]}
        assert speakers == {"host_a", "host_b"}
    finally:
        _clear_override()


async def test_endpoint_falls_back_when_provider_throws(client, monkeypatch):
    await _seed_completed("pod-task-2", "pod-user-2")
    monkeypatch.setattr(ps_module, "get_provider", lambda: _StubProviderBad())

    _override_user("pod-user-2")
    try:
        resp = client.get("/api/tasks/pod-task-2/podcast-script")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "fallback"
        assert body["turns"], "fallback must yield at least one turn"
    finally:
        _clear_override()


async def test_endpoint_falls_back_on_garbage_response(client, monkeypatch):
    await _seed_completed("pod-task-3", "pod-user-3")
    monkeypatch.setattr(ps_module, "get_provider", lambda: _StubProviderGarbage())

    _override_user("pod-user-3")
    try:
        resp = client.get("/api/tasks/pod-task-3/podcast-script")
        assert resp.status_code == 200
        assert resp.json()["model"] == "fallback"
    finally:
        _clear_override()


async def test_endpoint_400_when_task_pending(client, monkeypatch):
    await state.create_task("pod-task-4", "https://x", user_id="pod-user-4")
    monkeypatch.setattr(ps_module, "get_provider", lambda: _StubProviderOk())
    _override_user("pod-user-4")
    try:
        resp = client.get("/api/tasks/pod-task-4/podcast-script")
        assert resp.status_code == 400
    finally:
        _clear_override()


async def test_endpoint_404_for_foreign_task(client, monkeypatch):
    await _seed_completed("pod-task-5", "pod-user-O")
    monkeypatch.setattr(ps_module, "get_provider", lambda: _StubProviderOk())
    _override_user("pod-user-X")
    try:
        resp = client.get("/api/tasks/pod-task-5/podcast-script")
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_normalize_turns_drops_invalid_entries():
    from app.services.podcast_script import _normalize_turns

    out = _normalize_turns(
        [
            {"speaker": "host_a", "text": "valid"},
            {"speaker": "narrator", "text": "wrong role"},  # dropped
            {"speaker": "A", "text": "alt label"},
            {"speaker": "host_b", "text": ""},  # dropped
            "garbage",  # dropped
            {"speaker": "B", "text": "alt label B"},
        ]
    )
    assert len(out) == 3
    assert out[0]["speaker"] == "host_a"
    assert out[1]["speaker"] == "host_a"
    assert out[2]["speaker"] == "host_b"
