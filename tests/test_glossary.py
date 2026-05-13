"""Tests for the cross-lecture glossary (Batch B4).

The LLM provider is mocked at module level so we never make a real call.
We verify:
  • the build function calls generate_text with all user tasks' summaries
  • bad JSON from the provider degrades gracefully (empty terms)
  • cached glossary is served by GET /api/glossary
  • POST /api/glossary/refresh triggers a rebuild
  • cross-user task isolation (user-B never sees user-A's lectures)
"""
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Chapter, LessonResult
from app.services import glossary as glossary_mod


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed(task_id: str, user_id: str, summary: str):
    await state.create_task(task_id, f"https://x/{task_id}", user_id=user_id)
    await state.complete_task(
        task_id,
        LessonResult(
            summary=summary,
            chapters=[Chapter(title="פתיחה", content="x")],
        ),
    )


class _FakeProvider:
    """Stub LLM provider replacement."""

    name = "fake"
    supports_audio_upload = False
    supports_streaming = False

    def __init__(self, response: str):
        self._response = response
        self.calls: list[str] = []

    async def generate_text(self, prompt, *, max_tokens=65536, temperature=0.3, timeout=600.0):
        self.calls.append(prompt)
        return self._response


def _patch_provider(monkeypatch, response: str) -> _FakeProvider:
    fp = _FakeProvider(response)
    monkeypatch.setattr(glossary_mod, "get_provider", lambda: fp)
    return fp


async def test_get_glossary_empty_when_never_built(client):
    _override_user("g-user-1")
    try:
        resp = client.get("/api/glossary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["terms"] == []
        assert body["updated_at"] is None
    finally:
        _clear_override()


async def test_refresh_builds_glossary_from_user_tasks(client, monkeypatch):
    await _seed("glo-1", "g-user-2", "אומגה 3 חשוב לבריאות הלב")
    await _seed("glo-2", "g-user-2", "ויטמין D נחוץ לעצמות")

    fake = _patch_provider(
        monkeypatch,
        '{"terms": ['
        '{"term": "אומגה 3", "definition": "חומצת שומן חיונית", "sources": ["glo-1"]},'
        '{"term": "ויטמין D", "definition": "ויטמין השמש", "sources": ["glo-2"]}'
        ']}',
    )

    _override_user("g-user-2")
    try:
        resp = client.post("/api/glossary/refresh")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["task_count"] == 2
        terms = body["terms"]
        assert len(terms) == 2
        terms_by_name = {t["term"]: t for t in terms}
        assert "אומגה 3" in terms_by_name
        assert terms_by_name["אומגה 3"]["sources"] == ["glo-1"]
    finally:
        _clear_override()

    # Both summaries are in the prompt the provider received
    assert "אומגה 3" in fake.calls[0]
    assert "ויטמין D" in fake.calls[0]


async def test_refresh_drops_sources_referencing_other_users(client, monkeypatch):
    """LLM might hallucinate task ids — only the calling user's tasks are accepted."""
    await _seed("glo-own-1", "g-user-3", "סיכום של g-user-3")

    _patch_provider(
        monkeypatch,
        '{"terms": ['
        '{"term": "מונח א", "definition": "הגדרה",'
        ' "sources": ["glo-own-1", "glo-other-user"]}]}',
    )

    _override_user("g-user-3")
    try:
        resp = client.post("/api/glossary/refresh")
        assert resp.status_code == 200
        terms = resp.json()["terms"]
        assert terms[0]["sources"] == ["glo-own-1"]
    finally:
        _clear_override()


async def test_refresh_handles_malformed_json(client, monkeypatch):
    await _seed("glo-bad-1", "g-user-4", "טקסט")

    _patch_provider(monkeypatch, "not json at all")

    _override_user("g-user-4")
    try:
        resp = client.post("/api/glossary/refresh")
        # Bad JSON degrades to empty terms, NOT a 502
        assert resp.status_code == 200
        assert resp.json()["terms"] == []
    finally:
        _clear_override()


async def test_get_glossary_returns_cached_after_refresh(client, monkeypatch):
    await _seed("glo-c-1", "g-user-5", "סיכום למבחן הקאש")

    _patch_provider(
        monkeypatch,
        '{"terms": [{"term": "קאש", "definition": "זיכרון מטמון", "sources": []}]}',
    )

    _override_user("g-user-5")
    try:
        client.post("/api/glossary/refresh")
        resp = client.get("/api/glossary")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["terms"]) == 1
        assert body["terms"][0]["term"] == "קאש"
        assert body["updated_at"] is not None
    finally:
        _clear_override()


async def test_refresh_with_no_tasks_returns_empty_payload(client, monkeypatch):
    fake = _patch_provider(monkeypatch, '{"terms": []}')

    _override_user("g-user-empty")
    try:
        resp = client.post("/api/glossary/refresh")
        assert resp.status_code == 200
        body = resp.json()
        assert body["terms"] == []
        assert body["task_count"] == 0
    finally:
        _clear_override()

    # Empty library should NOT call the provider
    assert fake.calls == []
