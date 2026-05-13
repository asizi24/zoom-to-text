"""Tests for B5 outgoing webhooks (Slack/Discord notifications).

We exercise:
  • config CRUD (list/create/patch/delete) with ownership checks
  • https-only enforcement on POST /api/webhooks
  • completion notifier builds the right Slack vs Discord payload
  • notifier is fire-and-forget — a 500 from the upstream doesn't raise
  • disabled webhooks are skipped
"""
import json

import httpx
import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Chapter, LessonResult, TaskResponse, TaskStatus
from app.services import webhooks as webhooks_mod


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_completed_task(task_id: str, user_id: str) -> TaskResponse:
    await state.create_task(task_id, "https://x/hook", user_id=user_id)
    result = LessonResult(
        summary="זהו סיכום בדיקה לוובהוק",
        chapters=[Chapter(title="פרק 1", content="x")],
    )
    await state.complete_task(task_id, result)
    return await state.get_task_for_user(task_id, user_id)


# ── CRUD endpoints ──────────────────────────────────────────────────────────


async def test_list_webhooks_empty(client):
    _override_user("hook-user-1")
    try:
        resp = client.get("/api/webhooks")
        assert resp.status_code == 200
        assert resp.json() == {"webhooks": []}
    finally:
        _clear_override()


async def test_create_webhook_returns_record(client):
    _override_user("hook-user-2")
    try:
        resp = client.post(
            "/api/webhooks",
            json={"kind": "slack", "url": "https://hooks.slack.com/services/AAA/BBB/CCC"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["kind"] == "slack"
        assert body["enabled"] is True
        assert body["id"]
    finally:
        _clear_override()


async def test_create_webhook_rejects_http_url(client):
    _override_user("hook-user-3")
    try:
        resp = client.post(
            "/api/webhooks",
            json={"kind": "slack", "url": "http://hooks.slack.com/abc"},
        )
        assert resp.status_code == 400
    finally:
        _clear_override()


async def test_create_webhook_rejects_unknown_kind(client):
    _override_user("hook-user-4")
    try:
        resp = client.post(
            "/api/webhooks",
            json={"kind": "teams", "url": "https://example.com/hook"},
        )
        assert resp.status_code == 422  # pydantic pattern mismatch
    finally:
        _clear_override()


async def test_patch_webhook_toggles_enabled(client):
    _override_user("hook-user-5")
    try:
        created = client.post(
            "/api/webhooks",
            json={"kind": "discord", "url": "https://discord.com/api/webhooks/x/y"},
        ).json()
        wid = created["id"]
        resp = client.patch(f"/api/webhooks/{wid}", json={"enabled": False})
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
    finally:
        _clear_override()


async def test_delete_webhook_then_404(client):
    _override_user("hook-user-6")
    try:
        created = client.post(
            "/api/webhooks",
            json={"kind": "slack", "url": "https://hooks.slack.com/services/Z/Z/Z"},
        ).json()
        wid = created["id"]
        assert client.delete(f"/api/webhooks/{wid}").status_code == 204
        assert client.delete(f"/api/webhooks/{wid}").status_code == 404
    finally:
        _clear_override()


async def test_other_user_cannot_see_my_webhooks(client):
    # User A creates a webhook
    _override_user("hook-user-A")
    try:
        client.post(
            "/api/webhooks",
            json={"kind": "slack", "url": "https://hooks.slack.com/services/A/A/A"},
        )
    finally:
        _clear_override()

    # User B sees an empty list
    _override_user("hook-user-B")
    try:
        resp = client.get("/api/webhooks")
        assert resp.json() == {"webhooks": []}
    finally:
        _clear_override()


# ── Notifier (Slack/Discord POST payload) ───────────────────────────────────


async def test_notifier_sends_slack_payload_with_summary_preview(client, monkeypatch):
    task = await _seed_completed_task("hook-task-1", "hook-user-notify-1")
    await state.create_webhook(
        "hook-user-notify-1", "slack", "https://hooks.slack.com/services/X/Y/Z"
    )

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            {"url": str(request.url), "body": json.loads(request.content.decode())}
        )
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(webhooks_mod.httpx, "AsyncClient", fake_client)

    sent = await webhooks_mod.notify_task_completed("hook-user-notify-1", task)
    assert sent == 1
    assert len(captured) == 1
    body = captured[0]["body"]
    # Slack payload uses "text" + "attachments"
    assert "text" in body
    assert "attachments" in body
    assert "סיכום בדיקה" in body["attachments"][0]["text"]


async def test_notifier_sends_discord_payload(client, monkeypatch):
    task = await _seed_completed_task("hook-task-2", "hook-user-notify-2")
    await state.create_webhook(
        "hook-user-notify-2", "discord", "https://discord.com/api/webhooks/X/Y"
    )

    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(204)

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(webhooks_mod.httpx, "AsyncClient", fake_client)

    sent = await webhooks_mod.notify_task_completed("hook-user-notify-2", task)
    assert sent == 1
    body = captured[0]
    # Discord payload uses "content" + "embeds"
    assert "content" in body
    assert "embeds" in body


async def test_notifier_swallows_upstream_500(client, monkeypatch):
    task = await _seed_completed_task("hook-task-3", "hook-user-notify-3")
    await state.create_webhook(
        "hook-user-notify-3", "slack", "https://hooks.slack.com/services/A/B/C"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(webhooks_mod.httpx, "AsyncClient", fake_client)

    # Must not raise — failed sends are logged, not propagated
    sent = await webhooks_mod.notify_task_completed("hook-user-notify-3", task)
    assert sent == 0


async def test_notifier_skips_disabled_webhooks(client, monkeypatch):
    task = await _seed_completed_task("hook-task-4", "hook-user-notify-4")
    created = await state.create_webhook(
        "hook-user-notify-4", "slack", "https://hooks.slack.com/services/D/E/F"
    )
    await state.set_webhook_enabled(created["id"], "hook-user-notify-4", False)

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(webhooks_mod.httpx, "AsyncClient", fake_client)

    sent = await webhooks_mod.notify_task_completed("hook-user-notify-4", task)
    assert sent == 0
    assert calls == []


async def test_notifier_with_no_webhooks_is_a_noop(client):
    task = await _seed_completed_task("hook-task-5", "hook-user-notify-5")
    sent = await webhooks_mod.notify_task_completed("hook-user-notify-5", task)
    assert sent == 0
