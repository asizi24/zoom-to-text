"""
Tests for the share-link feature.

POST /api/tasks/{id}/share  — creates/returns a share token (auth required)
GET  /api/share/{token}     — returns the task result (public, no auth)
GET  /share/{token}         — serves index.html (public page route)

Auth is bypassed via dependency_overrides.
"""
import asyncio
import json

import pytest

from app import state as state_module
from app.api.deps import get_current_user
from app.models import LessonResult


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _seed_completed_task(task_id: str, user_id: str = "test-user") -> None:
    """Insert a completed task with a minimal LessonResult into the DB."""
    result = LessonResult(summary="Test summary", language="he")

    async def _setup():
        await state_module.create_task(task_id, url="http://test/rec", user_id=user_id)
        await state_module.complete_task(task_id, result)

    asyncio.run(_setup())


@pytest.fixture
def authed_client(client):
    from app.main import app as fastapi_app

    fastapi_app.dependency_overrides[get_current_user] = lambda: "test-user"
    yield client
    fastapi_app.dependency_overrides.pop(get_current_user, None)


# ── POST /api/tasks/{id}/share ─────────────────────────────────────────────────

def test_share_returns_url(authed_client):
    _seed_completed_task("share-task-1")
    r = authed_client.post("/api/tasks/share-task-1/share")
    assert r.status_code == 200
    body = r.json()
    assert "share_url" in body
    assert "/share/" in body["share_url"]


def test_share_is_idempotent(authed_client):
    """Calling share twice returns the same token/URL."""
    _seed_completed_task("share-task-idem")
    url1 = authed_client.post("/api/tasks/share-task-idem/share").json()["share_url"]
    url2 = authed_client.post("/api/tasks/share-task-idem/share").json()["share_url"]
    assert url1 == url2


def test_share_unknown_task_returns_404(authed_client):
    r = authed_client.post("/api/tasks/does-not-exist/share")
    assert r.status_code == 404


def test_share_incomplete_task_returns_400(authed_client):
    async def _setup():
        await state_module.create_task("share-pending", url="x", user_id="test-user")

    asyncio.run(_setup())
    r = authed_client.post("/api/tasks/share-pending/share")
    assert r.status_code == 400


def test_share_requires_auth(client):
    """Without auth override, endpoint must reject (get_current_user raises 401)."""
    _seed_completed_task("share-noauth", user_id="other-user")
    r = client.post("/api/tasks/share-noauth/share")
    assert r.status_code in (401, 403)


# ── GET /api/share/{token} ─────────────────────────────────────────────────────

def test_public_share_endpoint_returns_result(authed_client, client):
    _seed_completed_task("share-public-1")
    # Create share token via auth'd endpoint
    share_url = authed_client.post("/api/tasks/share-public-1/share").json()["share_url"]
    token = share_url.rstrip("/").rsplit("/", 1)[-1]

    # Fetch via public endpoint (no auth needed)
    r = client.get(f"/api/share/{token}")
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["summary"] == "Test summary"


def test_invalid_share_token_returns_404(client):
    r = client.get("/api/share/0" * 32)
    assert r.status_code == 404


def test_share_token_does_not_expose_audio(authed_client, client):
    """has_audio must always be False in shared responses."""
    _seed_completed_task("share-no-audio")
    share_url = authed_client.post("/api/tasks/share-no-audio/share").json()["share_url"]
    token = share_url.rstrip("/").rsplit("/", 1)[-1]

    body = client.get(f"/api/share/{token}").json()
    assert body["has_audio"] is False


# ── GET /share/{token} — page route ───────────────────────────────────────────

def test_share_page_route_serves_html(authed_client, client):
    """The /share/{token} page route must return 200 without requiring a session."""
    _seed_completed_task("share-page-1")
    share_url = authed_client.post("/api/tasks/share-page-1/share").json()["share_url"]
    # Extract path portion: /share/{token}
    path = "/" + share_url.split("//", 1)[-1].split("/", 1)[-1]

    r = client.get(path)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
