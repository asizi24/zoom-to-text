"""
Tests for the task cancellation feature.

POST /api/tasks/{id}/cancel — marks in-flight task as cancelled (auth required)

Cancelled status must be detected by the poller and pipeline must not attempt
to fail_task when TaskCancelledError is raised.
"""
import asyncio
import pytest

from app import state as state_module
from app.api.deps import get_current_user
from app.models import LessonResult, TaskStatus


def _seed_pending_task(task_id: str, user_id: str = "test-user") -> None:
    asyncio.run(state_module.create_task(task_id, url="http://test/rec", user_id=user_id))


def _seed_completed_task(task_id: str, user_id: str = "test-user") -> None:
    result = LessonResult(summary="Test", language="he")
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


# ── POST /api/tasks/{id}/cancel ────────────────────────────────────────────────

def test_cancel_pending_task_returns_200(authed_client):
    _seed_pending_task("cancel-pending-1")
    r = authed_client.post("/api/tasks/cancel-pending-1/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


def test_cancel_sets_db_status_to_cancelled(authed_client):
    _seed_pending_task("cancel-db-check")
    authed_client.post("/api/tasks/cancel-db-check/cancel")
    task = authed_client.get("/api/tasks/cancel-db-check")
    # Task should be gone from active view or show cancelled
    # cancel sets status=cancelled; get_task_for_user still returns it
    assert task.status_code in (200, 404)
    if task.status_code == 200:
        assert task.json()["status"] == "cancelled"


def test_cancel_completed_task_returns_400(authed_client):
    _seed_completed_task("cancel-done")
    r = authed_client.post("/api/tasks/cancel-done/cancel")
    assert r.status_code == 400


def test_cancel_unknown_task_returns_404(authed_client):
    r = authed_client.post("/api/tasks/does-not-exist/cancel")
    assert r.status_code == 404


def test_cancel_requires_auth(client):
    _seed_pending_task("cancel-noauth")
    r = client.post("/api/tasks/cancel-noauth/cancel")
    assert r.status_code in (401, 403)


def test_cancel_is_idempotent(authed_client):
    """Cancelling twice: first returns 200, second returns 400 (already cancelled)."""
    _seed_pending_task("cancel-twice")
    r1 = authed_client.post("/api/tasks/cancel-twice/cancel")
    r2 = authed_client.post("/api/tasks/cancel-twice/cancel")
    assert r1.status_code == 200
    assert r2.status_code == 400


# ── is_task_cancelled helper ───────────────────────────────────────────────────

def test_is_task_cancelled_returns_true_after_cancel():
    _seed_pending_task("cancel-flag-check")

    async def _run():
        await state_module.cancel_task("cancel-flag-check")
        return await state_module.is_task_cancelled("cancel-flag-check")

    result = asyncio.run(_run())
    assert result is True


def test_is_task_cancelled_returns_false_for_pending():
    _seed_pending_task("cancel-flag-pending")

    result = asyncio.run(state_module.is_task_cancelled("cancel-flag-pending"))
    assert result is False
