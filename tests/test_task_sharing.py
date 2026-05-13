"""Tests for B5 cohort/task sharing.

Verifies:
  • owner can grant another (existing) user read access
  • granting an unknown email returns 400
  • granted user appears in `/api/tasks/{id}/shares`
  • granted user can fetch the task via `/api/tasks/{id}/readable`
  • non-collaborator gets 404 from the same endpoint
  • granted user sees the task in `/api/shared-tasks`
  • DELETE revokes access
  • non-owner cannot grant access (404 on the share-list endpoint)
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


async def _seed_user(user_id: str, email: str) -> None:
    db = await state._get_db()
    await db.execute(
        "INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?,?,?)",
        [user_id, email.lower(), "2026-01-01T00:00:00+00:00"],
    )
    await db.commit()


async def _seed_completed(task_id: str, owner_id: str) -> None:
    await state.create_task(task_id, "https://x/share", user_id=owner_id)
    await state.complete_task(
        task_id,
        LessonResult(
            summary="הרצאה לבדיקת שיתוף",
            chapters=[Chapter(title="פרק", content="x")],
        ),
    )


async def test_owner_can_grant_share_to_existing_user(client):
    await _seed_user("share-owner-1", "owner1@example.com")
    await _seed_user("share-target-1", "target1@example.com")
    await _seed_completed("share-task-1", "share-owner-1")

    _override_user("share-owner-1")
    try:
        resp = client.post(
            "/api/tasks/share-task-1/shares",
            json={"email": "target1@example.com"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["user_id"] == "share-target-1"
        assert body["granted_by_user_id"] == "share-owner-1"

        listed = client.get("/api/tasks/share-task-1/shares").json()
        assert len(listed["shares"]) == 1
        assert listed["shares"][0]["email"] == "target1@example.com"
    finally:
        _clear_override()


async def test_grant_unknown_email_returns_400(client):
    await _seed_user("share-owner-2", "owner2@example.com")
    await _seed_completed("share-task-2", "share-owner-2")

    _override_user("share-owner-2")
    try:
        resp = client.post(
            "/api/tasks/share-task-2/shares",
            json={"email": "nobody@example.com"},
        )
        assert resp.status_code == 400
    finally:
        _clear_override()


async def test_collaborator_can_read_shared_task(client):
    await _seed_user("share-owner-3", "owner3@example.com")
    await _seed_user("share-target-3", "target3@example.com")
    await _seed_completed("share-task-3", "share-owner-3")
    await state.share_task("share-task-3", "share-target-3", "share-owner-3")

    _override_user("share-target-3")
    try:
        resp = client.get("/api/tasks/share-task-3/readable")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "share-task-3"
    finally:
        _clear_override()


async def test_non_collaborator_gets_404_on_readable(client):
    await _seed_user("share-owner-4", "owner4@example.com")
    await _seed_user("share-other-4", "other4@example.com")
    await _seed_completed("share-task-4", "share-owner-4")
    # Note: NO share granted to share-other-4

    _override_user("share-other-4")
    try:
        resp = client.get("/api/tasks/share-task-4/readable")
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_shared_tasks_list_for_target_user(client):
    await _seed_user("share-owner-5", "owner5@example.com")
    await _seed_user("share-target-5", "target5@example.com")
    await _seed_completed("share-task-5", "share-owner-5")
    await state.share_task("share-task-5", "share-target-5", "share-owner-5")

    _override_user("share-target-5")
    try:
        resp = client.get("/api/shared-tasks")
        assert resp.status_code == 200
        ids = [t["id"] for t in resp.json()["tasks"]]
        assert "share-task-5" in ids
    finally:
        _clear_override()


async def test_owner_can_revoke_share(client):
    await _seed_user("share-owner-6", "owner6@example.com")
    await _seed_user("share-target-6", "target6@example.com")
    await _seed_completed("share-task-6", "share-owner-6")
    await state.share_task("share-task-6", "share-target-6", "share-owner-6")

    _override_user("share-owner-6")
    try:
        resp = client.delete("/api/tasks/share-task-6/shares/share-target-6")
        assert resp.status_code == 204
    finally:
        _clear_override()

    # Confirm target lost access
    _override_user("share-target-6")
    try:
        resp = client.get("/api/tasks/share-task-6/readable")
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_non_owner_cannot_list_collaborators(client):
    await _seed_user("share-owner-7", "owner7@example.com")
    await _seed_user("share-other-7", "other7@example.com")
    await _seed_completed("share-task-7", "share-owner-7")

    _override_user("share-other-7")
    try:
        resp = client.get("/api/tasks/share-task-7/shares")
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_cannot_share_with_self(client):
    await _seed_user("share-owner-8", "owner8@example.com")
    await _seed_completed("share-task-8", "share-owner-8")

    _override_user("share-owner-8")
    try:
        resp = client.post(
            "/api/tasks/share-task-8/shares",
            json={"email": "owner8@example.com"},
        )
        assert resp.status_code == 400
    finally:
        _clear_override()
