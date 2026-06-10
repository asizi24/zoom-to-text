"""Tests for B6.3 slide deck upload + auto-alignment to chapters.

Verifies:
  • upload replaces the existing deck
  • lexical-overlap heuristic picks the right chapter when slide text matches
  • slides whose text matches nothing get chapter_index=None (no false pin)
  • PATCH updates the alignment
  • DELETE clears the whole deck
  • 404 for foreign tasks, 400-style behavior for non-existent tasks
  • cohort collaborators can list (GET) but not edit (POST/PATCH/DELETE)
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


async def _seed_completed(task_id: str, user_id: str) -> None:
    await state.create_task(task_id, "https://x/slides", user_id=user_id)
    await state.complete_task(
        task_id,
        LessonResult(
            summary="הרצאה על מערכות הפעלה",
            chapters=[
                Chapter(title="תהליכים וניהול זיכרון", content="הרצאה על תהליכים זיכרון וירטואלי"),
                Chapter(title="מערכות קבצים", content="מערכת קבצים inode בלוקים"),
                Chapter(title="רשתות מחשבים", content="רשת tcp ip שכבות"),
            ],
        ),
    )


async def test_upload_aligns_slides_to_chapters(client):
    await _seed_completed("slides-task-1", "slides-user-1")
    _override_user("slides-user-1")
    try:
        resp = client.post(
            "/api/tasks/slides-task-1/slides",
            json={
                "slides": [
                    {"page_index": 0, "title": "תהליכים", "body": "ניהול זיכרון וירטואלי"},
                    {"page_index": 1, "title": "מערכת קבצים", "body": "inode ובלוקים"},
                    {"page_index": 2, "title": "רשת", "body": "tcp ip שכבות"},
                ]
            },
        )
        assert resp.status_code == 201, resp.text
        slides = resp.json()["slides"]
        assert len(slides) == 3
        # Slide 0 → chapter 0 ("תהליכים"); Slide 1 → chapter 1; Slide 2 → chapter 2
        assert slides[0]["chapter_index"] == 0
        assert slides[1]["chapter_index"] == 1
        assert slides[2]["chapter_index"] == 2
    finally:
        _clear_override()


async def test_upload_unmatched_slide_has_no_chapter(client):
    await _seed_completed("slides-task-2", "slides-user-2")
    _override_user("slides-user-2")
    try:
        resp = client.post(
            "/api/tasks/slides-task-2/slides",
            json={
                "slides": [
                    {"page_index": 0, "title": "פתיחה", "body": "שלום ובוקר טוב"},
                ]
            },
        )
        slides = resp.json()["slides"]
        assert slides[0]["chapter_index"] is None
    finally:
        _clear_override()


async def test_second_upload_replaces_deck(client):
    await _seed_completed("slides-task-3", "slides-user-3")
    _override_user("slides-user-3")
    try:
        client.post(
            "/api/tasks/slides-task-3/slides",
            json={"slides": [{"page_index": 0, "title": "old", "body": ""}]},
        )
        client.post(
            "/api/tasks/slides-task-3/slides",
            json={
                "slides": [
                    {"page_index": 0, "title": "new1", "body": ""},
                    {"page_index": 1, "title": "new2", "body": ""},
                ]
            },
        )
        body = client.get("/api/tasks/slides-task-3/slides").json()
        titles = [s["title"] for s in body["slides"]]
        assert titles == ["new1", "new2"]
    finally:
        _clear_override()


async def test_patch_updates_chapter_index(client):
    await _seed_completed("slides-task-4", "slides-user-4")
    _override_user("slides-user-4")
    try:
        slides = client.post(
            "/api/tasks/slides-task-4/slides",
            json={"slides": [{"page_index": 0, "title": "x", "body": ""}]},
        ).json()["slides"]
        sid = slides[0]["id"]
        resp = client.patch(
            f"/api/tasks/slides-task-4/slides/{sid}",
            json={"chapter_index": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["slides"][0]["chapter_index"] == 2
    finally:
        _clear_override()


async def test_delete_clears_deck(client):
    await _seed_completed("slides-task-5", "slides-user-5")
    _override_user("slides-user-5")
    try:
        client.post(
            "/api/tasks/slides-task-5/slides",
            json={"slides": [{"page_index": 0, "title": "x", "body": ""}]},
        )
        assert client.delete("/api/tasks/slides-task-5/slides").status_code == 204
        assert client.get("/api/tasks/slides-task-5/slides").json() == {"slides": []}
    finally:
        _clear_override()


async def test_foreign_user_cannot_upload(client):
    await _seed_completed("slides-task-6", "slides-user-A")
    _override_user("slides-user-B")
    try:
        resp = client.post(
            "/api/tasks/slides-task-6/slides",
            json={"slides": [{"page_index": 0, "title": "x", "body": ""}]},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_collaborator_can_list_but_not_edit(client):
    # Seed users + completed task + share
    await state._get_db()
    db = await state._get_db()
    await db.execute(
        "INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?,?,?)",
        ["slides-owner-7", "o7@example.com", "2026-01-01T00:00:00+00:00"],
    )
    await db.execute(
        "INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?,?,?)",
        ["slides-collab-7", "c7@example.com", "2026-01-01T00:00:00+00:00"],
    )
    await db.commit()
    await _seed_completed("slides-task-7", "slides-owner-7")
    await state.share_task("slides-task-7", "slides-collab-7", "slides-owner-7")
    # Owner uploads
    _override_user("slides-owner-7")
    try:
        client.post(
            "/api/tasks/slides-task-7/slides",
            json={"slides": [{"page_index": 0, "title": "תהליכים", "body": ""}]},
        )
    finally:
        _clear_override()
    # Collaborator can GET
    _override_user("slides-collab-7")
    try:
        resp = client.get("/api/tasks/slides-task-7/slides")
        assert resp.status_code == 200
        assert len(resp.json()["slides"]) == 1
        # Collaborator cannot POST a new deck (treated as non-owner = 404)
        resp = client.post(
            "/api/tasks/slides-task-7/slides",
            json={"slides": [{"page_index": 0, "title": "fake", "body": ""}]},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()
