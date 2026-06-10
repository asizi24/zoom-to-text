"""Tests for B6.2 lesson recipes (saved processing presets)."""
import pytest

from app import state
from app.api import deps
from app.main import app


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def test_list_recipes_empty(client):
    _override_user("rec-user-1")
    try:
        resp = client.get("/api/recipes")
        assert resp.status_code == 200
        assert resp.json() == {"recipes": []}
    finally:
        _clear_override()


async def test_create_recipe_returns_record(client):
    _override_user("rec-user-2")
    try:
        resp = client.post(
            "/api/recipes",
            json={
                "name": "מצגת באנגלית",
                "mode": "whisper_api",
                "language": "en",
                "tags": ["english", "lecture"],
                "notes": "תמלל ב-Whisper כי הקלטות מסוננות",
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["id"]
        assert body["name"] == "מצגת באנגלית"
        assert body["mode"] == "whisper_api"
        assert body["language"] == "en"
        assert body["tags"] == ["english", "lecture"]
        assert "notes" in body
    finally:
        _clear_override()


async def test_create_recipe_rejects_invalid_mode(client):
    _override_user("rec-user-3")
    try:
        resp = client.post(
            "/api/recipes",
            json={"name": "test", "mode": "openai_whisper_v9"},
        )
        # Pydantic pattern mismatch
        assert resp.status_code == 422
    finally:
        _clear_override()


async def test_create_recipe_rejects_too_long_name(client):
    _override_user("rec-user-4")
    try:
        resp = client.post(
            "/api/recipes",
            json={"name": "x" * 100, "mode": "gemini_direct"},
        )
        assert resp.status_code == 422
    finally:
        _clear_override()


async def test_patch_recipe_updates_fields(client):
    _override_user("rec-user-5")
    try:
        created = client.post(
            "/api/recipes",
            json={"name": "preset 1", "mode": "gemini_direct"},
        ).json()
        rid = created["id"]
        resp = client.patch(
            f"/api/recipes/{rid}",
            json={"name": "preset 1 renamed", "language": "he"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "preset 1 renamed"
        assert body["language"] == "he"
        # Untouched field preserved:
        assert body["mode"] == "gemini_direct"
    finally:
        _clear_override()


async def test_delete_recipe_then_404(client):
    _override_user("rec-user-6")
    try:
        rid = client.post(
            "/api/recipes",
            json={"name": "doomed", "mode": "gemini_direct"},
        ).json()["id"]
        assert client.delete(f"/api/recipes/{rid}").status_code == 204
        assert client.get(f"/api/recipes/{rid}").status_code == 404
    finally:
        _clear_override()


async def test_other_user_cannot_see_my_recipes(client):
    _override_user("rec-user-A")
    try:
        client.post("/api/recipes", json={"name": "mine", "mode": "gemini_direct"})
    finally:
        _clear_override()

    _override_user("rec-user-B")
    try:
        resp = client.get("/api/recipes")
        assert resp.json() == {"recipes": []}
    finally:
        _clear_override()


async def test_other_user_cannot_patch_or_delete(client):
    _override_user("rec-user-X")
    try:
        rid = client.post(
            "/api/recipes",
            json={"name": "x", "mode": "gemini_direct"},
        ).json()["id"]
    finally:
        _clear_override()

    _override_user("rec-user-Y")
    try:
        assert client.patch(f"/api/recipes/{rid}", json={"name": "stolen"}).status_code == 404
        assert client.delete(f"/api/recipes/{rid}").status_code == 404
    finally:
        _clear_override()


async def test_list_recipes_ordered_newest_first(client):
    _override_user("rec-user-order")
    try:
        client.post("/api/recipes", json={"name": "older", "mode": "gemini_direct"})
        client.post("/api/recipes", json={"name": "middle", "mode": "gemini_direct"})
        client.post("/api/recipes", json={"name": "newest", "mode": "gemini_direct"})
        body = client.get("/api/recipes").json()
        names = [r["name"] for r in body["recipes"]]
        # Newest first
        assert names[0] == "newest"
        assert names[-1] == "older"
    finally:
        _clear_override()
