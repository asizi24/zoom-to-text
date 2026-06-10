"""
Tests for bulk-delete behavior.

DB-level: state.bulk_delete_tasks() returns the right partition of deleted/skipped.
Route-level: POST /api/tasks/bulk_delete authenticates, deletes only owned tasks,
             and removes audio files from disk.
"""
from pathlib import Path
from unittest.mock import patch

import pytest


def _login(client, monkeypatch) -> str:
    """Authenticate via magic link and return session_id."""
    import app.api.auth as auth_module
    captured: list[str] = []

    async def fake_send(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


# ── DB-level tests ────────────────────────────────────────────────────────────


async def test_bulk_delete_only_removes_owned(client):
    """Tasks owned by user_id are deleted; tasks owned by another user are skipped."""
    from app import state
    await state.create_task("bd1", "https://zoom.us/rec/owned-A", user_id="user-A")
    await state.create_task("bd2", "https://zoom.us/rec/owned-A2", user_id="user-A")
    await state.create_task("bd3", "https://zoom.us/rec/owned-B", user_id="user-B")

    out = await state.bulk_delete_tasks(["bd1", "bd2", "bd3"], user_id="user-A")

    assert set(out["deleted"]) == {"bd1", "bd2"}
    assert out["skipped"] == ["bd3"]
    # bd3 must still exist
    assert await state.get_task_for_user("bd3", "user-B") is not None


async def test_bulk_delete_returns_audio_paths(client):
    """audio_paths is populated with paths for tasks that had audio."""
    from app import state
    await state.create_task("a1", "https://zoom.us/rec/audio", user_id="u")
    await state.create_task("a2", "https://zoom.us/rec/no-audio", user_id="u")
    await state.set_audio_path("a1", "/data/downloads/a1.m4a")

    out = await state.bulk_delete_tasks(["a1", "a2"], user_id="u")

    assert sorted(out["deleted"]) == ["a1", "a2"]
    assert out["audio_paths"] == ["/data/downloads/a1.m4a"]


async def test_bulk_delete_empty_list(client):
    """Calling with no IDs returns empty buckets — no DB hit needed."""
    from app import state
    out = await state.bulk_delete_tasks([], user_id="u")
    assert out == {"deleted": [], "skipped": [], "audio_paths": []}


# ── Route-level tests ─────────────────────────────────────────────────────────


def test_bulk_delete_endpoint_requires_auth(client):
    """No session cookie → 401."""
    r = client.post("/api/tasks/bulk_delete", json={"task_ids": ["x"]})
    assert r.status_code == 401


async def test_bulk_delete_endpoint_removes_owned_only(client, monkeypatch):
    """End-to-end: owned tasks deleted, foreign task untouched, response shape correct."""
    from app import state
    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    await state.create_task("rt1", "https://zoom.us/A", user_id=user_id)
    await state.create_task("rt2", "https://zoom.us/B", user_id=user_id)
    await state.create_task("rt3", "https://zoom.us/C", user_id="other-user")

    r = client.post(
        "/api/tasks/bulk_delete",
        json={"task_ids": ["rt1", "rt2", "rt3"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data["deleted"]) == {"rt1", "rt2"}
    assert data["skipped"] == ["rt3"]

    # rt3 still present in DB
    assert await state.get_task_for_user("rt3", "other-user") is not None


def test_bulk_delete_validates_payload(client, monkeypatch):
    """Empty task_ids list → 422."""
    sid = _login(client, monkeypatch)
    r = client.post(
        "/api/tasks/bulk_delete",
        json={"task_ids": []},
        cookies={"session_id": sid},
    )
    assert r.status_code == 422


async def test_bulk_delete_endpoint_unlinks_audio_files(client, monkeypatch, tmp_path):
    """Audio files on disk are removed by the endpoint via _remove_audio_safely."""
    from app import state
    import app.api.routes as routes_module

    # Rebind _AUDIO_ROOT so the safety check accepts our tmp_path-based audio file.
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    monkeypatch.setattr(routes_module, "_AUDIO_ROOT", audio_root.resolve())

    sid = _login(client, monkeypatch)
    user_id = await state.get_session_user(sid)

    audio = audio_root / "rec.m4a"
    audio.write_bytes(b"fake audio data")
    assert audio.exists()

    await state.create_task("au1", "https://zoom.us/withaudio", user_id=user_id)
    await state.set_audio_path("au1", str(audio))

    r = client.post(
        "/api/tasks/bulk_delete",
        json={"task_ids": ["au1"]},
        cookies={"session_id": sid},
    )
    assert r.status_code == 200
    assert "au1" in r.json()["deleted"]
    assert not audio.exists(), "audio file should have been unlinked"
