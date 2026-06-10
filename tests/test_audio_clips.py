"""Tests for audio clip sharing (Batch B4).

ffmpeg is monkeypatched at the clip-extractor level so the suite doesn't
shell out for every test (and stays green on machines without ffmpeg).
We exercise:
  • the POST/list/DELETE auth'd endpoints with ownership checks
  • the public /clips/{id}.mp3 route returning bytes from extract_clip_bytes
  • validation of start/end range and max duration
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


async def _seed_task_with_audio(task_id: str, user_id: str, *, audio_path: str):
    await state.create_task(task_id, "https://x/clip", user_id=user_id)
    await state.complete_task(task_id, LessonResult(summary="x"))
    await state.set_audio_path(task_id, audio_path)


@pytest.fixture
def fake_audio_on_disk(tmp_path):
    """Write a placeholder audio file that just needs to exist on disk."""
    audio = tmp_path / "lecture.mp3"
    audio.write_bytes(b"fake-mp3-bytes-not-played")
    return str(audio)


async def test_create_clip_returns_share_url(client, monkeypatch, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-1", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        resp = client.post(
            "/api/tasks/clip-task-1/clips",
            json={"start_sec": 10.0, "end_sec": 20.0, "label": "intro"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["task_id"] == "clip-task-1"
        assert body["start_sec"] == 10.0
        assert body["end_sec"] == 20.0
        assert body["label"] == "intro"
        assert body["share_url"].endswith(f"/clips/{body['id']}.mp3")
    finally:
        _clear_override()


async def test_create_clip_rejects_invalid_range(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-2", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        resp = client.post(
            "/api/tasks/clip-task-2/clips",
            json={"start_sec": 30.0, "end_sec": 10.0},
        )
        assert resp.status_code == 400
    finally:
        _clear_override()


async def test_create_clip_enforces_max_duration(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-3", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        resp = client.post(
            "/api/tasks/clip-task-3/clips",
            json={"start_sec": 0.0, "end_sec": 600.0},  # 10 minutes — over the cap
        )
        assert resp.status_code == 400
        assert "300" in resp.json()["detail"]  # 300 sec = 5 min cap
    finally:
        _clear_override()


async def test_create_clip_blocks_other_users(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-4", "user-A", audio_path=fake_audio_on_disk)
    _override_user("user-B")
    try:
        resp = client.post(
            "/api/tasks/clip-task-4/clips",
            json={"start_sec": 0.0, "end_sec": 10.0},
        )
        assert resp.status_code == 404
    finally:
        _clear_override()


async def test_list_clips_returns_owned_only(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-5", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        client.post("/api/tasks/clip-task-5/clips", json={"start_sec": 0, "end_sec": 5})
        client.post("/api/tasks/clip-task-5/clips", json={"start_sec": 5, "end_sec": 10})
        resp = client.get("/api/tasks/clip-task-5/clips")
        assert resp.status_code == 200
        assert len(resp.json()["clips"]) == 2
    finally:
        _clear_override()


async def test_delete_clip(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-6", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        created = client.post(
            "/api/tasks/clip-task-6/clips", json={"start_sec": 0, "end_sec": 5}
        ).json()
        cid = created["id"]
        resp = client.delete(f"/api/clips/{cid}")
        assert resp.status_code == 204
        # Second delete returns 404
        assert client.delete(f"/api/clips/{cid}").status_code == 404
    finally:
        _clear_override()


async def test_public_clip_streams_audio_bytes(client, monkeypatch, fake_audio_on_disk):
    """The public /clips/{id}.mp3 route must return bytes from extract_clip_bytes."""
    await _seed_task_with_audio("clip-task-7", "user-1", audio_path=fake_audio_on_disk)

    fake_bytes = b"%MPEG sliced bytes here"

    async def _fake_extract(audio_path, start, end):
        assert audio_path == fake_audio_on_disk
        assert start == 0.0
        assert end == 5.0
        return fake_bytes

    # Patch on both modules where it's imported — main.py uses the function name
    # directly so we have to patch app.main.extract_clip_bytes.
    monkeypatch.setattr("app.main.extract_clip_bytes", _fake_extract)

    _override_user("user-1")
    try:
        created = client.post(
            "/api/tasks/clip-task-7/clips", json={"start_sec": 0.0, "end_sec": 5.0}
        ).json()
    finally:
        _clear_override()

    # Public endpoint — no auth needed
    resp = client.get(f"/clips/{created['id']}.mp3")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == fake_bytes


async def test_public_clip_404_on_unknown(client):
    resp = client.get("/clips/deadbeef00000000.mp3")
    assert resp.status_code == 404


async def test_public_clip_meta(client, fake_audio_on_disk):
    await _seed_task_with_audio("clip-task-8", "user-1", audio_path=fake_audio_on_disk)
    _override_user("user-1")
    try:
        created = client.post(
            "/api/tasks/clip-task-8/clips",
            json={"start_sec": 1.0, "end_sec": 6.0, "label": "meta-label"},
        ).json()
    finally:
        _clear_override()
    resp = client.get(f"/api/clips/{created['id']}/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "meta-label"
    assert body["start_sec"] == 1.0
    assert body["end_sec"] == 6.0
