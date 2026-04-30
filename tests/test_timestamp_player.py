"""
Tests for Feature 7 — timestamp player + /audio streaming endpoint.

Covers Range parsing, directory-traversal guard, 404 paths, full/partial
streaming, and the DELETE-also-wipes-file behavior.

Auth is bypassed via FastAPI dependency_overrides — simpler than dragging
the magic-link flow through every test.
"""
import asyncio
import re
from pathlib import Path

import pytest

from app import state as state_module
from app.api.deps import get_current_user
from app.api.routes import _parse_range


# ── _parse_range unit tests ─────────────────────────────────────────────────

def test_parse_range_none():
    assert _parse_range(None, 100) is None


def test_parse_range_missing_prefix():
    assert _parse_range("10-20", 100) is None


def test_parse_range_bad_bytes():
    assert _parse_range("bytes=abc", 100) is None
    assert _parse_range("bytes=", 100) is None


def test_parse_range_open_ended():
    assert _parse_range("bytes=10-", 100) == (10, 99)


def test_parse_range_closed():
    assert _parse_range("bytes=10-19", 100) == (10, 19)


def test_parse_range_suffix():
    assert _parse_range("bytes=-5", 100) == (95, 99)


def test_parse_range_out_of_bounds():
    assert _parse_range("bytes=90-200", 100) is None
    assert _parse_range("bytes=50-10", 100) is None


def test_parse_range_multi_takes_first():
    got = _parse_range("bytes=0-9,20-29", 100)
    assert got == (0, 9)


# ── Endpoint fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def authed_client(client, monkeypatch, tmp_path):
    """
    Extend `client`:
      - _AUDIO_ROOT is rebound to tmp_path/audio (it was snapshotted at import)
      - get_current_user is overridden to return a fixed user_id
    """
    from app.main import app as fastapi_app
    from app.api import routes as routes_module

    audio_root = (tmp_path / "audio").resolve()
    audio_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes_module, "_AUDIO_ROOT", audio_root)

    fastapi_app.dependency_overrides[get_current_user] = lambda: "test-user"

    yield client, tmp_path, audio_root

    fastapi_app.dependency_overrides.pop(get_current_user, None)


def _seed_task_with_audio(audio_root: Path, task_id: str, body: bytes) -> Path:
    p = audio_root / f"{task_id}.mp3"
    p.write_bytes(body)

    async def _setup():
        await state_module.create_task(task_id, url="http://test/rec", user_id=None)
        await state_module.set_audio_path(task_id, str(p))

    asyncio.run(_setup())
    return p


# ── Endpoint + security tests ───────────────────────────────────────────────

def test_missing_task_returns_404(authed_client):
    client, _, _ = authed_client
    r = client.get("/api/tasks/does-not-exist/audio")
    assert r.status_code == 404


def test_task_without_audio_returns_404(authed_client):
    client, _, _ = authed_client

    async def setup():
        await state_module.create_task("t-no-audio", url="x", user_id=None)

    asyncio.run(setup())
    r = client.get("/api/tasks/t-no-audio/audio")
    assert r.status_code == 404


def test_audio_path_outside_root_is_rejected(authed_client, tmp_path):
    """Defense in depth: if DB points outside _AUDIO_ROOT, refuse to serve."""
    client, _, _ = authed_client
    evil = tmp_path / "escape.mp3"
    evil.write_bytes(b"PWNED" * 10)

    async def setup():
        await state_module.create_task("t-escape", url="x", user_id=None)
        await state_module.set_audio_path("t-escape", str(evil))

    asyncio.run(setup())
    r = client.get("/api/tasks/t-escape/audio")
    assert r.status_code == 404
    assert b"PWNED" not in r.content


def test_full_stream_ok(authed_client):
    client, _, audio_root = authed_client
    body = b"\x00" * 512
    _seed_task_with_audio(audio_root, "t-full", body)
    r = client.get("/api/tasks/t-full/audio")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert int(r.headers["content-length"]) == len(body)
    assert r.content == body


def test_range_closed(authed_client):
    client, _, audio_root = authed_client
    body = bytes(range(256)) * 2
    _seed_task_with_audio(audio_root, "t-rng1", body)
    r = client.get("/api/tasks/t-rng1/audio", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 10-19/{len(body)}"
    assert int(r.headers["content-length"]) == 10
    assert r.content == body[10:20]


def test_range_suffix(authed_client):
    client, _, audio_root = authed_client
    body = bytes(range(100))
    _seed_task_with_audio(audio_root, "t-rng2", body)
    r = client.get("/api/tasks/t-rng2/audio", headers={"Range": "bytes=-5"})
    assert r.status_code == 206
    assert r.content == body[-5:]


def test_invalid_range_falls_back_to_200(authed_client):
    client, _, audio_root = authed_client
    body = b"abc"
    _seed_task_with_audio(audio_root, "t-rng3", body)
    r = client.get("/api/tasks/t-rng3/audio", headers={"Range": "bytes=what"})
    assert r.status_code == 200
    assert r.content == body


def test_delete_task_removes_audio_file(authed_client):
    client, _, audio_root = authed_client
    body = b"xyz" * 100
    p = _seed_task_with_audio(audio_root, "t-del", body)
    assert p.exists()
    r = client.delete("/api/tasks/t-del")
    assert r.status_code == 204
    assert not p.exists(), "audio file should be removed when the task is deleted"


# ── Transcriber timestamp-injection unit test ──────────────────────────────

def test_transcribe_sync_prepends_timestamps():
    """
    _transcribe_sync should emit [MM:SS] prefixes derived from each segment's
    `start` time.
    """
    from types import SimpleNamespace
    from app.services import transcriber

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            segs = [
                SimpleNamespace(start=0,   text="hello"),
                SimpleNamespace(start=75,  text="world"),
                SimpleNamespace(start=130, text="done"),
            ]
            info = SimpleNamespace(language="en")
            return iter(segs), info

    out, lang = transcriber._transcribe_sync(FakeModel(), "fake.mp3", "en", None)
    assert lang == "en"
    assert "[00:00] hello" in out
    assert "[01:15] world" in out
    assert "[02:10] done"  in out
    assert len(re.findall(r"\[\d\d:\d\d\]", out)) == 3
