"""
Tests for the task-management suite: cancel, retry, and storage retention.

Split into:
  - Unit tests (asyncio.run + isolated temp DB) for the state-layer helpers,
    the cancellation registry, the retention sweep, and the transcriber's
    cooperative-cancel checkpoint — all deterministic, no worker timing.
  - Endpoint tests (TestClient) for the HTTP contract of /cancel and /retry,
    including a full fail→retry→succeed round trip through the real worker.
"""
import asyncio
import time

import pytest

from app import cancellation, state
from app.config import settings
from app.models import LessonResult, TaskStatus
from app.services import transcriber
from app.services.errors import TaskCancelled


# ── Isolated-DB scenario helper ───────────────────────────────────────────────

def _run_with_db(tmp_path, monkeypatch, coro_factory):
    """Run one async scenario against a fresh temp DB in a single event loop.

    aiosqlite binds its connection to the loop that opened it, so the whole
    scenario must live inside one asyncio.run — mirrors tests/test_events_sse.
    """
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "tm.db")
        monkeypatch.setattr(state, "_db", None, raising=False)
        await state.init_db()
        try:
            await coro_factory()
        finally:
            await state.close_db()

    try:
        asyncio.run(wrapper())
    finally:
        # asyncio.run() leaves the thread with set_event_loop(None); restore a
        # usable current loop so sibling tests that still use the deprecated
        # asyncio.get_event_loop() (e.g. test_timestamp_player) keep working
        # regardless of collection order.
        asyncio.set_event_loop(asyncio.new_event_loop())


async def _backdate(task_id: str, iso: str) -> None:
    db = await state._get_db()
    await db.execute("UPDATE tasks SET created_at=? WHERE id=?", [iso, task_id])
    await db.commit()


_OLD_ISO = "2000-01-01T00:00:00+00:00"


# ── Cancellation registry ─────────────────────────────────────────────────────

def test_cancellation_registry_lifecycle():
    cancellation.request_cancel("abc")
    assert cancellation.is_cancelled("abc")
    cancellation.clear("abc")
    assert not cancellation.is_cancelled("abc")
    # clear is idempotent and never raises for an unknown id
    cancellation.clear("abc")
    cancellation.clear("never-seen")


# ── State layer: cancel / requeue / finalize ──────────────────────────────────

def test_cancel_task_sets_cancelled(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.cancel_task("t1")
        t = await state.get_task("t1")
        assert t.status == TaskStatus.CANCELLED
        assert "בוטל" in t.message

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_requeue_task_resets_error_and_partial(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.append_partial_transcript("t1", "[00:00] חלק מהתמלול ")
        await state.fail_task("t1", "boom", detail="stacktrace")
        assert (await state.get_task("t1")).status == TaskStatus.FAILED

        await state.requeue_task("t1")
        t = await state.get_task("t1")
        assert t.status == TaskStatus.PENDING
        assert t.error is None
        # Stale live-transcript preview must be cleared for the fresh run
        _, total = await state.get_partial_transcript("t1")
        assert total == 0

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_finalize_job_payload_scrubs_cookies_keeps_rest(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("j1", "https://z", user_id="u1")
        await state.set_job_payload("j1", {
            "url": "https://z", "mode": "gemini_direct",
            "language": "he", "cookies": "SECRET-COOKIE",
        })
        await state.finalize_job_payload("j1")
        p = await state.get_job_payload("j1")
        # Secret gone, everything needed for retry preserved
        assert p["cookies"] is None
        assert p["url"] == "https://z"
        assert p["mode"] == "gemini_direct"
        assert p["language"] == "he"

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── State layer: retention candidate query ────────────────────────────────────

def test_list_reclaimable_media_filters_by_age_and_status(tmp_path, monkeypatch):
    async def scenario():
        # old + completed + persisted audio  → reclaimable
        await state.create_task("old_c", "upload:a", user_id="u1")
        await state.set_audio_path("old_c", str(tmp_path / "a.mp3"))
        await state.complete_task("old_c", LessonResult(summary="s", language="he"))
        await _backdate("old_c", _OLD_ISO)

        # old + failed + retained upload source (payload file_path) → reclaimable
        await state.create_task("old_f", "upload:b", user_id="u1")
        await state.set_job_payload("old_f", {"file_path": "/data/downloads/b.mp4", "mode": "gemini_direct"})
        await state.fail_task("old_f", "boom")
        await _backdate("old_f", _OLD_ISO)

        # recent + completed → excluded by the cutoff (created_at = now)
        await state.create_task("new_c", "upload:c", user_id="u1")
        await state.set_audio_path("new_c", str(tmp_path / "c.mp3"))
        await state.complete_task("new_c", LessonResult(summary="s", language="he"))

        # old but still pending (non-terminal) → excluded by status
        await state.create_task("old_p", "upload:d", user_id="u1")
        await state.set_audio_path("old_p", str(tmp_path / "d.mp3"))
        await _backdate("old_p", _OLD_ISO)

        rows = await state.list_reclaimable_media("2020-01-01T00:00:00+00:00")
        ids = {r["id"] for r in rows}
        assert "old_c" in ids and "old_f" in ids
        assert "new_c" not in ids and "old_p" not in ids

        failed_row = next(r for r in rows if r["id"] == "old_f")
        assert failed_row["payload_file_path"] == "/data/downloads/b.mp4"

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Retention sweep (main.py) ─────────────────────────────────────────────────

def test_safe_to_delete_guards_data_root(tmp_path, monkeypatch):
    import app.main as main_module

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(main_module, "_DATA_ROOT", root.resolve())

    inside = root / "a.mp3"
    inside.write_bytes(b"x")
    outside = tmp_path / "b.mp3"
    outside.write_bytes(b"y")

    assert main_module._safe_to_delete(str(inside)) == inside.resolve()
    assert main_module._safe_to_delete(str(outside)) is None      # traversal guard
    assert main_module._safe_to_delete(None) is None
    assert main_module._safe_to_delete(str(root / "missing.mp3")) is None


def test_retention_sweep_reclaims_media_keeps_transcript(tmp_path, monkeypatch):
    import app.main as main_module

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    old_audio = audio_dir / "old.mp3"
    old_audio.write_bytes(b"x" * 256)

    async def scenario():
        monkeypatch.setattr(settings, "media_retention_days", 7)
        monkeypatch.setattr(main_module, "_DATA_ROOT", tmp_path.resolve())

        await state.create_task("old1", "upload:x", user_id="u1")
        await state.set_audio_path("old1", str(old_audio))
        await state.complete_task("old1", LessonResult(summary="keep-me", transcript="txt", language="he"))
        await _backdate("old1", _OLD_ISO)

        await main_module._retention_sweep()

        assert not old_audio.exists()                # heavy media reclaimed
        t = await state.get_task("old1")
        assert t is not None                         # DB row intact
        assert t.result.summary == "keep-me"         # transcript/summary preserved
        assert t.has_audio is False                  # audio_path nulled

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_retention_sweep_disabled_when_days_zero(tmp_path, monkeypatch):
    import app.main as main_module

    keep = tmp_path / "keep.mp3"
    keep.write_bytes(b"x")

    async def scenario():
        monkeypatch.setattr(settings, "media_retention_days", 0)   # disabled
        monkeypatch.setattr(main_module, "_DATA_ROOT", tmp_path.resolve())
        await state.create_task("old1", "upload:x", user_id="u1")
        await state.set_audio_path("old1", str(keep))
        await state.complete_task("old1", LessonResult(summary="s", language="he"))
        await _backdate("old1", _OLD_ISO)

        await main_module._retention_sweep()
        assert keep.exists()   # sweep is a no-op when retention is off

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Transcriber cooperative cancel ────────────────────────────────────────────

class _FakeInfo:
    duration = 10.0
    language = "he"


class _Seg:
    def __init__(self, text, start=0.0, end=1.0):
        self.text = text
        self.start = start
        self.end = end


class _SegModel:
    def __init__(self, segs):
        self._segs = segs

    def transcribe(self, path, **kwargs):
        return iter(self._segs), _FakeInfo()


def test_transcribe_sync_aborts_when_cancelled():
    model = _SegModel([_Seg("שלום"), _Seg("עולם")])
    with pytest.raises(TaskCancelled):
        transcriber._transcribe_sync(model, "x.mp3", "he", cancel_cb=lambda: True)


def test_transcribe_sync_completes_when_not_cancelled():
    model = _SegModel([_Seg("שלום", 0, 1), _Seg("עולם", 1, 2)])
    text, lang = transcriber._transcribe_sync(model, "x.mp3", "he", cancel_cb=lambda: False)
    assert "שלום" in text and "עולם" in text
    assert lang == "he"


# ── Endpoint contract: /cancel and /retry ─────────────────────────────────────

def _login(client, mock_email):
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


def _upload(client, mode: str = "gemini_direct") -> str:
    resp = client.post(
        "/api/tasks/upload",
        files={"file": ("lecture.mp4", b"fake media bytes", "video/mp4")},
        data={"mode": mode, "language": "he"},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _wait_for(client, task_id: str, statuses, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    task = None
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in statuses:
            return task
        time.sleep(0.05)
    pytest.fail(f"task {task_id} never reached {statuses}: {task}")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)


def test_cancel_unknown_task_404(client, mock_email, isolated_dirs):
    _login(client, mock_email)
    r = client.post("/api/tasks/no-such-task/cancel")
    assert r.status_code == 404


def test_retry_unknown_task_404(client, mock_email, isolated_dirs):
    _login(client, mock_email)
    r = client.post("/api/tasks/no-such-task/retry")
    assert r.status_code == 404


def _mock_gemini(monkeypatch):
    import app.services.summarizer as summarizer

    async def ok(audio_path, progress_cb):
        return LessonResult(summary="סיכום", chapters=[], quiz=[], language="he")

    async def no_cards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", ok)
    monkeypatch.setattr(summarizer, "generate_flashcards", no_cards)


def test_cancel_and_retry_rejected_for_completed(client, mock_email, monkeypatch, isolated_dirs):
    _mock_gemini(monkeypatch)
    _login(client, mock_email)
    task_id = _upload(client)
    _wait_for(client, task_id, ("completed",))

    # A finished task is neither cancellable nor retryable
    assert client.post(f"/api/tasks/{task_id}/cancel").status_code == 409
    assert client.post(f"/api/tasks/{task_id}/retry").status_code == 409


def test_fail_then_retry_succeeds(client, mock_email, monkeypatch, isolated_dirs):
    """Full round trip: a transient failure, then retry re-runs the same
    uploaded file and completes — proving the source file and payload survive
    the failure and the worker re-processes on demand."""
    import app.services.summarizer as summarizer

    calls = {"n": 0}

    async def flaky(audio_path, progress_cb):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient upstream error")
        return LessonResult(summary="הצליח בניסיון השני", chapters=[], quiz=[], language="he")

    async def no_cards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", flaky)
    monkeypatch.setattr(summarizer, "generate_flashcards", no_cards)

    _login(client, mock_email)
    task_id = _upload(client)
    _wait_for(client, task_id, ("failed",))

    r = client.post(f"/api/tasks/{task_id}/retry")
    assert r.status_code == 202, r.text

    task = _wait_for(client, task_id, ("completed",))
    assert task["result"]["summary"] == "הצליח בניסיון השני"
    assert calls["n"] == 2  # ran once (failed) + once (retry succeeded)
