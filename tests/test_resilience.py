"""
Tests for Phase 3 of the production-readiness work — resilience & data integrity:

  - Atomic writes: user upsert under concurrency, single-statement chat
    append (no lost messages, cap enforced, corrupt-column recovery),
    payload cookie scrub via json_set.
  - Whisper-pool drain: shutdown helper waits for in-flight thread work.
  - Orphan-file hygiene: upload rollback on post-write failure, DELETE
    reclaiming the retained source, retention sweep removing unreferenced
    strays but never payload-referenced files.
"""
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from app import state
from app.config import settings
from app.models import LessonResult


# ── Isolated-DB scenario helper (same pattern as test_state_guards) ───────────

def _run_with_db(tmp_path, monkeypatch, coro_factory):
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "resilience.db")
        monkeypatch.setattr(state, "_db", None, raising=False)
        await state.init_db()
        try:
            await coro_factory()
        finally:
            await state.close_db()

    try:
        asyncio.run(wrapper())
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


# ── Atomic user upsert ────────────────────────────────────────────────────────

def test_concurrent_user_creation_yields_one_row(tmp_path, monkeypatch):
    """Five concurrent first-logins for the same email must produce exactly one
    user and no UNIQUE-constraint crash (the old check-then-insert raced)."""
    async def scenario():
        ids = await asyncio.gather(
            *(state.get_or_create_user("race@example.com") for _ in range(5))
        )
        assert len(set(ids)) == 1
        db = await state._get_db()
        async with db.execute(
            "SELECT COUNT(*) AS n FROM users WHERE email=?", ["race@example.com"]
        ) as cursor:
            row = await cursor.fetchone()
        assert row["n"] == 1

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Atomic chat append ────────────────────────────────────────────────────────

def test_concurrent_chat_appends_lose_nothing(tmp_path, monkeypatch):
    """Ten concurrent appends must all land (the old read-modify-write let
    interleaved writers overwrite each other's messages)."""
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await asyncio.gather(
            *(state.append_chat_message("t1", "user", f"msg-{i}") for i in range(10))
        )
        history = await state.get_chat_history("t1")
        assert len(history) == 10
        assert {m["content"] for m in history} == {f"msg-{i}" for i in range(10)}

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_chat_history_cap_enforced(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        for i in range(state._MAX_CHAT_MESSAGES + 5):
            await state.append_chat_message("t1", "user", f"m{i}")
        history = await state.get_chat_history("t1")
        assert len(history) == state._MAX_CHAT_MESSAGES
        # Oldest dropped first — the newest message is always present
        assert history[-1]["content"] == f"m{state._MAX_CHAT_MESSAGES + 4}"

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_chat_append_recovers_from_corrupt_column(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        db = await state._get_db()
        await db.execute(
            "UPDATE tasks SET chat_history=? WHERE id=?", ["{not json", "t1"]
        )
        await db.commit()
        await state.append_chat_message("t1", "user", "fresh start")
        history = await state.get_chat_history("t1")
        assert history == [{"role": "user", "content": "fresh start"}]

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_chat_append_preserves_hebrew(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("t1", "upload:x", user_id="u1")
        await state.append_chat_message("t1", "user", 'שאלה על "מרכאות" ו-JSON')
        history = await state.get_chat_history("t1")
        assert history[0]["content"] == 'שאלה על "מרכאות" ו-JSON'

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Payload cookie scrub (json_set path) ──────────────────────────────────────

def test_finalize_payload_scrub_is_single_statement(tmp_path, monkeypatch):
    async def scenario():
        await state.create_task("j1", "https://z", user_id="u1")
        await state.set_job_payload("j1", {
            "url": "https://z", "mode": "gemini_direct",
            "language": "he", "cookies": "SECRET",
        })
        await state.finalize_job_payload("j1")
        payload = await state.get_job_payload("j1")
        assert payload["cookies"] is None
        assert payload["url"] == "https://z"

        # No payload at all → no-op, no crash
        await state.create_task("j2", "https://z", user_id="u1")
        await state.finalize_job_payload("j2")
        assert await state.get_job_payload("j2") is None

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── Whisper-pool drain ────────────────────────────────────────────────────────

def test_drain_waits_for_in_flight_thread_work():
    from app.services import transcriber

    done = {"flag": False}

    def slow_job():
        time.sleep(0.3)
        done["flag"] = True

    async def scenario():
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(transcriber._whisper_pool, slow_job)
        assert await transcriber.drain(timeout=5.0) is True
        assert done["flag"] is True, "drain returned before the thread job finished"
        await future

    asyncio.run(scenario())
    asyncio.set_event_loop(asyncio.new_event_loop())


def test_drain_times_out_but_does_not_raise():
    from app.services import transcriber

    release = {"stop": False}

    def wedged_job():
        while not release["stop"]:
            time.sleep(0.05)

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.run_in_executor(transcriber._whisper_pool, wedged_job)
        assert await transcriber.drain(timeout=0.2) is False
        release["stop"] = True  # unwedge so the pool is clean for later tests
        await transcriber.drain(timeout=5.0)

    asyncio.run(scenario())
    asyncio.set_event_loop(asyncio.new_event_loop())


# ── Orphan-file hygiene ───────────────────────────────────────────────────────

def _login(client, mock_email):
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)


def test_upload_rolls_back_file_when_enqueue_fails(
    client, mock_email, monkeypatch, isolated_dirs
):
    from app.services import worker

    def boom(task_id):
        raise RuntimeError("queue exploded")

    monkeypatch.setattr(worker, "enqueue", boom)
    _login(client, mock_email)

    with pytest.raises(RuntimeError, match="queue exploded"):
        client.post(
            "/api/tasks/upload",
            files={"file": ("lecture.mp4", b"bytes", "video/mp4")},
            data={"mode": "gemini_direct", "language": "he"},
        )

    assert list(settings.downloads_dir.iterdir()) == [], (
        "the uploaded file must be removed when task setup fails"
    )


def test_delete_task_reclaims_retained_upload_source(
    client, mock_email, monkeypatch, isolated_dirs
):
    """A failed upload keeps its source for /retry; deleting the task must
    remove that file too — the task row was its only reference."""
    import app.services.transcriber as transcriber

    async def boom(audio_path, language, task_id=None, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(transcriber, "transcribe", boom)
    _login(client, mock_email)

    resp = client.post(
        "/api/tasks/upload",
        files={"file": ("lecture.mp3", b"fake mp3 bytes", "audio/mpeg")},
        data={"mode": "whisper_local", "language": "he"},
    )
    task_id = resp.json()["task_id"]

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if client.get(f"/api/tasks/{task_id}").json()["status"] == "failed":
            break
        time.sleep(0.05)
    assert any(settings.downloads_dir.iterdir()), "source should be retained on failure"

    assert client.delete(f"/api/tasks/{task_id}").status_code == 204
    assert list(settings.downloads_dir.iterdir()) == []


def test_retention_sweep_removes_orphans_keeps_referenced(tmp_path, monkeypatch):
    import app.main as main_module

    data = tmp_path / "data"
    downloads = data / "downloads"
    downloads.mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", downloads)
    monkeypatch.setattr(settings, "media_retention_days", 7)
    monkeypatch.setattr(main_module, "_DATA_ROOT", data.resolve())

    old_ts = time.time() - 30 * 24 * 3600
    orphan = downloads / "stray.mp3"
    orphan.write_bytes(b"x" * 64)
    os.utime(orphan, (old_ts, old_ts))
    referenced = downloads / "kept.mp3"
    referenced.write_bytes(b"y" * 64)
    os.utime(referenced, (old_ts, old_ts))
    fresh = downloads / "fresh.mp3"
    fresh.write_bytes(b"z" * 64)

    async def scenario():
        # A recent pending task still references kept.mp3 — never delete it
        await state.create_task("pending1", "upload:kept.mp3", user_id="u1")
        await state.set_job_payload(
            "pending1", {"file_path": str(referenced), "mode": "whisper_local"}
        )
        await main_module._retention_sweep()

    _run_with_db(tmp_path, monkeypatch, scenario)

    assert not orphan.exists(), "old unreferenced stray must be reclaimed"
    assert referenced.exists(), "payload-referenced file must survive"
    assert fresh.exists(), "recent file must survive"


def test_orphan_sweep_refuses_foreign_downloads_dir(tmp_path, monkeypatch):
    """Guard: if downloads_dir isn't inside the data root (partially mocked
    env), the orphan pass must be a no-op rather than sweep the wrong tree."""
    import app.main as main_module

    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    victim = foreign / "old.mp3"
    victim.write_bytes(b"x")
    old_ts = time.time() - 30 * 24 * 3600
    os.utime(victim, (old_ts, old_ts))

    monkeypatch.setattr(settings, "downloads_dir", foreign)
    # _DATA_ROOT deliberately left pointing elsewhere (tmp data root)
    monkeypatch.setattr(main_module, "_DATA_ROOT", (tmp_path / "data").resolve())

    deleted, freed = main_module._sweep_orphan_downloads(
        datetime.now(timezone.utc) - timedelta(days=7), set()
    )
    assert deleted == 0
    assert victim.exists()
