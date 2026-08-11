"""Tests for the task-events hub and the SSE endpoint.

The endpoint test drives a real upload through the worker (like
test_upload_pipeline) and then reads the /events stream for the completed
task — for terminal tasks the stream sends snapshot + done and closes, so a
plain GET collects the whole body.
"""
import asyncio
import json
import time

import pytest

from app.config import settings
from app.events import TaskEventHub
from app.models import LessonResult


# ── Hub unit tests ────────────────────────────────────────────────────────────


def test_hub_delivers_to_subscribers():
    async def scenario():
        hub = TaskEventHub()
        q1 = hub.subscribe("t1")
        q2 = hub.subscribe("t1")
        other = hub.subscribe("t2")

        hub.publish("t1", {"type": "status", "progress": 50})

        assert q1.get_nowait()["progress"] == 50
        assert q2.get_nowait()["progress"] == 50
        assert other.empty(), "events must not leak across tasks"

    asyncio.run(scenario())


def test_hub_unsubscribe_stops_delivery():
    async def scenario():
        hub = TaskEventHub()
        q = hub.subscribe("t1")
        hub.unsubscribe("t1", q)
        hub.publish("t1", {"type": "status"})
        assert q.empty()

    asyncio.run(scenario())


def test_hub_drops_slow_subscriber_instead_of_blocking():
    async def scenario():
        hub = TaskEventHub()
        q = hub.subscribe("t1")
        for i in range(500):  # exceed the queue cap
            hub.publish("t1", {"i": i})
        # The slow queue got dropped; publishing again must not raise
        hub.publish("t1", {"type": "done"})

    asyncio.run(scenario())


# ── SSE endpoint (full stack) ─────────────────────────────────────────────────


def _login(client, mock_email):
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


def _upload(client, mode: str):
    resp = client.post(
        "/api/tasks/upload",
        files={"file": ("lecture.mp4", b"fake media bytes", "video/mp4")},
        data={"mode": mode, "language": "he"},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _wait_for_terminal(client, task_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    task = None
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    pytest.fail(f"task {task_id} never reached a terminal status: {task}")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_events_stream_for_completed_task(client, mock_email, monkeypatch, isolated_dirs):
    import app.services.summarizer as summarizer

    async def fake_summarize_audio(audio_path, progress_cb, language="he"):
        return LessonResult(summary="סיכום", chapters=[], quiz=[], language="he")

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", fake_summarize_audio)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    _login(client, mock_email)
    task_id = _upload(client, "gemini_direct")
    _wait_for_terminal(client, task_id)

    r = client.get(f"/api/tasks/{task_id}/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    assert events[0]["type"] == "snapshot"
    assert events[0]["status"] == "completed"
    assert events[0]["progress"] == 100
    assert events[-1]["type"] == "done"
    assert events[-1]["status"] == "completed"


def test_events_stream_unknown_task_404(client, mock_email, isolated_dirs):
    _login(client, mock_email)
    r = client.get("/api/tasks/no-such-task/events")
    assert r.status_code == 404
