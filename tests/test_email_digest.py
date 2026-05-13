"""Tests for the weekly email digest (Batch B2).

We exercise the pure HTML builder + the gating helper + the run_digest_cycle
orchestrator with the Resend call mocked. The scheduler itself is just an
asyncio loop wrapping run_digest_cycle — tested implicitly via that.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import LessonResult
from app.services import email_digest


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


# ── HTML builder ───────────────────────────────────────────────────────────

def test_build_digest_html_contains_titles_and_count():
    from app.models import TaskResponse, TaskStatus

    tasks = [
        TaskResponse(
            task_id="t1", status=TaskStatus.COMPLETED, progress=100, message="ok",
            created_at="2026-05-10T08:00:00+00:00", url="https://x/1",
            result=LessonResult(summary="פגישת תכנון Q3"),
        ),
        TaskResponse(
            task_id="t2", status=TaskStatus.COMPLETED, progress=100, message="ok",
            created_at="2026-05-11T08:00:00+00:00", url="https://x/2",
            result=LessonResult(summary="הרצאה — אומגה 3"),
        ),
    ]
    html = email_digest.build_digest_html("user@example.com", tasks, "https://app.example.com")
    assert "user@example.com" in html
    assert "פגישת תכנון Q3" in html
    assert "אומגה 3" in html
    assert "/?task=t1" in html
    assert "/?task=t2" in html
    assert "2026-05-10" in html
    assert "<strong>2</strong>" in html  # task count


def test_build_digest_html_escapes_user_content():
    from app.models import TaskResponse, TaskStatus

    tasks = [
        TaskResponse(
            task_id="t1", status=TaskStatus.COMPLETED, progress=100, message="ok",
            created_at="2026-05-10T08:00:00+00:00", url="https://x/1",
            result=LessonResult(summary="<script>alert('x')</script>"),
        ),
    ]
    html = email_digest.build_digest_html("u@x.com", tasks, "https://app/")
    # The literal tag must not appear unescaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ── Gating helper ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "delta_days,expected",
    [
        (1, False),
        (3, False),
        (6, False),
        (7, True),
        (10, True),
    ],
)
def test_user_due_for_digest_period(delta_days, expected):
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    last = (now - timedelta(days=delta_days)).isoformat()
    due = asyncio.get_event_loop().run_until_complete(
        email_digest._user_due_for_digest(last, now=now)
    )
    assert due is expected


def test_user_due_when_never_sent():
    due = asyncio.get_event_loop().run_until_complete(
        email_digest._user_due_for_digest(None)
    )
    assert due is True


# ── Preferences endpoints ──────────────────────────────────────────────────

def test_get_and_set_preferences_round_trip(client):
    # Seed the user
    async def _seed():
        await state.get_or_create_user("digest-user@example.com")
    asyncio.get_event_loop().run_until_complete(_seed())

    uid = asyncio.get_event_loop().run_until_complete(
        state.get_or_create_user("digest-user@example.com")
    )

    _override_user(uid)
    try:
        # Default: opted out
        resp = client.get("/api/auth/me/preferences")
        assert resp.status_code == 200
        assert resp.json()["email_digest_opt_in"] is False

        # Opt in
        resp = client.put(
            "/api/auth/me/preferences",
            json={"email_digest_opt_in": True},
        )
        assert resp.status_code == 200
        assert resp.json()["email_digest_opt_in"] is True

        # Opt back out
        resp = client.put(
            "/api/auth/me/preferences",
            json={"email_digest_opt_in": False},
        )
        assert resp.json()["email_digest_opt_in"] is False
    finally:
        _clear_override()


# ── End-to-end cycle (Resend mocked) ───────────────────────────────────────

def test_run_digest_cycle_sends_for_due_user_with_new_tasks(client, monkeypatch):
    """A user opted-in, never emailed before, with one new task → 1 send."""
    captured: dict = {}

    async def _fake_send(email_addr, body):
        captured["email"] = email_addr
        captured["body"] = body

    monkeypatch.setattr(email_digest, "send_digest_email", _fake_send)
    # Ensure the cycle doesn't bail out early due to missing API key
    monkeypatch.setattr(email_digest.settings, "resend_api_key", "test-key")

    async def _seed():
        uid = await state.get_or_create_user("alice@example.com")
        await state.set_email_digest_opt_in(uid, True)
        await state.create_task("digest-task-1", "https://x/1", user_id=uid)
        await state.complete_task(
            "digest-task-1", LessonResult(summary="הרצאה חדשה")
        )
        return uid

    uid = asyncio.get_event_loop().run_until_complete(_seed())

    sent = asyncio.get_event_loop().run_until_complete(email_digest.run_digest_cycle())
    assert sent == 1
    assert captured["email"] == "alice@example.com"
    assert "הרצאה חדשה" in captured["body"]

    # Calling again should NOT resend — last_digest_at is now fresh
    sent2 = asyncio.get_event_loop().run_until_complete(email_digest.run_digest_cycle())
    assert sent2 == 0


def test_run_digest_cycle_skips_opted_out_users(client, monkeypatch):
    """Opted-out users are never queried for new tasks."""
    calls = []

    async def _fake_send(email_addr, body):
        calls.append(email_addr)

    monkeypatch.setattr(email_digest, "send_digest_email", _fake_send)
    monkeypatch.setattr(email_digest.settings, "resend_api_key", "test-key")

    async def _seed():
        uid = await state.get_or_create_user("bob@example.com")
        # Note: NOT setting opt-in
        await state.create_task("digest-task-2", "https://x/2", user_id=uid)
        await state.complete_task("digest-task-2", LessonResult(summary="x"))

    asyncio.get_event_loop().run_until_complete(_seed())
    sent = asyncio.get_event_loop().run_until_complete(email_digest.run_digest_cycle())
    assert sent == 0
    assert calls == []


def test_run_digest_cycle_skips_users_without_new_tasks(client, monkeypatch):
    """Opted-in user with no completed tasks in the last 7d → no send."""
    calls = []

    async def _fake_send(email_addr, body):
        calls.append(email_addr)

    monkeypatch.setattr(email_digest, "send_digest_email", _fake_send)
    monkeypatch.setattr(email_digest.settings, "resend_api_key", "test-key")

    async def _seed():
        uid = await state.get_or_create_user("carol@example.com")
        await state.set_email_digest_opt_in(uid, True)
        # No tasks at all

    asyncio.get_event_loop().run_until_complete(_seed())
    sent = asyncio.get_event_loop().run_until_complete(email_digest.run_digest_cycle())
    assert sent == 0
