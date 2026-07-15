"""
Tests for Phase 2 of the production-readiness work — security hardening:

  - Explicit ENVIRONMENT gating: production refuses to start on a placeholder
    Resend key, and never logs magic-link tokens even if reached.
  - Session-cookie Secure flag: explicit COOKIE_SECURE override + heuristic default.
  - Credentials (magic tokens, session ids) come from secrets.token_urlsafe.
  - SSRF guard on download URLs: private/loopback targets and non-allowlisted
    hosts are rejected — at submission time (400) and in the downloader.
  - In-process rate limiting: 429 + envelope on exhausted budget.
  - Ownership: legacy user_id-NULL rows are invisible until adopted by the
    startup backfill.
"""
import asyncio
import logging
import re
import socket

import pytest

from app import state
from app.config import settings
from app.services import zoom_downloader
from app.services.zoom_downloader import ZoomDownloadError, ensure_url_allowed


# ── Isolated-DB scenario helper (same pattern as test_state_guards) ───────────

def _run_with_db(tmp_path, monkeypatch, coro_factory):
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "security.db")
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


# ── Environment gating ────────────────────────────────────────────────────────

def test_production_startup_requires_real_resend_key(tmp_path, monkeypatch):
    """ENVIRONMENT=production + placeholder key must fail fast at startup."""
    import app.state as state_module

    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "prod.db")
    monkeypatch.setattr(state_module, "_db", None, raising=False)
    monkeypatch.setattr(settings, "environment", "production", raising=False)
    monkeypatch.setattr(settings, "resend_api_key", "dummy")

    from fastapi.testclient import TestClient
    from app.main import app

    with pytest.raises(RuntimeError, match="RESEND_API_KEY"):
        with TestClient(app):
            pass


def test_production_never_logs_magic_link(client, monkeypatch, caplog):
    """Even if the unreachable branch is reached (key wiped at runtime), the
    token must not appear in the logs in production."""
    monkeypatch.setattr(settings, "environment", "production", raising=False)
    monkeypatch.setattr(settings, "resend_api_key", "")

    with caplog.at_level(logging.INFO, logger="app.api.auth"):
        resp = client.post("/api/auth/request", json={"email": "allowed@example.com"})

    assert resp.status_code == 200  # same anti-enumeration response as always
    assert not any("DEV LOGIN" in r.message for r in caplog.records)
    assert not any("token=" in r.message for r in caplog.records)
    assert any("NOT sent" in r.message for r in caplog.records)


def test_dev_bypass_still_works_in_development(client, monkeypatch, caplog):
    monkeypatch.setattr(settings, "resend_api_key", "")
    with caplog.at_level(logging.INFO, logger="app.api.auth"):
        client.post("/api/auth/request", json={"email": "allowed@example.com"})
    assert any("DEV LOGIN" in r.message for r in caplog.records)


# ── Cookie policy ─────────────────────────────────────────────────────────────

def _login_set_cookie_header(client, mock_email) -> str:
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    return resp.headers["set-cookie"].lower()


def test_cookie_secure_default_heuristic(client, mock_email):
    """base_url http://testserver (not localhost) → Secure by default."""
    header = _login_set_cookie_header(client, mock_email)
    assert "secure" in header
    assert "httponly" in header
    assert "samesite=lax" in header


def test_cookie_secure_explicit_override(client, mock_email, monkeypatch):
    """COOKIE_SECURE=false (plain-http LAN serving) drops only the Secure flag."""
    monkeypatch.setattr(settings, "cookie_secure", False, raising=False)
    header = _login_set_cookie_header(client, mock_email)
    assert "secure" not in header
    assert "httponly" in header


# ── Credential generation ─────────────────────────────────────────────────────

def test_credentials_are_urlsafe_high_entropy(tmp_path, monkeypatch):
    async def scenario():
        user_id = await state.get_or_create_user("a@example.com")
        token = await state.create_magic_token(user_id)
        session = await state.create_session(user_id)
        for cred in (token, session):
            assert len(cred) >= 40                      # token_urlsafe(32) → 43 chars
            assert re.fullmatch(r"[A-Za-z0-9_-]+", cred)
            assert "-" in cred or cred.count("-") == 0  # not a uuid4 shape (8-4-4-4-12)
        assert not re.fullmatch(r"[0-9a-f-]{36}", token)
        assert not re.fullmatch(r"[0-9a-f-]{36}", session)

    _run_with_db(tmp_path, monkeypatch, scenario)


# ── SSRF guard ────────────────────────────────────────────────────────────────

async def test_url_guard_rejects_private_literal_ips():
    for url in (
        "http://192.168.1.10/rec/share/x",
        "http://10.0.0.5/x",
        "http://127.0.0.1:8000/api/tasks",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data/",
    ):
        with pytest.raises(ZoomDownloadError):
            await ensure_url_allowed(url)


async def test_url_guard_rejects_host_resolving_to_private(monkeypatch):
    async def fake_resolve(host):
        return {"10.0.0.5"}

    monkeypatch.setattr(zoom_downloader, "_resolve_host", fake_resolve)
    with pytest.raises(ZoomDownloadError):
        await ensure_url_allowed("https://internal.corp.example/x")


async def test_url_guard_accepts_public_host(monkeypatch):
    async def fake_resolve(host):
        return {"142.250.185.78"}

    monkeypatch.setattr(zoom_downloader, "_resolve_host", fake_resolve)
    await ensure_url_allowed("https://zoom.us/rec/share/abc")  # must not raise


async def test_url_guard_dns_failure_is_clear_error(monkeypatch):
    async def fake_resolve(host):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(zoom_downloader, "_resolve_host", fake_resolve)
    with pytest.raises(ZoomDownloadError):
        await ensure_url_allowed("https://definitely-not-a-real-host.example/x")


async def test_url_guard_allowlist_suffix_match(monkeypatch):
    monkeypatch.setattr(settings, "allowed_download_hosts", "zoom.us", raising=False)

    async def fake_resolve(host):
        return {"142.250.185.78"}

    monkeypatch.setattr(zoom_downloader, "_resolve_host", fake_resolve)

    await ensure_url_allowed("https://zoom.us/rec/1")             # exact
    await ensure_url_allowed("https://us02web.zoom.us/rec/1")     # subdomain
    with pytest.raises(ZoomDownloadError):
        await ensure_url_allowed("https://evil.com/rec/1")        # not listed
    with pytest.raises(ZoomDownloadError):
        await ensure_url_allowed("https://notzoom.us/rec/1")      # suffix is anchored


async def test_url_guard_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "block_private_download_targets", False, raising=False)
    await ensure_url_allowed("http://127.0.0.1/x")  # must not raise


def _login(client, mock_email):
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


def test_create_task_rejects_private_url_with_400(client, mock_email):
    """Submission-time rejection: immediate 400, no task row created."""
    _login(client, mock_email)
    r = client.post(
        "/api/tasks",
        json={"url": "http://127.0.0.1:9/rec", "mode": "gemini_direct"},
    )
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"
    assert client.get("/api/tasks").json() == []  # nothing was queued


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_auth_rate_limit_returns_429_envelope(client, monkeypatch):
    from app import ratelimit

    monkeypatch.setattr(settings, "rate_limit_enabled", True, raising=False)
    monkeypatch.setattr(settings, "rate_limit_auth_per_minute", 3, raising=False)
    ratelimit.reset()

    for _ in range(3):
        assert client.post(
            "/api/auth/request", json={"email": "nobody@example.com"}
        ).status_code == 200

    r = client.post("/api/auth/request", json={"email": "nobody@example.com"})
    assert r.status_code == 429
    assert r.json()["code"] == "rate_limited"
    ratelimit.reset()


def test_tasks_rate_limit_covers_url_submission(client, mock_email, monkeypatch):
    from app import ratelimit

    _login(client, mock_email)
    monkeypatch.setattr(settings, "rate_limit_enabled", True, raising=False)
    monkeypatch.setattr(settings, "rate_limit_tasks_per_minute", 1, raising=False)
    ratelimit.reset()

    # First submission consumes the budget (SSRF 400 still counts — the
    # dependency runs before the endpoint body).
    first = client.post(
        "/api/tasks", json={"url": "http://127.0.0.1:9/rec", "mode": "gemini_direct"}
    )
    assert first.status_code == 400

    second = client.post(
        "/api/tasks", json={"url": "http://127.0.0.1:9/rec", "mode": "gemini_direct"}
    )
    assert second.status_code == 429
    assert second.json()["code"] == "rate_limited"
    ratelimit.reset()


# ── Legacy ownership ──────────────────────────────────────────────────────────

def test_ownerless_rows_hidden_until_backfilled(tmp_path, monkeypatch):
    async def scenario():
        uid = await state.get_or_create_user("a@example.com")
        await state.create_task("legacy", "upload:x", user_id=None)

        # Strict owner check: an ownerless row is nobody's row
        assert await state.get_task_for_user("legacy", uid) is None

        adopted = await state.backfill_task_owners(uid)
        assert adopted == 1
        task = await state.get_task_for_user("legacy", uid)
        assert task is not None and task.task_id == "legacy"

        # Idempotent: nothing left to adopt
        assert await state.backfill_task_owners(uid) == 0

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_other_users_task_still_looks_like_404(tmp_path, monkeypatch):
    async def scenario():
        owner = await state.get_or_create_user("owner@example.com")
        intruder = await state.get_or_create_user("intruder@example.com")
        await state.create_task("private-task", "upload:x", user_id=owner)
        assert await state.get_task_for_user("private-task", intruder) is None
        assert (await state.get_task_for_user("private-task", owner)) is not None

    _run_with_db(tmp_path, monkeypatch, scenario)
