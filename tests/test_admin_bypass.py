"""
Tests for admin-email bypass of rate limits.

Business rules under test:
  • Users whose email is in settings.admin_emails are NOT subject to the
    per-user 24-hour task quota — they may post unlimited tasks.
  • Admin users also bypass the per-IP rate limit applied by
    `app.rate_limit.limiter`.
  • `state.is_admin_user(user_id)` correctly checks email membership.
  • `state.reset_admin_flags()` clears block_until / is_banned for admin
    rows (idempotent; safe to call on every boot).
  • The `/api/capabilities` endpoint reports `is_admin` truthfully so the
    UI can show a badge.
"""
import pytest

from app.config import settings


# Reset the shared IP limiter between tests so rate state never bleeds across.
@pytest.fixture(autouse=True)
def _reset_ip_limiter():
    from app.rate_limit import limiter
    limiter._windows.clear()
    yield
    limiter._windows.clear()


def _login(client, monkeypatch, email: str = "allowed@example.com") -> str:
    """Run the magic-link flow for `email` and return the session_id cookie."""
    import app.api.auth as auth_module
    captured: list[str] = []

    async def _fake_magic(_email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", _fake_magic)
    # Make sure this email is in the allowed-list so /auth/request actually
    # creates a token. Tests run with allowed_emails="allowed@example.com",
    # so callers passing a custom email must extend the list themselves.
    client.post("/api/auth/request", json={"email": email})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def _post_task(client, sid: str):
    return client.post(
        "/api/tasks",
        json={"url": "https://zoom.us/rec/test", "mode": "gemini_direct"},
        cookies={"session_id": sid},
    )


# ── Per-user quota bypass ─────────────────────────────────────────────────────


def test_admin_can_submit_unlimited_tasks(client, monkeypatch):
    """
    Admin email bypasses the user_daily_task_limit (which is normally 2/24h).
    Five back-to-back submissions all succeed.
    """
    monkeypatch.setattr(settings, "admin_emails", "allowed@example.com")
    # Disable the IP rate limit so 5 requests within one minute don't trip it
    # (we test the IP bypass separately).
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)

    sid = _login(client, monkeypatch)

    for _ in range(5):
        assert _post_task(client, sid).status_code == 202


def test_non_admin_still_hits_quota(client, monkeypatch):
    """Sanity check — bypass really requires admin membership, not bypassing all users."""
    monkeypatch.setattr(settings, "admin_emails", "someone-else@example.com")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)

    # Stub out the Resend warning email so the 3rd-request block path does not
    # try to make a real HTTPS call (which fails under corporate SSL).
    import app.api.auth as auth_module

    async def _no_warning(_email: str) -> None:
        return None

    monkeypatch.setattr(auth_module, "_send_rate_limit_warning_email", _no_warning)

    sid = _login(client, monkeypatch)

    assert _post_task(client, sid).status_code == 202
    assert _post_task(client, sid).status_code == 202
    # 3rd request — 429 (the user_daily_task_limit kicks in).
    assert _post_task(client, sid).status_code == 429


# ── IP rate-limit bypass ──────────────────────────────────────────────────────


def test_admin_bypasses_ip_rate_limit(client, monkeypatch):
    """
    With rate_limit_per_minute=1 and admin_emails set to the test email, an
    admin can submit two requests within the same minute without seeing 429.
    """
    monkeypatch.setattr(settings, "admin_emails", "allowed@example.com")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    sid = _login(client, monkeypatch)

    assert _post_task(client, sid).status_code == 202
    assert _post_task(client, sid).status_code == 202


# ── Helpers ───────────────────────────────────────────────────────────────────


async def test_is_admin_user_checks_email_membership(client, monkeypatch):
    """is_admin_user() reads admin_emails fresh on every call (lowercase compared)."""
    from app import state

    monkeypatch.setattr(settings, "admin_emails", "ADMIN@Example.COM, other@x.io")

    admin_uid = await state.get_or_create_user("admin@example.com")
    user_uid = await state.get_or_create_user("regular@example.com")

    assert await state.is_admin_user(admin_uid) is True
    assert await state.is_admin_user(user_uid) is False


async def test_reset_admin_flags_clears_block_and_ban(client, monkeypatch):
    """reset_admin_flags() unblocks any admin row even if already permanently banned."""
    from app import state

    monkeypatch.setattr(settings, "admin_emails", "boss@example.com")

    boss_uid = await state.get_or_create_user("boss@example.com")
    db = await state._get_db()
    await db.execute(
        "UPDATE users SET block_until=?, is_banned=1, request_timestamps=? WHERE id=?",
        ["2099-01-01T00:00:00+00:00", '["2099-01-01T00:00:00+00:00"]', boss_uid],
    )
    await db.commit()

    cleared = await state.reset_admin_flags()
    assert cleared >= 1

    async with db.execute(
        "SELECT block_until, is_banned, request_timestamps FROM users WHERE id=?",
        [boss_uid],
    ) as cursor:
        row = await cursor.fetchone()
    assert row["block_until"] is None
    assert row["is_banned"] == 0
    assert row["request_timestamps"] == "[]"


# ── Capabilities exposes admin flag ──────────────────────────────────────────


def test_capabilities_reports_is_admin_true_for_admin(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_emails", "allowed@example.com")
    sid = _login(client, monkeypatch)
    r = client.get("/api/capabilities", cookies={"session_id": sid})
    assert r.status_code == 200
    assert r.json()["is_admin"] is True


def test_capabilities_reports_is_admin_false_for_regular_user(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_emails", "someone-else@example.com")
    sid = _login(client, monkeypatch)
    r = client.get("/api/capabilities", cookies={"session_id": sid})
    assert r.status_code == 200
    assert r.json()["is_admin"] is False
