"""
Tests for the per-user 24-hour task rate-limiting, blocking, and banning system.

Business rules under test:
  • A user may submit at most 2 processing requests in any 24-hour window.
  • The 3rd attempt in that window:
      - Returns HTTP 429 (Too Many Requests)
      - Triggers a 24-hour account block
      - Sends a warning email to the user via Resend
  • Any request while blocked:
      - Permanently bans the account
      - Returns HTTP 403 (Forbidden)
  • Any subsequent request by a permanently banned account: HTTP 403
  • After the 24-hour block expires, the user is automatically unblocked
    (old timestamps are >24h old and pruned, giving the user a fresh slate).

Design notes
------------
- Each test gets a fresh SQLite DB via the shared `client` fixture.
- Authentication uses the real magic-link flow to ensure the user exists in
  the DB and the rate-limit dependency sees a real user_id.
- Only the Resend HTTP call is mocked (mock_warning_email fixture).
- Time is controlled via monkeypatching `app.state._now` for the expiry test.
"""
from datetime import datetime, timezone, timedelta
import pytest


# ── Reset shared limiter between tests ────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_ip_limiter():
    """Clear the in-memory IP-based limiter so it doesn't bleed across tests."""
    from app.rate_limit import limiter
    limiter._windows.clear()
    yield
    limiter._windows.clear()


# ── Auth & request helpers ────────────────────────────────────────────────────

def _login(client, monkeypatch) -> str:
    """Authenticate via magic-link flow. Returns the session_id cookie value."""
    import app.api.auth as auth_module
    captured: list[str] = []

    async def _fake_magic(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", _fake_magic)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def _post_task(client, sid: str):
    """POST one task-from-URL request. Returns the response."""
    return client.post(
        "/api/tasks",
        json={"url": "https://zoom.us/rec/test", "mode": "gemini_direct"},
        cookies={"session_id": sid},
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_warning_email(monkeypatch):
    """
    Capture rate-limit warning emails instead of calling Resend.
    Returns the list of email addresses that would have been warned.
    """
    sent: list[str] = []

    async def _fake_warn(email: str) -> None:
        sent.append(email)

    import app.api.auth as auth_module
    monkeypatch.setattr(auth_module, "_send_rate_limit_warning_email", _fake_warn)
    return sent


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_allows_first_two_requests(client, monkeypatch, mock_warning_email):
    """Requests 1 and 2 within a 24-hour window must both return 202."""
    sid = _login(client, monkeypatch)

    assert _post_task(client, sid).status_code == 202
    assert _post_task(client, sid).status_code == 202


def test_third_request_returns_429(client, monkeypatch, mock_warning_email):
    """The 3rd request in a 24-hour window must return 429 Too Many Requests."""
    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st — allowed
    _post_task(client, sid)  # 2nd — allowed
    resp = _post_task(client, sid)  # 3rd — must be rejected

    assert resp.status_code == 429


def test_warning_email_sent_on_third_request(client, monkeypatch, mock_warning_email):
    """On the 3rd (blocking) request the warning email must be dispatched once."""
    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st
    _post_task(client, sid)  # 2nd
    _post_task(client, sid)  # 3rd → triggers block + email

    assert mock_warning_email == ["allowed@example.com"]


def test_no_warning_email_on_first_two_requests(client, monkeypatch, mock_warning_email):
    """Warning email must NOT be sent for requests 1 or 2."""
    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st
    _post_task(client, sid)  # 2nd

    assert mock_warning_email == []


def test_request_while_blocked_returns_403(client, monkeypatch, mock_warning_email):
    """A request made while the 24-hour block is active must return 403 Forbidden."""
    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st
    _post_task(client, sid)  # 2nd
    _post_task(client, sid)  # 3rd → 24h block applied

    resp = _post_task(client, sid)  # 4th — while blocked → permanent ban + 403
    assert resp.status_code == 403


def test_banned_user_always_gets_403(client, monkeypatch, mock_warning_email):
    """After the permanent ban is applied, every subsequent request returns 403."""
    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st
    _post_task(client, sid)  # 2nd
    _post_task(client, sid)  # 3rd → blocked
    _post_task(client, sid)  # 4th → permanently banned

    # All further requests must also be rejected
    assert _post_task(client, sid).status_code == 403
    assert _post_task(client, sid).status_code == 403


def test_upload_endpoint_also_enforces_user_rate_limit(
    client, monkeypatch, mock_warning_email
):
    """POST /api/tasks/upload must enforce the same per-user daily limit."""
    import io
    import app.services.processor as _proc

    async def _noop(**kwargs):
        pass

    monkeypatch.setattr(_proc, "run_pipeline_from_file", _noop)

    sid = _login(client, monkeypatch)

    def _upload():
        return client.post(
            "/api/tasks/upload",
            files={"file": ("rec.mp3", io.BytesIO(b"\xff\xfb" + b"\x00" * 1024), "audio/mpeg")},
            data={"mode": "gemini_direct", "language": "he"},
            cookies={"session_id": sid},
        )

    assert _upload().status_code == 202  # 1st
    assert _upload().status_code == 202  # 2nd
    assert _upload().status_code == 429  # 3rd → blocked


def test_expired_block_resets_limits(client, monkeypatch, mock_warning_email):
    """
    After the 24-hour block_until timestamp passes, the user is automatically
    unblocked. Old request timestamps are also >24h old and pruned, giving a
    fresh quota.
    """
    import app.state as state_module

    sid = _login(client, monkeypatch)

    _post_task(client, sid)  # 1st
    _post_task(client, sid)  # 2nd
    assert _post_task(client, sid).status_code == 429  # 3rd → blocked

    # Time-travel: shift _now() 25 hours forward so the block has expired
    future = datetime.now(timezone.utc) + timedelta(hours=25)
    monkeypatch.setattr(state_module, "_now", lambda: future)

    resp = _post_task(client, sid)
    assert resp.status_code == 202  # allowed again after block expires
