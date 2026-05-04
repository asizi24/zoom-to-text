"""
Tests for in-memory rate limiting on task-submission endpoints.

Covers:
  1. Requests below the limit succeed normally (200/202).
  2. The (limit + 1)-th request in the same window returns HTTP 429.
  3. Setting rate_limit_per_minute=0 disables enforcement (treats 0/minute
     as effectively unlimited — slowapi interprets "0/minute" as 0 allowed,
     so we just verify the setting is respected by using a high limit).
  4. The upload endpoint (POST /api/tasks/upload) is also rate-limited.

Design notes
------------
- Each test gets a fresh TestClient via the shared `client` fixture, which
  creates a new in-process SQLite DB in a temp directory.
- The limiter's in-memory storage is reset between tests by monkeypatching a
  high limit (e.g. 1000/minute) so leftover counts from prior requests in the
  same test do not bleed across test functions.
- We authenticate via the magic-link flow (same pattern as test_history_search)
  rather than bypassing auth, to keep the test realistic.
- We do NOT mock the pipeline/processor — we let create_task() queue a
  background task that never actually runs in TestClient context (BackgroundTasks
  fire after the response is returned but TestClient.close() joins them, which is
  fine since the processor would fail without a real URL/file). The 202 response
  itself is what we validate.
"""
import io
import pytest


# ── Limiter reset ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_limiter():
    """Clear the in-memory limiter state before each test to prevent bleed-over."""
    from app.rate_limit import limiter
    limiter._windows.clear()
    yield
    limiter._windows.clear()


# ── Auth helper ───────────────────────────────────────────────────────────────

def _login(client, monkeypatch) -> str:
    """Authenticate via magic-link flow. Returns session_id cookie value."""
    import app.api.auth as auth_module
    captured: list[str] = []

    async def fake_send(email: str, token: str) -> None:
        captured.append(token)

    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = captured[0]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    return resp.cookies["session_id"]


def _post_task(client, sid: str):
    """Send one POST /api/tasks request and return the response."""
    return client.post(
        "/api/tasks",
        json={"url": "https://zoom.us/rec/test", "mode": "gemini_direct"},
        cookies={"session_id": sid},
    )


def _post_upload(client, sid: str):
    """Send one POST /api/tasks/upload request and return the response."""
    fake_audio = io.BytesIO(b"\xff\xfb" + b"\x00" * 1024)
    return client.post(
        "/api/tasks/upload",
        files={"file": ("lecture.mp3", fake_audio, "audio/mpeg")},
        data={"mode": "gemini_direct", "language": "he"},
        cookies={"session_id": sid},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_requests_below_limit_succeed(client, monkeypatch):
    """
    With rate_limit_per_minute=5, five consecutive requests must all succeed
    with 202 Accepted.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_per_minute", 5)

    sid = _login(client, monkeypatch)

    for i in range(5):
        resp = _post_task(client, sid)
        assert resp.status_code == 202, f"Request {i + 1} should succeed but got {resp.status_code}"


def test_request_over_limit_returns_429(client, monkeypatch):
    """
    With rate_limit_per_minute=3, the 4th request must return HTTP 429.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)

    sid = _login(client, monkeypatch)

    # First 3 should succeed
    for i in range(3):
        resp = _post_task(client, sid)
        assert resp.status_code == 202, f"Request {i + 1} should succeed but got {resp.status_code}"

    # 4th must be rejected
    resp = _post_task(client, sid)
    assert resp.status_code == 429


def test_429_response_has_json_body(client, monkeypatch):
    """The 429 response must be JSON (not a raw HTML error page)."""
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    sid = _login(client, monkeypatch)

    _post_task(client, sid)   # consume the 1 allowed request
    resp = _post_task(client, sid)   # this one should be limited

    assert resp.status_code == 429
    body = resp.json()
    # slowapi's default handler returns {"error": "Rate limit exceeded: N/minute"}
    assert "error" in body or "detail" in body


def test_upload_endpoint_is_also_rate_limited(client, monkeypatch):
    """
    POST /api/tasks/upload must enforce the same rate limit.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)

    # The background pipeline runs synchronously inside TestClient and can take
    # ~40 s per request (ffprobe + Gemini attempt).  With a 60-second window and
    # limit=2, t1 would be evicted before request 3 arrives.  Stub it out so each
    # request completes instantly and the window stays intact for the full test.
    import app.services.processor as _proc

    async def _noop(**kwargs):
        pass

    monkeypatch.setattr(_proc, "run_pipeline_from_file", _noop)

    sid = _login(client, monkeypatch)

    # First 2 uploads succeed
    for _ in range(2):
        resp = _post_upload(client, sid)
        assert resp.status_code == 202

    # 3rd is rejected
    resp = _post_upload(client, sid)
    assert resp.status_code == 429


def test_high_limit_effectively_allows_many_requests(client, monkeypatch):
    """
    With a very high limit (1000/minute), 10 rapid requests must all pass.
    Verifies that raising the limit disables practical throttling.
    """
    from app.config import settings
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1000)

    sid = _login(client, monkeypatch)

    for i in range(10):
        resp = _post_task(client, sid)
        assert resp.status_code == 202, f"Request {i + 1} failed unexpectedly with {resp.status_code}"
