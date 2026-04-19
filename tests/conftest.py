"""
Shared fixtures for all tests.

Key design decisions:
- Each test gets a fresh SQLite DB in a temp directory.
- TestClient triggers the app lifespan (startup/shutdown), which calls init_db()
  and close_db(), so the DB is always in a clean state between tests.
- Resend HTTP calls are patched to avoid real network calls.
"""
import pytest
import app.state as state_module
from app.config import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated temp database."""
    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(state_module, "_db", None)
    monkeypatch.setattr(settings, "allowed_emails", "allowed@example.com")
    monkeypatch.setattr(settings, "resend_api_key", "test_key")
    monkeypatch.setattr(settings, "base_url", "http://testserver")
    monkeypatch.setattr(settings, "cors_origin", "http://testserver")

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_email(monkeypatch):
    """Capture sent magic links instead of calling Resend."""
    sent = []

    async def fake_send(email: str, token: str) -> None:
        sent.append({"email": email, "token": token})

    import app.api.auth as auth_module
    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    return sent
