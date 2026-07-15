"""
Shared fixtures for all tests.

Key design decisions:
- Each test gets a fresh SQLite DB in a temp directory.
- TestClient triggers the app lifespan (startup/shutdown), which calls init_db()
  and close_db(), so the DB is always in a clean state between tests.
- Resend HTTP calls are patched to avoid real network calls.
- raising=False on setattr calls that target attributes added in later tasks,
  so the fixture doesn't crash if run before those tasks are complete.
"""
import pytest
import app.state as state_module
from app.config import settings


def _isolate(tmp_path, monkeypatch):
    """Common per-test isolation: fresh DB + deterministic settings."""
    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(state_module, "_db", None, raising=False)  # added in Task 3
    monkeypatch.setattr(settings, "allowed_emails", "allowed@example.com", raising=False)  # added in Task 2
    monkeypatch.setattr(settings, "resend_api_key", "test_key", raising=False)  # added in Task 2
    monkeypatch.setattr(settings, "base_url", "http://testserver")
    monkeypatch.setattr(settings, "cors_origin", "http://testserver", raising=False)  # added in Task 2
    # Deterministic regardless of the host machine's env: dev mode (prod mode
    # refuses to start without a real Resend key) and no rate limiting (many
    # tests log in / submit rapidly). The rate-limit tests re-enable it.
    monkeypatch.setattr(settings, "environment", "development", raising=False)
    monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated temp database.

    Setup is forced 'complete' so GET / behaves as it always did (login gate,
    not the first-boot wizard). Wizard-flow tests use `wizard_client` instead.
    """
    _isolate(tmp_path, monkeypatch)

    import app.services.runtime_config as runtime_config

    async def _already_complete() -> bool:
        return True

    monkeypatch.setattr(runtime_config, "is_setup_complete", _already_complete)

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def wizard_client(tmp_path, monkeypatch):
    """TestClient for the first-boot Setup Wizard: setup_complete starts false
    (the real kv flag on a fresh DB), so GET / serves the wizard and the
    /api/setup/* endpoints are active until POST /api/setup/complete."""
    _isolate(tmp_path, monkeypatch)

    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def mock_email(monkeypatch):
    """Capture sent magic links instead of calling Resend. Requires Task 5 (auth.py) to exist."""
    sent = []

    async def fake_send(email: str, token: str) -> None:
        sent.append({"email": email, "token": token})

    import app.api.auth as auth_module
    monkeypatch.setattr(auth_module, "_send_magic_link_email", fake_send)
    return sent
