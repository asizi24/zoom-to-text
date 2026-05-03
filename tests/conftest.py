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


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated temp database."""
    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(state_module, "_db", None, raising=False)  # added in Task 3
    monkeypatch.setattr(settings, "allowed_emails", "allowed@example.com", raising=False)  # added in Task 2
    monkeypatch.setattr(settings, "resend_api_key", "test_key", raising=False)  # added in Task 2
    monkeypatch.setattr(settings, "base_url", "http://testserver")
    monkeypatch.setattr(settings, "cors_origin", "http://testserver", raising=False)  # added in Task 2

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


@pytest.fixture
def lti_env(client, tmp_path, monkeypatch):
    """
    LTI test isolation. Layered on top of `client` so the test DB is already
    initialised (the lti_oidc_state table is created in app.state.init_db()).

    Redirects:
      * keys module → tmp_path/lti_keys/{private,public}.pem
      * config module → tmp_path/lti_platforms.json
      * oidc module → fresh JWKS cache

    Clears each module's singleton caches so prior tests don't leak state.

    Returns a small env object with .write_platforms(records) so the test
    can drop platform records into the JSON file and force a config reload.
    """
    import json as _json
    from app.services.lti import config as lti_config
    from app.services.lti import keys as lti_keys
    from app.services.lti import oidc as lti_oidc

    keys_dir = tmp_path / "lti_keys"
    monkeypatch.setattr(lti_keys, "KEYS_DIR", keys_dir)
    monkeypatch.setattr(lti_keys, "PRIVATE_PATH", keys_dir / "private.pem")
    monkeypatch.setattr(lti_keys, "PUBLIC_PATH", keys_dir / "public.pem")
    monkeypatch.setattr(lti_keys, "_private_key", None)
    monkeypatch.setattr(lti_keys, "_public_key", None)

    platforms_file = tmp_path / "lti_platforms.json"
    monkeypatch.setattr(lti_config, "PLATFORMS_FILE", platforms_file)
    monkeypatch.setattr(lti_config, "_platforms", None)

    monkeypatch.setattr(lti_oidc, "_jwks_cache", {})

    class _LtiEnv:
        def __init__(self):
            self.platforms_file = platforms_file
            self.keys_dir = keys_dir

        def write_platforms(self, records: list[dict]) -> None:
            self.platforms_file.write_text(
                _json.dumps(records), encoding="utf-8"
            )
            lti_config.reset_cache()

    return _LtiEnv()
