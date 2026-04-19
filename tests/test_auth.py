"""Tests for Magic Link authentication flow."""
from app.config import settings


def test_config_has_auth_fields():
    """Settings must expose the three new auth fields."""
    assert hasattr(settings, "allowed_emails")
    assert hasattr(settings, "resend_api_key")
    assert hasattr(settings, "cors_origin")


import pytest


def test_magic_token_full_flow(client, mock_email):
    """Request a magic link, extract token, verify it — should create session cookie."""
    resp = client.post("/api/auth/request", json={"email": "allowed@example.com"})
    assert resp.status_code == 200
    assert len(mock_email) == 1
    token = mock_email[0]["token"]
    assert token

    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    assert "session_id" in resp.cookies


def test_invalid_token_returns_400(client, monkeypatch):
    """verify endpoint returns 400 when token is invalid or expired."""
    import app.state as state_module

    async def fake_consume(token: str):
        return None

    monkeypatch.setattr(state_module, "consume_magic_token", fake_consume)
    resp = client.get("/api/auth/verify?token=any-token", follow_redirects=False)
    assert resp.status_code == 400


def test_magic_token_used_twice_rejected(client, mock_email):
    """A used token cannot be used again."""
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]

    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302

    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 400


def test_unknown_email_returns_generic_message(client, mock_email):
    """Unknown emails get the same response as known ones (no enumeration)."""
    resp = client.post("/api/auth/request", json={"email": "hacker@evil.com"})
    assert resp.status_code == 200
    assert len(mock_email) == 0
