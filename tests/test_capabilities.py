"""Tests for GET /api/capabilities."""
import pytest


def test_capabilities_returns_gemini_defaults(client):
    """With default settings, the endpoint reports Gemini + all four modes."""
    # The /api/capabilities endpoint is public (no auth) so the UI can read
    # provider info before the user logs in. Test does not need the auth
    # header that the protected endpoints require.
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "gemini"
    assert data["supports_audio_upload"] is True
    assert data["supports_streaming"] is True
    assert "gemini_direct" in data["available_modes"]
    assert "whisper_local" in data["available_modes"]
    assert "whisper_api" in data["available_modes"]
    assert "ivrit_ai" in data["available_modes"]


def test_capabilities_hides_gemini_direct_for_openrouter(client, monkeypatch):
    """When provider can't upload audio, gemini_direct mode is excluded."""
    from app.config import settings
    from app.services.llm_providers import _reset_provider_cache

    monkeypatch.setattr(settings, "llm_provider", "openrouter", raising=False)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test", raising=False)
    _reset_provider_cache()

    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "openrouter"
    assert data["supports_audio_upload"] is False
    assert "gemini_direct" not in data["available_modes"]
    assert "whisper_local" in data["available_modes"]
