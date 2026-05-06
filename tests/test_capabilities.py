"""Tests for GET /api/capabilities."""
import pytest


def test_capabilities_returns_gemini_defaults(client):
    """With default settings (no OpenAI key), reports Gemini + modes except whisper_api."""
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["llm_provider"] == "gemini"
    assert data["supports_audio_upload"] is True
    assert data["supports_streaming"] is True
    assert "gemini_direct" in data["available_modes"]
    assert "whisper_local" in data["available_modes"]
    assert "ivrit_ai" in data["available_modes"]
    # whisper_api requires OPENAI_API_KEY — absent in test env, so excluded


def test_capabilities_excludes_whisper_api_when_no_openai_key(client, monkeypatch):
    """whisper_api must be absent from available_modes when OPENAI_API_KEY is empty."""
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "")
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    assert "whisper_api" not in resp.json()["available_modes"]


def test_capabilities_includes_whisper_api_when_key_set(client, monkeypatch):
    """whisper_api appears in available_modes when OPENAI_API_KEY is non-empty."""
    from app.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-key-123")
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    assert "whisper_api" in resp.json()["available_modes"]


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
