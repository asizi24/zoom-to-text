"""
Tests for the Smart Summary LLM provider layer.

Ollama is exercised through httpx.MockTransport (no live server, no extra test
dependency); Gemini through a stubbed client; the factory against the real kv
store so backend selection is verified end to end.
"""
import asyncio
import json

import httpx
import pytest

from app import state
from app.config import settings
from app.services import runtime_config
from app.services.llm import GeminiProvider, OllamaProvider, get_summary_provider
from app.services.llm.base import LLMError


def _run(coro):
    try:
        return asyncio.run(coro)
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _run_with_db(tmp_path, monkeypatch, coro_factory):
    async def wrapper():
        monkeypatch.setattr(state, "DB_PATH", tmp_path / "llm.db")
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


# ── OllamaProvider ─────────────────────────────────────────────────────────────

def test_ollama_complete_returns_trimmed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] == "gemma2:9b"
        assert body["stream"] is False
        assert body["system"] == "sys"
        assert body["options"]["temperature"] == 0.1
        return httpx.Response(200, json={"response": "  שלום world  "})

    provider = OllamaProvider(
        "http://ollama:11434", "gemma2:9b", transport=httpx.MockTransport(handler)
    )
    assert _run(provider.complete("hi", system="sys", temperature=0.1)) == "שלום world"


def test_ollama_missing_model_raises_llmerror():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    provider = OllamaProvider(
        "http://ollama:11434", "nope", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError) as exc_info:
        _run(provider.complete("hi"))
    assert "nope" in exc_info.value.user_message


def test_ollama_transport_error_raises_llmerror():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        "http://ollama:11434", "gemma2:9b", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMError):
        _run(provider.complete("hi"))


def test_ollama_pull_streams_progress_dicts():
    ndjson = (
        json.dumps({"status": "pulling manifest"}) + "\n"
        + json.dumps({"status": "downloading", "total": 100, "completed": 50}) + "\n"
        + "\n"  # blank line must be skipped
        + json.dumps({"status": "success"}) + "\n"
    )

    def handler(request):
        assert request.url.path == "/api/pull"
        assert json.loads(request.content)["name"] == "gemma2:9b"
        return httpx.Response(200, content=ndjson.encode())

    provider = OllamaProvider(
        "http://ollama:11434", "gemma2:9b", transport=httpx.MockTransport(handler)
    )

    async def collect():
        return [event async for event in provider.pull()]

    events = _run(collect())
    assert events[0]["status"] == "pulling manifest"
    assert events[1]["completed"] == 50
    assert events[-1]["status"] == "success"


def test_ollama_list_models():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "gemma2:9b"}, {"name": "qwen2.5:7b"}]})

    provider = OllamaProvider(
        "http://ollama:11434", "gemma2:9b", transport=httpx.MockTransport(handler)
    )
    assert _run(provider.list_models()) == ["gemma2:9b", "qwen2.5:7b"]


def test_ollama_is_available_true_and_false():
    up = OllamaProvider(
        "http://ollama:11434", "m",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "0.1"})),
    )
    assert _run(up.is_available()) is True

    def down(request):
        raise httpx.ConnectError("no server")

    provider = OllamaProvider("http://ollama:11434", "m", transport=httpx.MockTransport(down))
    assert _run(provider.is_available()) is False


# ── GeminiProvider ─────────────────────────────────────────────────────────────

def test_gemini_complete(monkeypatch):
    import app.services.llm.gemini_provider as gp

    class _FakeModels:
        async def generate_content(self, model, contents, config):
            assert model == "gemini-2.5-flash"
            assert config.system_instruction == "s"
            return type("R", (), {"text": " done "})()

    fake_client = type("C", (), {"aio": type("A", (), {"models": _FakeModels()})()})()
    monkeypatch.setattr(gp, "_get_client", lambda key: fake_client)

    provider = GeminiProvider(api_key="k", model="gemini-2.5-flash")
    assert _run(provider.complete("hi", system="s")) == "done"


# ── Factory (backend selection via kv store) ───────────────────────────────────

def test_factory_defaults_to_ollama(tmp_path, monkeypatch):
    async def scenario():
        provider = await get_summary_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.model == settings.ollama_model

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_factory_selects_gemini_when_configured(tmp_path, monkeypatch):
    async def scenario():
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "gemini")
        await runtime_config.set(runtime_config.KEY_GEMINI_API_KEY, "test-key")
        provider = await get_summary_provider()
        assert isinstance(provider, GeminiProvider)
        assert provider.api_key == "test-key"

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_factory_honors_ollama_model_override(tmp_path, monkeypatch):
    async def scenario():
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "ollama")
        await runtime_config.set(runtime_config.KEY_OLLAMA_MODEL, "qwen2.5:7b")
        provider = await get_summary_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen2.5:7b"

    _run_with_db(tmp_path, monkeypatch, scenario)


def test_factory_gemini_without_any_key_raises(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings, "google_api_key", "", raising=False)
        monkeypatch.setattr(settings, "google_application_credentials", "nonexistent.json", raising=False)
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "gemini")
        with pytest.raises(LLMError):
            await get_summary_provider()

    _run_with_db(tmp_path, monkeypatch, scenario)
