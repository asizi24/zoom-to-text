"""
Tests for Task 2.3 — WebSocket streaming transcription endpoint.

Gate: ENABLE_STREAMING must be True to accept connections.
Code-only; runs on a home server with enough RAM for Faster-Whisper.

Protocol:
  Client → Server:
    1. Optional JSON {"type": "config", "language": "he"}
    2. Binary frames: raw audio bytes (any number of chunks)
    3. JSON {"type": "done"}
  Server → Client:
    - JSON {"type": "done", "transcript": "..."} on success
    - JSON {"type": "error", "message": "..."} on failure
"""
import json
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketDisconnect

from app import state
from app.config import settings


# ── Shared helpers ─────────────────────────────────────────────────────────────

AUTH_HEADERS = {"Cookie": "session_id=fake-session-id"}


@pytest.fixture
def streaming_client(client, monkeypatch):
    """TestClient with streaming enabled and a mocked authenticated session."""
    monkeypatch.setattr(settings, "enable_streaming", True, raising=False)
    monkeypatch.setattr(state, "get_session_user", AsyncMock(return_value="user-123"))
    return client


# ── Gate tests ─────────────────────────────────────────────────────────────────

def test_enable_streaming_defaults_to_false():
    """Feature gate must be off by default — never on in production."""
    assert settings.enable_streaming is False


def test_streaming_gate_closes_when_disabled(client):
    """With ENABLE_STREAMING=False the WS upgrade is rejected (code 1008)."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
            ws.receive_text()  # triggers close detection after server sends close(1008)
    assert exc_info.value.code == 1008


# ── Auth tests ─────────────────────────────────────────────────────────────────

def test_streaming_rejects_missing_session(client, monkeypatch):
    """Enabled but no session cookie → close with code 3401."""
    monkeypatch.setattr(settings, "enable_streaming", True, raising=False)
    # No Cookie header → session_id=None → get_session_user short-circuits to None
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.receive_text()  # triggers close detection after server sends close(3401)
    assert exc_info.value.code == 3401


def test_streaming_rejects_invalid_session(client, monkeypatch):
    """Enabled but session resolves to None (expired) → close with code 3401."""
    monkeypatch.setattr(settings, "enable_streaming", True, raising=False)
    monkeypatch.setattr(state, "get_session_user", AsyncMock(return_value=None))
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
            ws.receive_text()  # triggers close detection after server sends close(3401)
    assert exc_info.value.code == 3401


# ── Protocol tests ─────────────────────────────────────────────────────────────

def test_streaming_empty_done_returns_empty_transcript(streaming_client):
    """Sending 'done' without any audio chunks → {"type": "done", "transcript": ""}."""
    with streaming_client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
        ws.send_text(json.dumps({"type": "done"}))
        msg = ws.receive_json()
    assert msg == {"type": "done", "transcript": ""}


def test_streaming_config_message_sets_language(streaming_client, monkeypatch):
    """Config message accepted; its language is passed on to the transcriber."""
    calls = []

    async def fake_transcribe(path, lang, task_id=None):
        calls.append(lang)
        return "שלום", "he"

    from app.services import transcriber
    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)

    with streaming_client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
        ws.send_text(json.dumps({"type": "config", "language": "he"}))
        ws.send_bytes(b"audio-bytes")
        ws.send_text(json.dumps({"type": "done"}))
        msg = ws.receive_json()

    assert msg == {"type": "done", "transcript": "שלום"}
    assert calls == ["he"]


def test_streaming_binary_chunks_combined(streaming_client, monkeypatch):
    """Multiple binary chunks are concatenated before being passed to the transcriber."""
    received_size = []

    async def fake_transcribe(path, lang, task_id=None):
        from pathlib import Path
        received_size.append(Path(path).stat().st_size)
        return "hello", "en"

    from app.services import transcriber
    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)

    chunk_a = b"aaaa"
    chunk_b = b"bbbb"
    with streaming_client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
        ws.send_bytes(chunk_a)
        ws.send_bytes(chunk_b)
        ws.send_text(json.dumps({"type": "done"}))
        msg = ws.receive_json()

    assert msg == {"type": "done", "transcript": "hello"}
    assert received_size == [len(chunk_a) + len(chunk_b)]


def test_streaming_transcription_error_returns_error_message(streaming_client, monkeypatch):
    """If transcriber raises, the client receives {"type": "error", "message": "..."}."""
    async def failing_transcribe(path, lang, task_id=None):
        raise RuntimeError("model exploded")

    from app.services import transcriber
    monkeypatch.setattr(transcriber, "transcribe", failing_transcribe)

    with streaming_client.websocket_connect("/ws/transcribe", headers=AUTH_HEADERS) as ws:
        ws.send_bytes(b"audio")
        ws.send_text(json.dumps({"type": "done"}))
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert "model exploded" in msg["message"]
