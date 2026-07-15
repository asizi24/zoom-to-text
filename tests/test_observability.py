"""
Tests for Phase 1 of the production-readiness work:

  - Structured logging: JSON/text formatters + contextvar correlation ids
    (app/logging_config.py)
  - Standard API error envelope: {"detail", "code", "request_id"} on every
    error response (app/api/errors.py)
  - X-Request-ID binding: minted or honored per request, echoed on the
    response, and matching the envelope's request_id
  - /health (liveness) vs /ready (dependency checks) split
"""
import json
import logging

from app.logging_config import (
    JsonFormatter,
    TextFormatter,
    request_id_var,
    task_id_var,
)


# ── Formatter unit tests ──────────────────────────────────────────────────────


def _record(msg: str = "hello", exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )


def test_json_formatter_emits_valid_json_with_core_fields():
    line = JsonFormatter().format(_record("processing started"))
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["msg"] == "processing started"
    assert "ts" in payload
    # No ids bound → the keys are omitted, not null
    assert "request_id" not in payload
    assert "task_id" not in payload


def test_json_formatter_preserves_hebrew():
    line = JsonFormatter().format(_record("מתמלל עם Whisper"))
    assert "מתמלל" in line  # ensure_ascii=False — no \uXXXX soup
    assert json.loads(line)["msg"] == "מתמלל עם Whisper"


def test_json_formatter_includes_bound_correlation_ids():
    rid_token = request_id_var.set("req-abc")
    tid_token = task_id_var.set("task-123")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(rid_token)
        task_id_var.reset(tid_token)
    assert payload["request_id"] == "req-abc"
    assert payload["task_id"] == "task-123"


def test_json_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("it failed", exc_info=sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_text_formatter_tags_context_only_when_bound():
    fmt = TextFormatter()
    assert "[req=" not in fmt.format(_record())

    token = task_id_var.set("task-9")
    try:
        line = fmt.format(_record())
    finally:
        task_id_var.reset(token)
    assert "[task=task-9]" in line


# ── Error envelope (full stack via TestClient) ────────────────────────────────


def test_401_uses_envelope(client):
    r = client.get("/api/tasks")
    assert r.status_code == 401
    body = r.json()
    assert body["detail"] == "Not authenticated"
    assert body["code"] == "unauthorized"
    assert "request_id" in body


def test_404_unknown_route_uses_envelope(client):
    r = client.get("/api/no-such-route")
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "not_found"
    assert isinstance(body["detail"], str)


def test_422_validation_error_has_string_detail(client):
    r = client.post("/api/auth/request", json={})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    # FastAPI's default puts a list in detail; the envelope flattens it so the
    # frontend can always render body.detail directly.
    assert isinstance(body["detail"], str)
    assert "email" in body["detail"]
    assert isinstance(body["errors"], list)


# ── Request-id header binding ─────────────────────────────────────────────────


def test_response_carries_generated_request_id(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert len(r.headers["x-request-id"]) >= 8


def test_inbound_request_id_is_honored_and_matches_envelope(client):
    r = client.get("/api/tasks", headers={"X-Request-ID": "proxy-id-42"})
    assert r.headers["x-request-id"] == "proxy-id-42"
    assert r.json()["request_id"] == "proxy-id-42"  # header ↔ envelope correlation


def test_inbound_request_id_is_sanitized(client):
    r = client.get("/health", headers={"X-Request-ID": "abc!!123<evil>\n"})
    assert r.headers["x-request-id"] == "abc123evil"


# ── Liveness vs readiness ─────────────────────────────────────────────────────


def test_health_reports_version(client):
    body = client.get("/health").json()
    from app import __version__

    assert body == {"status": "ok", "version": __version__}


def test_ready_ok_when_all_dependencies_up(client, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ffmpeg")
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    # TestClient lifespan ran: DB initialized, worker pool started
    assert body["checks"] == {"database": True, "worker": True, "ffmpeg": True}


def test_ready_degraded_when_ffmpeg_missing(client, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["ffmpeg"] is False
    assert body["checks"]["database"] is True
