"""
Regression tests for the architecture / async / DevOps hardening pass.

These lock in four fixes that are easy to silently regress because they only
misbehave under failure or disconnect conditions a happy-path test never hits:

  * Issue 2 — Fail-fast on startup when LLM_PROVIDER=gemini but no Google creds.
  * Issue 5 — Uploaded file is deleted from disk if the DB insert fails (no orphan).
  * Issue 6 — Chat model reply is persisted via a post-stream BackgroundTask,
              not inside the generator's control flow.
  * Issue 8 — cors_origin parses a comma-separated env string into a list[str].

The other four issues (1 docker volume path, 3 to_thread file moves, 4 to_thread
exports, 7 awaited watcher cancellation) are covered structurally elsewhere or
are pure config; see the review notes for why they need no new test here.
"""
import io

import pytest
from unittest.mock import patch

from app import state
from app.api import deps
from app.config import Settings, settings
from app.main import app as fastapi_app
from app.models import LessonResult


# ── Issue 8: CORS origin parsing ───────────────────────────────────────────────

class TestCorsOriginParsing:
    """cors_origin is declared as a str but must always normalize to list[str]."""

    def test_comma_separated_string_becomes_list(self):
        s = Settings(_env_file=None, cors_origin="https://a.example.com, https://b.example.com")
        assert s.cors_origin == ["https://a.example.com", "https://b.example.com"]

    def test_single_origin_becomes_one_item_list(self):
        s = Settings(_env_file=None, cors_origin="https://only.example.com")
        assert s.cors_origin == ["https://only.example.com"]

    def test_empty_string_becomes_empty_list(self):
        s = Settings(_env_file=None, cors_origin="")
        assert s.cors_origin == []

    def test_blank_segments_are_dropped(self):
        s = Settings(_env_file=None, cors_origin="https://a, ,https://b,")
        assert s.cors_origin == ["https://a", "https://b"]


# ── Issue 2: Fail-fast on startup ──────────────────────────────────────────────

def test_lifespan_fails_fast_when_gemini_provider_missing_credentials(tmp_path, monkeypatch):
    """LLM_PROVIDER=gemini + no API key + no creds file ⇒ startup raises RuntimeError.

    Masking this as a warning lets the container boot 'healthy' and then 500 on
    the first real request; fail-fast makes the orchestrator restart-loop + alert.
    """
    from fastapi.testclient import TestClient
    import app.state as state_module

    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "ff.db")
    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)
    monkeypatch.setattr(settings, "google_api_key", "", raising=False)
    # Point at a path that does NOT exist so the creds-file branch is skipped
    # (the repo ships a real key.json at the default path).
    monkeypatch.setattr(
        settings, "google_application_credentials", str(tmp_path / "nope.json"), raising=False
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(RuntimeError, match="requires them"):
        with TestClient(fastapi_app):
            pass


def test_lifespan_boots_when_non_google_provider_missing_credentials(tmp_path, monkeypatch):
    """ollama needs no Google creds — startup must NOT fail (offline deploy)."""
    from fastapi.testclient import TestClient
    import app.state as state_module

    monkeypatch.setattr(state_module, "DB_PATH", tmp_path / "ok.db")
    monkeypatch.setattr(settings, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(settings, "google_api_key", "", raising=False)
    monkeypatch.setattr(
        settings, "google_application_credentials", str(tmp_path / "nope.json"), raising=False
    )
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    # Entering and exiting the context must not raise.
    with TestClient(fastapi_app):
        pass


# ── Issue 5: No orphaned upload when the DB insert fails ────────────────────────

@pytest.fixture
def authed_client(client):
    fastapi_app.dependency_overrides[deps.get_current_user] = lambda: "test-user"
    yield client
    fastapi_app.dependency_overrides.pop(deps.get_current_user, None)


def test_upload_db_failure_removes_orphaned_file(authed_client, tmp_path, monkeypatch):
    """If state.create_task raises after the file is streamed, the file is unlinked."""
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    monkeypatch.setattr(settings, "downloads_dir", downloads, raising=False)

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB outage")

    monkeypatch.setattr("app.api.routes.state.create_task", _boom)

    # conftest's client uses raise_server_exceptions=True, so the re-raised
    # RuntimeError surfaces here — but only AFTER the route's cleanup runs.
    with pytest.raises(RuntimeError, match="simulated DB outage"):
        authed_client.post(
            "/api/tasks/upload",
            files=[("file", ("recording.mp3", io.BytesIO(b"x" * 8192), "audio/mpeg"))],
        )

    # The streamed upload must not be left behind on the volume.
    assert list(downloads.iterdir()) == []


# ── Issue 6: Chat reply persisted via post-stream BackgroundTask ────────────────

async def test_chat_persists_model_reply_after_stream(client):
    """The assembled model reply is saved once the SSE stream closes.

    The save lives in a Starlette BackgroundTask (not the generator's finally),
    so it survives a mid-stream client disconnect without corrupting history.
    Here we assert the happy-path wiring: after the stream, history holds both
    the user question and the full model reply.
    """
    await state.create_task("chat-hard-1", "https://x/chat", user_id="chat-user")
    await state.complete_task("chat-hard-1", LessonResult(summary="נושא ההרצאה"))

    async def _fake_stream(context, history, question):
        for chunk in ["שלום ", "עולם"]:
            yield chunk

    fastapi_app.dependency_overrides[deps.get_current_user] = lambda: "chat-user"
    try:
        with patch("app.services.summarizer.stream_chat_response", _fake_stream):
            r = client.post("/api/tasks/chat-hard-1/chat", json={"question": "מה קורה?"})
            assert r.status_code == 200
            assert "שלום" in r.text

        history = await state.get_chat_history("chat-hard-1")
        roles = [m["role"] for m in history]
        assert "user" in roles, history
        model_replies = [m["content"] for m in history if m["role"] == "model"]
        assert model_replies == ["שלום עולם"], history
    finally:
        fastapi_app.dependency_overrides.pop(deps.get_current_user, None)


# ── Phase 2 hardening ──────────────────────────────────────────────────────────

# Issue 10: Task IDOR — get_task_for_user must not serve unowned (NULL) rows.

async def test_get_task_for_user_rejects_null_owned_task(client):
    """A NULL-owned (legacy/unowned) task is readable by NOBODY via this path.

    Previously `(user_id=? OR user_id IS NULL)` let any authenticated user read
    every unowned task by guessing its id — a classic IDOR. Ownership is now
    strict.
    """
    await state.create_task("idor-null", "https://x/idor", user_id=None)
    assert await state.get_task_for_user("idor-null", "any-user") is None

    # A properly-owned task is still served to its owner — and only its owner.
    await state.create_task("idor-owned", "https://x/idor2", user_id="owner-1")
    owned = await state.get_task_for_user("idor-owned", "owner-1")
    assert owned is not None and owned.task_id == "idor-owned"
    assert await state.get_task_for_user("idor-owned", "intruder") is None


# Issue 11: magic tokens are stored hashed, never in plaintext.

async def test_magic_token_is_stored_hashed(client):
    """Only the SHA-256 digest hits the DB; the round-trip still authenticates."""
    import hashlib

    user_id = await state.get_or_create_user("hash-test@example.com")
    token = await state.create_magic_token(user_id)

    db = await state._get_db()
    async with db.execute(
        "SELECT token FROM magic_tokens WHERE user_id=?", [user_id]
    ) as cur:
        row = await cur.fetchone()
    stored = row["token"]

    assert stored != token, "plaintext token must not be persisted"
    assert stored == hashlib.sha256(token.encode("utf-8")).hexdigest()

    # The emailed plaintext still consumes correctly, exactly once.
    assert await state.consume_magic_token(token) == user_id
    assert await state.consume_magic_token(token) is None


# Issue 4: oversized uploads are rejected up front, before any disk write.

def test_upload_rejects_oversized_content_length(authed_client, tmp_path, monkeypatch):
    """A body past max_upload_bytes returns 413 and writes nothing to the volume."""
    downloads = tmp_path / "dl"
    downloads.mkdir()
    monkeypatch.setattr(settings, "downloads_dir", downloads, raising=False)
    monkeypatch.setattr(settings, "max_upload_bytes", 1024, raising=False)

    r = authed_client.post(
        "/api/tasks/upload",
        files=[("file", ("recording.mp3", io.BytesIO(b"x" * 5000), "audio/mpeg"))],
    )
    assert r.status_code == 413
    assert list(downloads.iterdir()) == []  # fail-fast: nothing streamed to disk
