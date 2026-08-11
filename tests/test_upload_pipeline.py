"""End-to-end tests for the local file upload flow.

Covers the exact chain the UI exercises:
  POST /api/tasks/upload → payload persisted in SQLite → worker queue →
  processor.run_pipeline_from_file → transcriber/summarizer → completed task.

Only the heavy externals (Gemini, Whisper, ffmpeg) are mocked — routing,
auth, the worker loop, SQLite persistence and audio persistence are real.
"""
import time

import pytest

from app.config import settings
from app.models import LessonResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _login(client, mock_email):
    """Full magic-link login so upload requests carry a valid session cookie.

    The cookie is set with Secure for any non-localhost base_url; httpx drops
    Secure cookies over plain http://testserver, so we carry it explicitly.
    """
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


def _upload(client, mode: str, filename: str = "lecture.mp4"):
    resp = client.post(
        "/api/tasks/upload",
        files={"file": (filename, b"fake media bytes", "video/mp4")},
        data={"mode": mode, "language": "he"},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _wait_for_terminal(client, task_id: str, timeout: float = 15.0) -> dict:
    """Poll like the frontend does until the task completes or fails."""
    task = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200, resp.text
        task = resp.json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    pytest.fail(f"task {task_id} never reached a terminal status: {task}")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point data/downloads at a temp dir so tests never dirty the repo."""
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)


@pytest.fixture
def fake_result():
    return LessonResult(
        summary="סיכום בדיקה",
        chapters=[],
        quiz=[],
        language="he",
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_upload_gemini_direct_completes(
    client, mock_email, monkeypatch, isolated_dirs, fake_result
):
    """GEMINI_DIRECT upload: no extraction, straight to summarize_audio."""
    import app.services.summarizer as summarizer

    async def fake_summarize_audio(audio_path, progress_cb, language="he"):
        return fake_result

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", fake_summarize_audio)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    _login(client, mock_email)
    task_id = _upload(client, "gemini_direct")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "completed", task
    assert task["result"]["summary"] == "סיכום בדיקה"


def test_upload_transcription_only_completes(
    client, mock_email, monkeypatch, isolated_dirs
):
    """TRANSCRIPTION_ONLY upload: extraction attempted (falls back gracefully
    without ffmpeg), local transcription, and NO Gemini call at all."""
    import app.services.summarizer as summarizer
    import app.services.transcriber as transcriber

    async def fake_ivrit(audio_path, language, task_id=None):
        return "תמלול לדוגמה", "he"

    async def gemini_must_not_run(*args, **kwargs):
        raise AssertionError("Gemini must not be called in TRANSCRIPTION_ONLY")

    monkeypatch.setattr(transcriber, "transcribe_ivrit_ai", fake_ivrit)
    monkeypatch.setattr(summarizer, "summarize_transcript", gemini_must_not_run)
    monkeypatch.setattr(summarizer, "generate_flashcards", gemini_must_not_run)

    _login(client, mock_email)
    task_id = _upload(client, "transcription_only")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "completed", task
    assert task["result"]["transcript"] == "תמלול לדוגמה"
    assert task["result"]["summary"] == ""


def test_upload_whisper_local_completes(
    client, mock_email, monkeypatch, isolated_dirs, fake_result
):
    """WHISPER_LOCAL upload: extraction → transcribe → summarize_transcript."""
    import app.services.summarizer as summarizer
    import app.services.transcriber as transcriber

    async def fake_transcribe(audio_path, language, task_id=None, **kwargs):
        return "טקסט מתומלל", "he"

    async def fake_summarize_transcript(transcript, progress_cb):
        return fake_result

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)
    monkeypatch.setattr(summarizer, "summarize_transcript", fake_summarize_transcript)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "completed", task
    assert task["result"]["transcript"] == "טקסט מתומלל"


def test_upload_failure_is_reported_not_silent(
    client, mock_email, monkeypatch, isolated_dirs
):
    """When the pipeline blows up, the task must end 'failed' with a message —
    never stuck in pending/transcribing forever."""
    import app.services.transcriber as transcriber

    async def boom(audio_path, language, task_id=None):
        raise RuntimeError("simulated transcription crash")

    monkeypatch.setattr(transcriber, "transcribe", boom)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "failed", task
    assert task["error"]


def test_upload_payload_repointed_after_extraction(
    client, mock_email, monkeypatch, isolated_dirs, fake_result
):
    """After extraction deletes the original upload, the durable job payload
    must point at the extracted MP3 — otherwise a server restart mid-
    transcription sees a missing file and wrongly fails the resumed task."""
    from pathlib import Path

    import app.services.audio_preprocessor as ap
    import app.services.summarizer as summarizer
    import app.services.transcriber as transcriber
    from app import state as state_module

    async def fake_extract(src_path, delete_source=True):
        dest = Path(src_path).with_suffix(".mp3")
        dest.write_bytes(b"fake mp3")
        Path(src_path).unlink()
        return str(dest)

    payloads = []
    real_set = state_module.set_job_payload

    async def spy_set(task_id, payload):
        payloads.append(dict(payload))
        await real_set(task_id, payload)

    async def fake_transcribe(audio_path, language, task_id=None, **kwargs):
        return "טקסט", "he"

    async def fake_summarize_transcript(transcript, progress_cb):
        return fake_result

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(ap, "extract_audio_track", fake_extract)
    monkeypatch.setattr(state_module, "set_job_payload", spy_set)
    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)
    monkeypatch.setattr(summarizer, "summarize_transcript", fake_summarize_transcript)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local", filename="lecture.mp4")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "completed", task
    assert payloads[0]["file_path"].endswith(".mp4")   # as uploaded
    assert payloads[-1]["file_path"].endswith(".mp3")  # repointed post-extraction


def test_upload_rejects_unknown_extension(client, mock_email, isolated_dirs):
    resp_login = _login(client, mock_email)
    resp = client.post(
        "/api/tasks/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"mode": "gemini_direct", "language": "he"},
    )
    assert resp.status_code == 400
