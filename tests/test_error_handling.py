"""End-to-end tests for error reporting in the pipeline."""
import time
import pytest
from app.config import settings
from app.models import LessonResult
from app.services.errors import PipelineError

# ── Helpers (mirrored from test_upload_pipeline.py) ─────────────────────────────

def _login(client, mock_email):
    """Full magic-token login so upload requests carry a valid session cookie."""
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
def fake_result():
    return LessonResult(
        summary="סיכום בדיקה",
        chapters=[],
        quiz=[],
        language="he",
    )

@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point data/downloads at a temp dir so tests never dirty the repo."""
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)
    return data


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_download_failure_updates_task_with_hebrew_message(
    client, mock_email, monkeypatch, isolated_dirs
):
    """When the download fails, the task must end 'failed' with a Hebrew message."""
    from app.services import zoom_downloader

    async def boom_download(url, task_id, cookies_netscape=None, extract_to_mp3=True):
        raise RuntimeError("yt-dlp failed to download URL")

    monkeypatch.setattr(zoom_downloader, "download_audio", boom_download)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "failed"
    # The user-facing error should be localized (as per processor.py logic for unexpected errors)
    # or specific to the exception type if it's a PipelineError.
    # Based on current code: RuntimeError -> "שגיאה בלתי צפויה בעיבוד — נסה שוב"
    assert "שגיאה" in task["message"] or "failed" in task["message"].lower()
    assert "yt-dlp failed" in (task["error"] or "")


def test_pipeline_error_updates_task_with_custom_message(
    client, mock_email, monkeypatch, isolated_dirs
):
    """When a PipelineError is raised, the user-facing message should be preserved."""
    from app.services import zoom_downloader

    async def boom_download(url, task_id, cookies_netscape=None, extract_to_mp3=True):
        raise PipelineError("⚠️ URL לא תקין", detail="invalid url format")

    monkeypatch.setattr(zoom_downloader, "download_audio", boom_download)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "failed"
    # The error column should contain our custom user message.
    assert "⚠️ URL לא תקין" in (task["error"] or "")


def test_transcription_failure_updates_with_hebrew_message(
    client, mock_email, monkeypatch, isolated_dirs, fake_result
):
    """When transcription fails (e.g. Whisper crashes), the task must end 'failed'."""
    from app.services import transcriber
    from app.services import summarizer

    async def fake_transcribe(audio_path, language, task_id=None, **kwargs):
        raise RuntimeError("Whisper process crashed")

    async def fake_summarize_transcript(transcript, progress_cb):
        return fake_result

    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)
    monkeypatch.setattr(summarizer, "summarize_transcript", fake_summarize_transcript)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    assert task["status"] == "failed"
    assert "שגיאה" in (task["error"] or "")


def test_unexpected_exception_updates_async_with_generic_hebrew_message(
    client, mock_email, monkeypatch, isolated_dirs, fake_result
):
    """If an unexpected Exception occurs during summarization, the user sees a generic Hebrew error."""
    from app.services import summarizer

    async def boom_summarize(transcript, progress_cb):
        raise ValueError("Unexpected data format")

    monkeypatch.setattr(summarizer, "summarize_transcript", boom_summarize)

    _login(client, mock_email)
    task_id = _upload(client, "whisper_local")
    task = _wait_for_terminal(client, task_id)

    # Check the database directly for error_detail
    db_task = await state.get_task(task_id)
    assert db_task is not None
    assert "ValueError: Unexpected data format" in (db_task.error_detail or "")

    # The API response should still show the user-facing Hebrew message
    assert "שגיאה בלתי צפויה" in task["message"]
