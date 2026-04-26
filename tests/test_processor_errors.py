"""
Phase B tests for Task 1.5 — ProcessingError wired into processor.py.

Each stage wraps its native exceptions in a ProcessingError carrying the
right stage + code + user_message.
"""
import asyncio

import pytest


# ── _run_stage helper ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_stage_classifies_unknown_exception_with_stage():
    from app.errors import ProcessingError, ProcessingStage
    from app.services.processor import _run_stage

    async def boom():
        raise RuntimeError("totally unknown failure mode")

    with pytest.raises(ProcessingError) as exc_info:
        await _run_stage(ProcessingStage.TRANSCRIBE, boom())

    pe = exc_info.value
    assert pe.stage == ProcessingStage.TRANSCRIBE
    assert pe.code == "unknown"
    assert "totally unknown failure mode" in pe.technical_details


@pytest.mark.asyncio
async def test_run_stage_passes_processing_error_through_unchanged():
    from app.errors import ProcessingError, ProcessingStage
    from app.services.processor import _run_stage

    inner = ProcessingError(
        stage=ProcessingStage.DIARIZE,
        code="diarize_failed",
        user_message="diarize broke",
        technical_details="x",
    )

    async def raises():
        raise inner

    with pytest.raises(ProcessingError) as exc_info:
        await _run_stage(ProcessingStage.SUMMARIZE, raises())

    # The inner ProcessingError keeps its stage — not overwritten by the wrapper
    assert exc_info.value is inner
    assert exc_info.value.stage == ProcessingStage.DIARIZE


@pytest.mark.asyncio
async def test_run_stage_classifies_gemini_429_in_summarize_stage():
    from app.errors import ProcessingStage
    from app.services.processor import _run_stage

    async def rate_limited():
        raise RuntimeError("Gemini API HTTP 429: quota exceeded")

    with pytest.raises(Exception) as exc_info:
        await _run_stage(ProcessingStage.SUMMARIZE, rate_limited())

    pe = exc_info.value
    assert pe.code == "llm_rate_limit"
    assert pe.stage == ProcessingStage.SUMMARIZE


# ── End-to-end through run_pipeline_from_file ────────────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_from_file_persists_user_message_on_summarize_failure(
    monkeypatch, tmp_path
):
    """A 429 inside summarize → fail_task receives the Hebrew user_message."""
    from app.models import ProcessingMode
    from app.services import processor

    # Fake state functions — capture what gets persisted
    captured = {}

    async def fake_update_task(task_id, status, progress, message):
        pass

    async def fake_fail_task(task_id, error, *args, **kwargs):
        captured["error"] = error
        captured["args"] = args
        captured["kwargs"] = kwargs

    async def fake_complete_task(task_id, result):
        captured["completed"] = True

    monkeypatch.setattr(processor.state, "update_task", fake_update_task)
    monkeypatch.setattr(processor.state, "fail_task", fake_fail_task)
    monkeypatch.setattr(processor.state, "complete_task", fake_complete_task)
    monkeypatch.setattr(processor.state, "set_audio_path", lambda *a, **kw: asyncio.sleep(0))

    # Make summarize_audio fail with a Gemini 429
    async def boom_summarize(audio_path, progress_cb=None):
        raise RuntimeError("Gemini API HTTP 429: quota exceeded")

    monkeypatch.setattr(processor.summarizer, "summarize_audio", boom_summarize)

    # Skip flashcards (irrelevant once we fail) and audio cleanup
    async def fake_cleanup(p):
        pass
    monkeypatch.setattr(processor.zoom_downloader, "cleanup_audio", fake_cleanup)

    # Provide a fake audio file
    fake_audio = tmp_path / "fake.mp3"
    fake_audio.write_bytes(b"fake")

    await processor.run_pipeline_from_file(
        task_id="t1",
        file_path=str(fake_audio),
        mode=ProcessingMode.GEMINI_DIRECT,
        language="he",
    )

    assert "error" in captured
    assert "מכס" in captured["error"]  # Hebrew "quota"
    # complete_task was NOT called
    assert "completed" not in captured


@pytest.mark.asyncio
async def test_run_pipeline_from_file_classifies_unknown_failure(
    monkeypatch, tmp_path
):
    """Any totally-unknown failure still results in a Hebrew user_message."""
    from app.models import ProcessingMode
    from app.services import processor

    captured = {}

    async def fake_update(task_id, status, progress, message):
        pass

    async def fake_fail(task_id, error, *args, **kwargs):
        captured["error"] = error

    monkeypatch.setattr(processor.state, "update_task", fake_update)
    monkeypatch.setattr(processor.state, "fail_task", fake_fail)
    monkeypatch.setattr(processor.state, "complete_task", lambda *a, **k: asyncio.sleep(0))

    async def boom(audio_path, progress_cb=None):
        raise ValueError("some weird internal thing")

    monkeypatch.setattr(processor.summarizer, "summarize_audio", boom)
    monkeypatch.setattr(
        processor.zoom_downloader, "cleanup_audio",
        lambda p: asyncio.sleep(0),
    )

    fake_audio = tmp_path / "fake.mp3"
    fake_audio.write_bytes(b"fake")

    await processor.run_pipeline_from_file(
        task_id="t2",
        file_path=str(fake_audio),
        mode=ProcessingMode.GEMINI_DIRECT,
        language="he",
    )

    assert "some weird internal thing" in captured["error"] or "שגיאה" in captured["error"]
