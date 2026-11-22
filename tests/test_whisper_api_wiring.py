"""
Wiring test for WHISPER_API mode.

Regression: processor passed resume_offset= to transcriber.transcribe_via_api,
which doesn't accept it, so every WHISPER_API task failed with TypeError.
The fake below copies the real function's signature exactly (no **kwargs),
so a call the real function would reject is rejected here too.
"""
import inspect

import pytest

from app import state
from app.models import LessonResult, ProcessingMode
from app.services import processor, summarizer, transcriber


@pytest.mark.asyncio
async def test_processor_calls_whisper_api_with_its_real_signature(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "DB_PATH", tmp_path / "whisper_api_test.db")
    monkeypatch.setattr(state, "_db", None, raising=False)
    await state.init_db()
    await state.create_task("t-api", url="", user_id=None)

    real_sig = inspect.signature(transcriber.transcribe_via_api)
    calls = []

    async def fake_whisper_api(*args, **kwargs):
        real_sig.bind(*args, **kwargs)  # TypeError if the real one would reject it
        calls.append(kwargs)
        return "[00:00] api transcript", "he"

    async def fake_summarize(transcript, **kwargs):
        return LessonResult(summary="ok")

    monkeypatch.setattr(transcriber, "transcribe_via_api", fake_whisper_api)
    monkeypatch.setattr(summarizer, "summarize_transcript_with_ollama", fake_summarize)

    try:
        result = await processor._process_audio(
            task_id="t-api",
            audio_path=str(tmp_path / "fake.mp3"),
            mode=ProcessingMode.WHISPER_API,
            language="he",
        )
    finally:
        await state.close_db()

    assert len(calls) == 1
    assert result.transcript == "[00:00] api transcript"
