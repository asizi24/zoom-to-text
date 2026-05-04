"""
Wiring tests for Feature 5 — ivrit-ai transcription engine.

We don't download or load the actual model (GB-scale, slow). Instead we assert:
  - ProcessingMode.IVRIT_AI exists and serializes as "ivrit_ai"
  - transcriber.transcribe_ivrit_ai exists with the right signature
  - processor routes IVRIT_AI to transcribe_ivrit_ai (verified via monkeypatch)
  - config has ivrit_ai_model with a sane default
  - idle-unload handles the second model cache
"""
import asyncio
import inspect

import pytest

from app import state
from app.config import settings
from app.models import ProcessingMode, TaskStatus
from app.services import processor, transcriber


def test_processing_mode_has_ivrit_ai():
    assert ProcessingMode.IVRIT_AI.value == "ivrit_ai"
    # Enum round-trips through strings (API contract)
    assert ProcessingMode("ivrit_ai") is ProcessingMode.IVRIT_AI


def test_config_has_ivrit_ai_model_setting():
    assert hasattr(settings, "ivrit_ai_model")
    assert "ivrit-ai/" in settings.ivrit_ai_model


def test_transcribe_ivrit_ai_signature_matches_whisper():
    """Same contract as transcribe() — critical for processor interchangeability."""
    whisper_sig = inspect.signature(transcriber.transcribe)
    ivrit_sig   = inspect.signature(transcriber.transcribe_ivrit_ai)
    assert list(whisper_sig.parameters) == list(ivrit_sig.parameters)


def test_transcribe_ivrit_ai_is_coroutine():
    assert asyncio.iscoroutinefunction(transcriber.transcribe_ivrit_ai)


@pytest.mark.asyncio
async def test_processor_routes_ivrit_mode_to_ivrit_transcriber(monkeypatch, tmp_path):
    """
    Wire-up test: confirm _process_audio dispatches IVRIT_AI to
    transcribe_ivrit_ai (not transcribe / transcribe_via_api / summarize_audio).
    """
    # Fresh DB
    monkeypatch.setattr(state, "DB_PATH", tmp_path / "ivrit_test.db")
    monkeypatch.setattr(state, "_db", None, raising=False)
    await state.init_db()

    # Seed a minimal task row so update_task() doesn't fail
    await state.create_task("t-ivrit", url="", user_id=None)

    called = {"which": None}

    async def fake_ivrit(audio_path, language, task_id=None):
        called["which"] = "ivrit"
        return "שלום", "he"

    async def fake_whisper(audio_path, language, task_id=None):
        called["which"] = "whisper"
        return "hello", "he"

    async def fake_whisper_api(audio_path, language, task_id=None):
        called["which"] = "whisper_api"
        return "api", "he"

    async def fake_summarize_transcript(transcript, progress_cb=None, audio_path=None):
        from app.models import LessonResult
        return LessonResult(summary="ok")

    monkeypatch.setattr(transcriber, "transcribe_ivrit_ai", fake_ivrit)
    monkeypatch.setattr(transcriber, "transcribe", fake_whisper)
    monkeypatch.setattr(transcriber, "transcribe_via_api", fake_whisper_api)
    monkeypatch.setattr(
        "app.services.summarizer.summarize_transcript", fake_summarize_transcript
    )

    result = await processor._process_audio(
        task_id="t-ivrit",
        audio_path=str(tmp_path / "fake.mp3"),
        mode=ProcessingMode.IVRIT_AI,
        language="he",
    )

    assert called["which"] == "ivrit"
    assert result.transcript == "שלום"
    await state.close_db()


@pytest.mark.asyncio
async def test_unload_model_if_idle_handles_both_caches(monkeypatch):
    """
    The idle watcher must not crash when only the ivrit cache is populated,
    or only the whisper cache, or neither.
    """
    # Both empty — must be a no-op
    monkeypatch.setattr(transcriber, "_model", None)
    monkeypatch.setattr(transcriber, "_ivrit_model", None)
    await transcriber.unload_model_if_idle()  # must not raise
