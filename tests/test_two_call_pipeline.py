"""
Phase 4 tests: parallel two-call orchestration in summarize_audio /
summarize_transcript.

Covers:
- Happy path: synthesis + extraction merged into a single LessonResult.
- Extraction failure → graceful skip, task still completes.
- Synthesis failure → propagates as today (task fails).
- raw_llm_response populated only when LLM_DEBUG_RAW_RESPONSES is true.
- content_type from synthesis call ends up on the LessonResult.
"""
import asyncio
import json

import pytest


# ── _merge_results helper ─────────────────────────────────────────────────────

def test_merge_results_combines_synthesis_and_extraction():
    """The merge helper layers extraction fields onto the synthesis LessonResult."""
    from app.models import (
        ActionItem,
        Decision,
        LessonResult,
    )
    from app.services.summarizer import _merge_results

    synthesis = LessonResult(
        summary="s",
        chapters=[],
        quiz=[],
        content_type="meeting",
    )
    extraction = {
        "action_items": [ActionItem(owner="A", task="t")],
        "decisions": [Decision(decision="d")],
        "open_questions": [],
        "sentiment_analysis": None,
        "objections_tracked": [],
    }

    merged = _merge_results(
        synthesis, extraction, raw_summary=None, raw_extraction=None
    )

    assert merged.summary == "s"
    assert merged.content_type == "meeting"
    assert len(merged.action_items) == 1
    assert merged.action_items[0].owner == "A"
    assert len(merged.decisions) == 1
    assert merged.raw_llm_response is None  # flag is off by default


def test_merge_results_handles_none_extraction():
    """When extraction failed entirely, merge returns synthesis-only LessonResult."""
    from app.models import LessonResult
    from app.services.summarizer import _merge_results

    synthesis = LessonResult(summary="s")
    merged = _merge_results(synthesis, None, raw_summary=None, raw_extraction=None)
    assert merged.summary == "s"
    assert merged.action_items == []
    assert merged.decisions == []
    assert merged.sentiment_analysis is None
    assert merged.objections_tracked == []


def test_merge_results_persists_raw_responses_when_flag_on(monkeypatch):
    """When LLM_DEBUG_RAW_RESPONSES=true, raw_llm_response is populated."""
    from app.config import settings
    from app.models import LessonResult
    from app.services.summarizer import _merge_results

    monkeypatch.setattr(settings, "llm_debug_raw_responses", True, raising=False)

    synthesis = LessonResult(summary="s")
    merged = _merge_results(
        synthesis, {}, raw_summary="raw1", raw_extraction="raw2"
    )

    assert merged.raw_llm_response is not None
    assert merged.raw_llm_response.summary_call == "raw1"
    assert merged.raw_llm_response.extraction_call == "raw2"


def test_merge_results_ignores_raw_when_flag_off(monkeypatch):
    """Default behavior: raw responses are NOT persisted."""
    from app.config import settings
    from app.models import LessonResult
    from app.services.summarizer import _merge_results

    monkeypatch.setattr(settings, "llm_debug_raw_responses", False, raising=False)

    synthesis = LessonResult(summary="s")
    merged = _merge_results(
        synthesis, {}, raw_summary="raw1", raw_extraction="raw2"
    )
    assert merged.raw_llm_response is None


# ── summarize_transcript: parallel orchestration ──────────────────────────────

@pytest.mark.asyncio
async def test_summarize_transcript_runs_both_calls_and_merges(monkeypatch):
    """Happy path: text mode runs synthesis + extraction and merges."""
    from app.models import LessonResult
    from app.services import summarizer as s

    # Stub synthesis: returns parsed LessonResult and raw text
    synth_called = {"n": 0}

    def fake_synth_text_capture(transcript):
        synth_called["n"] += 1
        return (
            LessonResult(summary="meeting summary", content_type="meeting"),
            "raw synthesis text",
        )

    # Stub extraction: returns dict + raw text
    extract_called = {"n": 0}

    def fake_extract_text(transcript):
        extract_called["n"] += 1
        from app.models import ActionItem
        return (
            {"action_items": [ActionItem(owner="Asaf", task="ship")],
             "decisions": [], "open_questions": [],
             "sentiment_analysis": None, "objections_tracked": []},
            "raw extraction text",
        )

    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth_text_capture)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract_text)
    # Skip critique pipeline
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("some Hebrew transcript")

    assert synth_called["n"] == 1
    assert extract_called["n"] == 1
    assert result.summary == "meeting summary"
    assert result.content_type == "meeting"
    assert len(result.action_items) == 1
    assert result.action_items[0].owner == "Asaf"


@pytest.mark.asyncio
async def test_summarize_transcript_extraction_failure_graceful_skip(monkeypatch):
    """If extraction raises, the task still completes with synthesis-only fields."""
    from app.models import LessonResult
    from app.services import summarizer as s

    def fake_synth_text_capture(transcript):
        return (LessonResult(summary="ok"), "raw")

    def fake_extract_fails(transcript):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth_text_capture)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract_fails)
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("transcript")

    assert result.summary == "ok"
    assert result.action_items == []
    assert result.decisions == []
    assert result.sentiment_analysis is None


@pytest.mark.asyncio
async def test_summarize_transcript_synthesis_failure_propagates(monkeypatch):
    """If synthesis fails, the whole task fails (no graceful skip)."""
    from app.services import summarizer as s

    def fake_synth_fails(transcript):
        raise RuntimeError("Gemini exploded")

    def fake_extract_text(transcript):
        from app.models import ActionItem
        return ({"action_items": [], "decisions": [],
                 "open_questions": [], "sentiment_analysis": None,
                 "objections_tracked": []}, "raw")

    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth_fails)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract_text)

    with pytest.raises(RuntimeError, match="Gemini exploded"):
        await s.summarize_transcript("transcript")


@pytest.mark.asyncio
async def test_summarize_transcript_raw_responses_persisted_when_flag_on(monkeypatch):
    """LLM_DEBUG_RAW_RESPONSES=true → raw_llm_response populated end-to-end."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "llm_debug_raw_responses", True, raising=False)

    def fake_synth(transcript):
        return (LessonResult(summary="s"), "S-RAW")

    def fake_extract(transcript):
        return ({"action_items": [], "decisions": [],
                 "open_questions": [], "sentiment_analysis": None,
                 "objections_tracked": []}, "E-RAW")

    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract)
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("transcript")

    assert result.raw_llm_response is not None
    assert result.raw_llm_response.summary_call == "S-RAW"
    assert result.raw_llm_response.extraction_call == "E-RAW"


# ── summarize_audio: parallel orchestration (Gemini provider) ─────────────────

@pytest.mark.asyncio
async def test_summarize_audio_runs_both_calls_and_merges(monkeypatch):
    """Audio mode: upload once, synthesize + extract in parallel, cleanup once."""
    from app.config import settings
    from app.models import LessonResult, ActionItem
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    upload_called = {"n": 0}
    cleanup_called = {"n": 0}
    synth_called = {"n": 0}
    extract_called = {"n": 0}

    fake_audio_handle = object()

    def fake_upload(audio_path, progress_cb=None):
        upload_called["n"] += 1
        return fake_audio_handle

    def fake_cleanup(audio_file):
        cleanup_called["n"] += 1

    def fake_synth_audio(audio_file, progress_cb=None):
        synth_called["n"] += 1
        assert audio_file is fake_audio_handle
        return (LessonResult(summary="audio summary", content_type="lecture"), "raw S")

    def fake_extract_audio(audio_file):
        extract_called["n"] += 1
        assert audio_file is fake_audio_handle
        return ({"action_items": [ActionItem(owner="A", task="t")],
                 "decisions": [], "open_questions": [],
                 "sentiment_analysis": None, "objections_tracked": []}, "raw E")

    monkeypatch.setattr(s, "_upload_audio_to_gemini", fake_upload)
    monkeypatch.setattr(s, "_delete_gemini_file", fake_cleanup)
    monkeypatch.setattr(s, "_synthesize_audio_capture", fake_synth_audio)
    monkeypatch.setattr(s, "_extract_audio_capture", fake_extract_audio)

    result = await s.summarize_audio("/tmp/fake.mp3")

    assert upload_called["n"] == 1
    assert synth_called["n"] == 1
    assert extract_called["n"] == 1
    assert cleanup_called["n"] == 1
    assert result.summary == "audio summary"
    assert result.content_type == "lecture"
    assert len(result.action_items) == 1


@pytest.mark.asyncio
async def test_summarize_audio_cleans_up_even_when_extraction_fails(monkeypatch):
    """Cleanup of uploaded audio runs even on extraction failure."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    cleanup_called = {"n": 0}

    monkeypatch.setattr(s, "_upload_audio_to_gemini", lambda p, cb=None: object())

    def fake_cleanup(audio_file):
        cleanup_called["n"] += 1

    monkeypatch.setattr(s, "_delete_gemini_file", fake_cleanup)
    monkeypatch.setattr(
        s, "_synthesize_audio_capture",
        lambda f, cb=None: (LessonResult(summary="s"), "raw"),
    )
    monkeypatch.setattr(
        s, "_extract_audio_capture",
        lambda f: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = await s.summarize_audio("/tmp/fake.mp3")
    assert result.summary == "s"
    assert result.action_items == []
    assert cleanup_called["n"] == 1
