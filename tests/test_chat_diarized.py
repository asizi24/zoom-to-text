"""
Tests for the diarized-transcript handling in the chat context builder.

`_build_lesson_context()` is the single place where chat / ask endpoints turn
a completed LessonResult into a string for the LLM. After Track E it must:
  * Prefer diarized_transcript over raw transcript
  * Embed the speaker_map ("מיפוי דוברים: Speaker A=אסף, ...")
  * Fall back to raw transcript when no diarization is available
"""
import pytest

from app.api.routes import _build_lesson_context
from app.models import LessonResult


def test_context_uses_diarized_transcript_when_available():
    result = LessonResult(
        summary="overview",
        transcript="raw line one\nraw line two",
        diarized_transcript="Speaker A: שלום\nSpeaker B: בוקר טוב",
    )
    ctx = _build_lesson_context(result)
    assert "Speaker A: שלום" in ctx
    assert "Speaker B: בוקר טוב" in ctx
    # Diarized label preferred
    assert "תמלול לפי דוברים" in ctx
    # Raw transcript NOT embedded (would double-count and waste tokens)
    assert "raw line one" not in ctx


def test_context_falls_back_to_raw_transcript():
    result = LessonResult(
        summary="overview",
        transcript="just the raw text",
        diarized_transcript=None,
    )
    ctx = _build_lesson_context(result)
    assert "just the raw text" in ctx
    # Plain transcript label
    assert "תמלול:" in ctx
    assert "תמלול לפי דוברים" not in ctx


def test_context_embeds_speaker_map():
    result = LessonResult(
        summary="x",
        diarized_transcript="Speaker A: hi",
        speaker_map={"Speaker A": "אסף", "Speaker B": "מרצה"},
    )
    ctx = _build_lesson_context(result)
    assert "מיפוי דוברים" in ctx
    assert "Speaker A=אסף" in ctx
    assert "Speaker B=מרצה" in ctx


def test_context_no_speaker_map_no_mapping_line():
    result = LessonResult(
        summary="x",
        diarized_transcript="Speaker A: hi",
        speaker_map=None,
    )
    ctx = _build_lesson_context(result)
    assert "מיפוי דוברים" not in ctx


def test_context_caps_transcript_at_30k_chars():
    """Long transcripts are truncated so the prompt doesn't blow up."""
    long_text = "x" * 100_000
    result = LessonResult(
        summary="x",
        diarized_transcript=long_text,
    )
    ctx = _build_lesson_context(result)
    # The truncated slice must appear, the full text must NOT
    assert "x" * 30_000 in ctx
    assert "x" * 30_001 not in ctx
