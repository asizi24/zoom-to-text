"""
Phase C tests for Task 1.2 — diarization wired into summarize_transcript.

Covers:
- Diarization runs BEFORE synthesis + extraction; both downstream calls
  receive the diarized transcript.
- Diarization failure → graceful skip (downstream calls get raw transcript;
  diarized_transcript and speaker_map remain None).
- ENABLE_DIARIZATION=False → diarization is NOT called.
- LessonResult is populated with diarized_transcript + speaker_map on success.
"""
import pytest


@pytest.mark.asyncio
async def test_summarize_transcript_runs_diarize_then_synth_then_extract(monkeypatch):
    """Happy path: diarize → (synth ‖ extract), both downstream get diarized text."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "enable_diarization", True, raising=False)
    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    order: list[str] = []

    def fake_diarize(transcript):
        order.append("diarize")
        assert transcript == "raw transcript"
        return ("Speaker A: raw transcript", {"Speaker A": "אסף"})

    def fake_synth(text):
        order.append(f"synth({text})")
        return (LessonResult(summary="ok", content_type="meeting"), "raw S")

    def fake_extract(text):
        order.append(f"extract({text})")
        return (
            {"action_items": [], "decisions": [], "open_questions": [],
             "sentiment_analysis": None, "objections_tracked": []},
            "raw E",
        )

    monkeypatch.setattr(s, "_diarize_transcript_sync", fake_diarize)
    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract)
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("raw transcript")

    assert order[0] == "diarize"
    # Both downstream calls received the diarized text (not the raw one)
    assert "synth(Speaker A: raw transcript)" in order
    assert "extract(Speaker A: raw transcript)" in order

    # Persisted on LessonResult
    assert result.diarized_transcript == "Speaker A: raw transcript"
    assert result.speaker_map == {"Speaker A": "אסף"}


@pytest.mark.asyncio
async def test_summarize_transcript_diarization_failure_falls_back_to_raw(monkeypatch):
    """If diarize raises, downstream calls receive the raw transcript."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "enable_diarization", True, raising=False)
    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    received_by_downstream: list[str] = []

    def fake_diarize_fails(transcript):
        raise ValueError("simulated diarization JSON parse failure")

    def fake_synth(text):
        received_by_downstream.append(text)
        return (LessonResult(summary="ok"), "raw")

    def fake_extract(text):
        received_by_downstream.append(text)
        return (
            {"action_items": [], "decisions": [], "open_questions": [],
             "sentiment_analysis": None, "objections_tracked": []},
            "raw",
        )

    monkeypatch.setattr(s, "_diarize_transcript_sync", fake_diarize_fails)
    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract)
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("the raw transcript")

    # Both downstream calls received the RAW transcript
    assert received_by_downstream == ["the raw transcript", "the raw transcript"]
    # Diarization fields stay unset
    assert result.diarized_transcript is None
    assert result.speaker_map is None
    # Task still completes
    assert result.summary == "ok"


@pytest.mark.asyncio
async def test_summarize_transcript_skips_diarization_when_flag_disabled(monkeypatch):
    """ENABLE_DIARIZATION=False → diarize is not called."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "enable_diarization", False, raising=False)
    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    diarize_called = {"n": 0}

    def fake_diarize(transcript):
        diarize_called["n"] += 1
        return ("Speaker A: x", {})

    def fake_synth(text):
        return (LessonResult(summary="ok"), "raw")

    def fake_extract(text):
        return (
            {"action_items": [], "decisions": [], "open_questions": [],
             "sentiment_analysis": None, "objections_tracked": []},
            "raw",
        )

    monkeypatch.setattr(s, "_diarize_transcript_sync", fake_diarize)
    monkeypatch.setattr(s, "_synthesize_text_capture", fake_synth)
    monkeypatch.setattr(s, "_extract_text_capture", fake_extract)
    monkeypatch.setattr(s, "_apply_critique_pipeline", lambda r, cb=None: r)

    result = await s.summarize_transcript("raw transcript")

    assert diarize_called["n"] == 0
    assert result.diarized_transcript is None
    assert result.speaker_map is None


@pytest.mark.asyncio
async def test_summarize_audio_does_not_run_diarization(monkeypatch):
    """GEMINI_DIRECT path skips diarization entirely (audio model handles speakers)."""
    from app.config import settings
    from app.models import LessonResult
    from app.services import summarizer as s

    monkeypatch.setattr(settings, "enable_diarization", True, raising=False)
    monkeypatch.setattr(settings, "llm_provider", "gemini", raising=False)

    diarize_called = {"n": 0}

    def fake_diarize(transcript):
        diarize_called["n"] += 1
        return ("Speaker A: x", {})

    monkeypatch.setattr(s, "_diarize_transcript_sync", fake_diarize)
    monkeypatch.setattr(s, "_upload_audio_to_gemini", lambda p, cb=None: object())
    monkeypatch.setattr(s, "_delete_gemini_file", lambda f: None)
    monkeypatch.setattr(
        s, "_synthesize_audio_capture",
        lambda f, cb=None: (LessonResult(summary="audio summary"), "raw"),
    )
    monkeypatch.setattr(
        s, "_extract_audio_capture",
        lambda f: ({"action_items": [], "decisions": [], "open_questions": [],
                    "sentiment_analysis": None, "objections_tracked": []}, "raw"),
    )

    result = await s.summarize_audio("/tmp/fake.mp3")

    assert diarize_called["n"] == 0
    assert result.summary == "audio summary"
    assert result.diarized_transcript is None
