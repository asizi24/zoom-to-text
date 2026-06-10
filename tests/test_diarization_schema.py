"""
Phase A tests for Task 1.2 (Gemini-based diarization).

Covers schema additions on LessonResult and the ENABLE_DIARIZATION
config flag.
"""
import json


def test_lesson_result_has_optional_diarized_transcript():
    from app.models import LessonResult

    r = LessonResult()
    assert r.diarized_transcript is None
    assert r.speaker_map is None


def test_lesson_result_diarization_round_trip():
    from app.models import LessonResult

    r = LessonResult(
        summary="s",
        diarized_transcript="Speaker A: hi\nSpeaker B: hello",
        speaker_map={"Speaker A": "Asaf", "Speaker B": "Dan"},
    )
    blob = r.model_dump(mode="json")
    parsed = LessonResult.model_validate(json.loads(json.dumps(blob)))
    assert parsed.diarized_transcript == "Speaker A: hi\nSpeaker B: hello"
    assert parsed.speaker_map == {"Speaker A": "Asaf", "Speaker B": "Dan"}


def test_lesson_result_loads_old_json_without_diarization_fields():
    """Old SQLite tasks (pre-1.2) must deserialize cleanly."""
    from app.models import LessonResult

    old = {"summary": "s", "chapters": [], "quiz": [], "language": "he"}
    r = LessonResult.model_validate(old)
    assert r.diarized_transcript is None
    assert r.speaker_map is None


def test_enable_diarization_default_true():
    from app.config import settings
    assert settings.enable_diarization is True
