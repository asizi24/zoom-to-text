"""
Tests for the schema upgrade (UPGRADE_PROMPT.md Task 1.1).

Covers backward-compat with old SQLite JSON blobs and the shape of the
five new optional fields on LessonResult plus the structured
raw_llm_response debug field.
"""
import json
import pytest


# ── Backward compatibility ─────────────────────────────────────────────────────

def test_lesson_result_loads_old_json_without_new_fields():
    """Old SQLite tasks (pre-schema-upgrade) must deserialize cleanly."""
    from app.models import LessonResult

    old_blob = {
        "transcript": "old transcript",
        "summary": "old summary",
        "chapters": [{"title": "ch1", "content": "...", "key_points": ["a", "b"]}],
        "quiz": [
            {
                "question": "Q?",
                "options": ["א. 1", "ב. 2", "ג. 3", "ד. 4"],
                "correct_answer": "א. 1",
                "explanation": "",
            }
        ],
        "flashcards": [],
        "language": "he",
    }

    r = LessonResult.model_validate(old_blob)
    assert r.summary == "old summary"
    # New fields default empty / None
    assert r.action_items == []
    assert r.decisions == []
    assert r.open_questions == []
    assert r.sentiment_analysis is None
    assert r.objections_tracked == []
    assert r.content_type is None
    assert r.raw_llm_response is None


def test_lesson_result_round_trips_through_json():
    """JSON serialization must be stable for SQLite persistence."""
    from app.models import LessonResult, ActionItem, Decision

    r = LessonResult(
        summary="s",
        content_type="meeting",
        action_items=[ActionItem(owner="Asaf", task="ship the spec")],
        decisions=[Decision(decision="use option B", stakeholders=["Asaf", "Dan"])],
    )
    blob = r.model_dump(mode="json")
    parsed = LessonResult.model_validate(json.loads(json.dumps(blob)))
    assert parsed.action_items[0].owner == "Asaf"
    assert parsed.decisions[0].stakeholders == ["Asaf", "Dan"]
    assert parsed.content_type == "meeting"


# ── ActionItem ─────────────────────────────────────────────────────────────────

def test_action_item_minimal_only_owner_and_task():
    from app.models import ActionItem

    a = ActionItem(owner="Asaf", task="ship the feature")
    assert a.owner == "Asaf"
    assert a.task == "ship the feature"
    assert a.deadline is None
    assert a.priority is None
    assert a.source_quote is None


def test_action_item_full():
    from app.models import ActionItem

    a = ActionItem(
        owner="Dan",
        task="write the design doc",
        deadline="EOW",
        priority="high",
        source_quote="Dan, can you draft this by Friday?",
    )
    assert a.deadline == "EOW"
    assert a.priority == "high"


# ── Decision ──────────────────────────────────────────────────────────────────

def test_decision_minimal():
    from app.models import Decision

    d = Decision(decision="ship Cycle B")
    assert d.decision == "ship Cycle B"
    assert d.stakeholders == []
    assert d.context is None
    assert d.source_quote is None


def test_decision_with_stakeholders():
    from app.models import Decision

    d = Decision(
        decision="adopt option B",
        context="evaluating prompt strategies",
        stakeholders=["Asaf", "Dan"],
        source_quote="we go with parallel calls",
    )
    assert d.stakeholders == ["Asaf", "Dan"]


# ── OpenQuestion ──────────────────────────────────────────────────────────────

def test_open_question_minimal():
    from app.models import OpenQuestion

    q = OpenQuestion(question="who owns rollout?")
    assert q.question == "who owns rollout?"
    assert q.raised_by is None
    assert q.context is None


# ── SentimentAnalysis ─────────────────────────────────────────────────────────

def test_sentiment_analysis_full_shape():
    from app.models import SentimentAnalysis, PerSpeakerSentiment, ToneShift

    s = SentimentAnalysis(
        overall_tone="constructive",
        per_speaker_sentiment=[
            PerSpeakerSentiment(speaker="Speaker A", sentiment="positive"),
            PerSpeakerSentiment(speaker="Speaker B", sentiment="negative", rationale="pushed back hard"),
        ],
        shifts_in_tone=[ToneShift(at="00:12:30", **{"from": "neutral", "to": "tense"}, trigger="budget mention")],
    )
    assert s.overall_tone == "constructive"
    assert len(s.per_speaker_sentiment) == 2
    assert s.shifts_in_tone[0].from_tone == "neutral"
    assert s.shifts_in_tone[0].to_tone == "tense"


def test_sentiment_analysis_minimal():
    from app.models import SentimentAnalysis

    s = SentimentAnalysis(overall_tone="neutral")
    assert s.per_speaker_sentiment == []
    assert s.shifts_in_tone == []


def test_tone_shift_serializes_with_from_to_aliases():
    """ToneShift uses 'from'/'to' aliases since 'from' is a Python keyword."""
    from app.models import ToneShift

    t = ToneShift(**{"at": "00:05:00", "from": "calm", "to": "heated"})
    blob = t.model_dump(mode="json", by_alias=True)
    assert blob["from"] == "calm"
    assert blob["to"] == "heated"


# ── Objection ─────────────────────────────────────────────────────────────────

def test_objection_resolved_optional_bool():
    from app.models import Objection

    o1 = Objection(objection="cost is too high")
    assert o1.resolved is None  # unresolved/unknown by default

    o2 = Objection(
        objection="timeline is unrealistic",
        raised_by="Dan",
        response_given="will revisit Q3",
        resolved=False,
    )
    assert o2.resolved is False


# ── RawLLMResponse ────────────────────────────────────────────────────────────

def test_raw_llm_response_structured_object():
    from app.models import RawLLMResponse

    r = RawLLMResponse(summary_call="raw text 1", extraction_call="raw text 2")
    assert r.summary_call == "raw text 1"
    assert r.extraction_call == "raw text 2"


def test_raw_llm_response_partial_extraction_only():
    """When the synthesis call's raw text is unavailable but extraction's is."""
    from app.models import RawLLMResponse

    r = RawLLMResponse(extraction_call="extraction raw")
    assert r.summary_call is None
    assert r.extraction_call == "extraction raw"


# ── Chapter.start_time ────────────────────────────────────────────────────────

def test_chapter_without_start_time_loads_as_none():
    """Old Chapter blobs without start_time must deserialize with start_time=None."""
    from app.models import Chapter

    ch = Chapter(title="Intro", content="some content", key_points=["a", "b"])
    assert ch.start_time is None


def test_chapter_with_start_time_loads_correctly():
    from app.models import Chapter

    ch = Chapter(title="Hooks", content="...", key_points=[], start_time="[07:23]")
    assert ch.start_time == "[07:23]"


def test_lesson_result_old_json_chapter_missing_start_time():
    """Old SQLite blobs whose chapters lack start_time must still validate."""
    from app.models import LessonResult

    blob = {
        "summary": "s",
        "chapters": [{"title": "ch1", "content": "...", "key_points": []}],
        "quiz": [],
        "language": "he",
    }
    r = LessonResult.model_validate(blob)
    assert r.chapters[0].start_time is None


# ── content_type ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctype", ["lecture", "meeting", "discussion"])
def test_lesson_result_accepts_known_content_types(ctype):
    from app.models import LessonResult

    r = LessonResult(content_type=ctype)
    assert r.content_type == ctype


def test_lesson_result_content_type_is_optional():
    from app.models import LessonResult

    r = LessonResult()
    assert r.content_type is None
