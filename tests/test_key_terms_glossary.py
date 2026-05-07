"""
Tests for the Key Terms Glossary feature.

Covers:
- KeyTerm Pydantic model
- _parse_extraction_response returns key_terms list
- LessonResult backward compatibility (old JSON without key_terms → defaults to [])
- _EXTRACTION_PROMPT mentions key_terms
- _merge_results copies key_terms from extraction dict to LessonResult
- Obsidian exporter renders ## Key Terms section when terms present
- Obsidian exporter omits the section when key_terms is empty
"""

import json

import pytest

from app.models import LessonResult, TaskResponse, TaskStatus


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_task(result: LessonResult) -> TaskResponse:
    return TaskResponse(
        task_id="aaaa-bbbb",
        status=TaskStatus.COMPLETED,
        progress=100,
        message="ok",
        created_at="2026-05-07T10:00:00+00:00",
        url="https://zoom.us/rec/test",
        result=result,
        has_audio=False,
    )


# ── KeyTerm model ─────────────────────────────────────────────────────────────


def test_key_term_model_has_required_fields():
    """KeyTerm must have term and definition; context is optional."""
    from app.models import KeyTerm

    kt = KeyTerm(
        term="Bell's theorem", definition="A theorem about quantum entanglement"
    )
    assert kt.term == "Bell's theorem"
    assert kt.definition == "A theorem about quantum entanglement"
    assert kt.context is None


def test_key_term_model_accepts_context():
    """KeyTerm with context populates correctly."""
    from app.models import KeyTerm

    kt = KeyTerm(
        term="Superposition",
        definition="A quantum state that is a combination of multiple basis states",
        context="Introduced in the first chapter of the lecture",
    )
    assert kt.context == "Introduced in the first chapter of the lecture"


def test_key_term_model_requires_term_and_definition():
    """Missing term or definition raises ValidationError."""
    from pydantic import ValidationError
    from app.models import KeyTerm

    with pytest.raises(ValidationError):
        KeyTerm(definition="no term provided")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        KeyTerm(term="no definition provided")  # type: ignore[call-arg]


# ── LessonResult schema ───────────────────────────────────────────────────────


def test_lesson_result_has_key_terms_field_defaulting_to_empty():
    """LessonResult.key_terms defaults to [] when not supplied."""
    r = LessonResult(summary="A lecture about physics")
    assert r.key_terms == []


def test_lesson_result_loads_old_json_without_key_terms():
    """Old stored JSON that has no key_terms key → key_terms defaults to []."""
    old_json = json.dumps(
        {
            "summary": "An old lecture",
            "chapters": [],
            "quiz": [],
            "flashcards": [],
            "language": "en",
            # key_terms deliberately absent
        }
    )
    r = LessonResult.model_validate_json(old_json)
    assert r.key_terms == []


def test_lesson_result_loads_key_terms_when_present():
    """JSON with key_terms list → LessonResult.key_terms is populated."""
    data = {
        "summary": "Quantum lecture",
        "key_terms": [
            {"term": "Entanglement", "definition": "Correlated quantum states"},
            {
                "term": "Superposition",
                "definition": "Multiple states simultaneously",
                "context": "First chapter",
            },
        ],
    }
    r = LessonResult.model_validate(data)
    assert len(r.key_terms) == 2
    assert r.key_terms[0].term == "Entanglement"
    assert r.key_terms[1].context == "First chapter"


# ── Parser ────────────────────────────────────────────────────────────────────


def test_parse_extraction_includes_key_terms():
    """key_terms in LLM JSON → parsed as list of KeyTerm objects."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps(
        {
            "action_items": [],
            "decisions": [],
            "open_questions": [],
            "sentiment_analysis": None,
            "objections_tracked": [],
            "key_terms": [
                {
                    "term": "Bell's theorem",
                    "definition": "Limits on hidden-variable theories",
                },
                {
                    "term": "Non-locality",
                    "definition": "Quantum correlations across distances",
                    "context": "Debated after the Aspect experiment",
                },
            ],
        },
        ensure_ascii=False,
    )

    out = _parse_extraction_response(raw)
    assert "key_terms" in out
    assert len(out["key_terms"]) == 2
    assert out["key_terms"][0].term == "Bell's theorem"
    assert out["key_terms"][1].context == "Debated after the Aspect experiment"


def test_parse_extraction_key_terms_absent_defaults_to_empty():
    """If LLM omits key_terms entirely → defaults to []."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps(
        {
            "action_items": [],
            "decisions": [],
            "open_questions": [],
            "sentiment_analysis": None,
            "objections_tracked": [],
            # key_terms absent
        }
    )
    out = _parse_extraction_response(raw)
    assert out.get("key_terms", []) == []


def test_parse_extraction_key_terms_as_empty_list():
    """LLM returns key_terms: [] explicitly → empty list."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps(
        {
            "action_items": [],
            "decisions": [],
            "open_questions": [],
            "sentiment_analysis": None,
            "objections_tracked": [],
            "key_terms": [],
        }
    )
    out = _parse_extraction_response(raw)
    assert out["key_terms"] == []


# ── Prompt ────────────────────────────────────────────────────────────────────


def test_extraction_prompt_includes_key_terms():
    """_EXTRACTION_PROMPT must mention key_terms so the LLM knows to produce it."""
    from app.services.summarizer import _EXTRACTION_PROMPT

    assert "key_terms" in _EXTRACTION_PROMPT


def test_extraction_prompt_key_terms_example_has_term_and_definition():
    """Prompt must show the {term, definition} shape to guide the model."""
    from app.services.summarizer import _EXTRACTION_PROMPT

    assert '"term"' in _EXTRACTION_PROMPT or "'term'" in _EXTRACTION_PROMPT
    assert '"definition"' in _EXTRACTION_PROMPT or "'definition'" in _EXTRACTION_PROMPT


# ── Merger ────────────────────────────────────────────────────────────────────


def test_merge_results_copies_key_terms_to_lesson_result():
    """_merge_results sets synthesis.key_terms from the extraction dict."""
    from app.models import KeyTerm
    from app.services.summarizer import _merge_results

    synthesis = LessonResult(summary="Physics lecture")
    extraction = {
        "action_items": [],
        "decisions": [],
        "open_questions": [],
        "sentiment_analysis": None,
        "objections_tracked": [],
        "key_terms": [
            KeyTerm(term="Qubit", definition="A quantum bit"),
        ],
    }

    merged = _merge_results(
        synthesis, extraction, raw_summary=None, raw_extraction=None
    )
    assert len(merged.key_terms) == 1
    assert merged.key_terms[0].term == "Qubit"


def test_merge_results_key_terms_empty_when_extraction_none():
    """When extraction call fails (None) key_terms stays []."""
    from app.services.summarizer import _merge_results

    synthesis = LessonResult(summary="Some lecture")
    merged = _merge_results(
        synthesis, extraction=None, raw_summary=None, raw_extraction=None
    )
    assert merged.key_terms == []


def test_merge_results_key_terms_empty_when_extraction_missing_field():
    """When extraction dict has no key_terms key → stays []."""
    from app.services.summarizer import _merge_results

    synthesis = LessonResult(summary="Some lecture")
    extraction = {
        "action_items": [],
        "decisions": [],
        "open_questions": [],
        "sentiment_analysis": None,
        "objections_tracked": [],
        # key_terms deliberately absent
    }
    merged = _merge_results(
        synthesis, extraction, raw_summary=None, raw_extraction=None
    )
    assert merged.key_terms == []


# ── Obsidian exporter ─────────────────────────────────────────────────────────


def test_obsidian_export_includes_key_terms_section():
    """When key_terms is populated, export includes ## Key Terms section."""
    from app.models import KeyTerm
    from app.services.exporters.markdown import build_obsidian_markdown

    r = LessonResult(
        summary="A quantum physics lecture",
        key_terms=[
            KeyTerm(term="Entanglement", definition="Correlated quantum states"),
            KeyTerm(
                term="Superposition",
                definition="Multiple states simultaneously",
                context="Introduced early",
            ),
        ],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## Key Terms" in md
    assert "Entanglement" in md
    assert "Correlated quantum states" in md
    assert "Superposition" in md
    assert "Multiple states simultaneously" in md


def test_obsidian_export_includes_context_when_present():
    """When a KeyTerm has context, it appears in the export."""
    from app.models import KeyTerm
    from app.services.exporters.markdown import build_obsidian_markdown

    r = LessonResult(
        summary="Lecture",
        key_terms=[
            KeyTerm(term="Qubit", definition="Quantum bit", context="Chapter 2"),
        ],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "Chapter 2" in md


def test_obsidian_export_omits_key_terms_section_when_empty():
    """When key_terms is [], the ## Key Terms section must not appear."""
    from app.services.exporters.markdown import build_obsidian_markdown

    r = LessonResult(summary="A plain lecture")
    md = build_obsidian_markdown(_make_task(r))
    assert "## Key Terms" not in md


def test_obsidian_export_omits_key_terms_section_for_meetings():
    """Meetings don't usually have key terms; section should be absent when empty."""
    from app.services.exporters.markdown import build_obsidian_markdown

    r = LessonResult(summary="A product review meeting", content_type="meeting")
    md = build_obsidian_markdown(_make_task(r))
    assert "## Key Terms" not in md
