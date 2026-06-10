"""
Tests for the extraction call (Call 2 of the two-call pipeline).

Covers:
- The parser `_parse_extraction_response` accepts the contract shape and
  defaults missing fields to empty.
- The text-mode extraction helper sends a temperature=0.2 request.
- Failures in extraction don't crash the pipeline (handled in Phase 4 tests).
"""
import json
import pytest


# ── Parser ─────────────────────────────────────────────────────────────────────

def test_parse_extraction_full_payload():
    """All five fields populated → dict with parsed sub-models."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps({
        "action_items": [
            {"owner": "Asaf", "task": "ship the schema",
             "deadline": "EOW", "priority": "high",
             "source_quote": "Asaf will ship by Friday"}
        ],
        "decisions": [
            {"decision": "use option B",
             "context": "evaluating prompt strategies",
             "stakeholders": ["Asaf", "Dan"],
             "source_quote": "we go with parallel calls"}
        ],
        "open_questions": [
            {"question": "who owns rollout?", "raised_by": "Dan",
             "context": "bringing it up at next standup"}
        ],
        "sentiment_analysis": {
            "overall_tone": "constructive",
            "per_speaker_sentiment": [
                {"speaker": "Speaker A", "sentiment": "positive"}
            ],
            "shifts_in_tone": [
                {"at": "00:12:00", "from": "neutral", "to": "tense",
                 "trigger": "budget mention"}
            ]
        },
        "objections_tracked": [
            {"objection": "cost is too high", "raised_by": "Dan",
             "response_given": "will revisit Q3", "resolved": False}
        ]
    }, ensure_ascii=False)

    out = _parse_extraction_response(raw)

    assert len(out["action_items"]) == 1
    assert out["action_items"][0].owner == "Asaf"

    assert len(out["decisions"]) == 1
    assert out["decisions"][0].stakeholders == ["Asaf", "Dan"]

    assert len(out["open_questions"]) == 1
    assert out["open_questions"][0].question == "who owns rollout?"

    assert out["sentiment_analysis"].overall_tone == "constructive"
    assert out["sentiment_analysis"].shifts_in_tone[0].from_tone == "neutral"
    assert out["sentiment_analysis"].shifts_in_tone[0].to_tone == "tense"

    assert len(out["objections_tracked"]) == 1
    assert out["objections_tracked"][0].resolved is False


def test_parse_extraction_empty_lecture_payload():
    """Lecture content typically returns empty arrays — no crash."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps({
        "action_items": [],
        "decisions": [],
        "open_questions": [
            {"question": "is the proof generalizable to higher dimensions?"}
        ],
        "sentiment_analysis": None,
        "objections_tracked": []
    }, ensure_ascii=False)

    out = _parse_extraction_response(raw)
    assert out["action_items"] == []
    assert out["decisions"] == []
    assert out["objections_tracked"] == []
    assert out["sentiment_analysis"] is None
    assert len(out["open_questions"]) == 1


def test_parse_extraction_strips_markdown_fences():
    """Provider may wrap JSON in ```json ... ``` despite instructions."""
    from app.services.summarizer import _parse_extraction_response

    raw = (
        "```json\n"
        + json.dumps({"action_items": [{"owner": "A", "task": "t"}]})
        + "\n```"
    )
    out = _parse_extraction_response(raw)
    assert out["action_items"][0].owner == "A"


def test_parse_extraction_missing_fields_default_empty():
    """Provider returns only some keys → others default to empty/None."""
    from app.services.summarizer import _parse_extraction_response

    raw = json.dumps({
        "action_items": [{"owner": "A", "task": "t"}]
        # decisions, open_questions, sentiment_analysis, objections_tracked all absent
    })
    out = _parse_extraction_response(raw)
    assert len(out["action_items"]) == 1
    assert out["decisions"] == []
    assert out["open_questions"] == []
    assert out["sentiment_analysis"] is None
    assert out["objections_tracked"] == []


def test_parse_extraction_invalid_json_raises():
    """Non-JSON garbage → raise (caller handles via best-effort skip)."""
    from app.services.summarizer import _parse_extraction_response

    with pytest.raises((ValueError, RuntimeError)):
        _parse_extraction_response("totally not json {{")


# ── Prompt content ─────────────────────────────────────────────────────────────

def test_extraction_prompt_includes_required_keys():
    """The prompt must mention all 5 output keys so the LLM knows the contract."""
    from app.services.summarizer import _EXTRACTION_PROMPT

    for key in ("action_items", "decisions", "open_questions",
                "sentiment_analysis", "objections_tracked"):
        assert key in _EXTRACTION_PROMPT, f"prompt is missing {key}"


def test_extraction_prompt_has_bilingual_few_shots():
    """Prompt includes one Hebrew and one English few-shot example."""
    from app.services.summarizer import _EXTRACTION_PROMPT

    # Hebrew letters
    has_hebrew = any("֐" <= ch <= "׿" for ch in _EXTRACTION_PROMPT)
    assert has_hebrew, "extraction prompt should include a Hebrew example"
    # English letters (basic latin alphabet block)
    has_english_word = any(
        word.isascii() and word.isalpha() and len(word) >= 4
        for word in _EXTRACTION_PROMPT.split()
    )
    assert has_english_word, "extraction prompt should include an English example"


def test_extraction_prompt_specifies_temperature_intent():
    """The temperature constant for extraction is 0.2 per the spec."""
    from app.services.summarizer import _EXTRACTION_TEMPERATURE
    assert _EXTRACTION_TEMPERATURE == 0.2


# ── Sync helper (text mode) ────────────────────────────────────────────────────

def test_extract_artifacts_text_sync_uses_temperature_02(monkeypatch):
    """The sync text extraction helper invokes the LLM with temperature=0.2."""
    from app.services import summarizer as s

    captured = {}

    class FakeResponse:
        def __init__(self, text):
            self._text = text
        @property
        def text(self):
            return self._text
        @property
        def candidates(self):
            class Part:
                text = self._text
                thought = False
            class C:
                content = type("X", (), {"parts": [Part()]})()
            return [C()]

    def fake_generate(client, contents, max_retries=3, config=None):
        captured["config"] = config
        captured["contents"] = contents
        return FakeResponse(json.dumps({"action_items": []}))

    monkeypatch.setattr(s, "_generate_with_retry", fake_generate)
    monkeypatch.setattr(s, "_get_client", lambda: object())

    out = s._extract_artifacts_text_sync("some transcript")

    assert out["action_items"] == []
    cfg = captured["config"]
    assert cfg is not None
    assert cfg.temperature == 0.2
