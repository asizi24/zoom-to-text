"""
Phase B tests for Task 1.2 — diarization prompt + sync helper.

Covers parsing the Gemini diarization response, the prompt's bilingual
few-shots, and that the sync helper invokes the LLM with temperature=0.1.
"""
import json

import pytest


# ── Parser ────────────────────────────────────────────────────────────────────

def test_parse_diarization_full_payload():
    from app.services.summarizer import _parse_diarization_response

    raw = json.dumps({
        "speaker_count": 2,
        "speaker_map": {"Speaker A": "אסף", "Speaker B": "דן"},
        "diarized_transcript": "Speaker A: שלום, אני אסף.\nSpeaker B: היי, דן כאן.",
    }, ensure_ascii=False)

    diarized, smap = _parse_diarization_response(raw)
    assert "Speaker A:" in diarized
    assert "Speaker B:" in diarized
    assert smap == {"Speaker A": "אסף", "Speaker B": "דן"}


def test_parse_diarization_empty_speaker_map():
    """Single-lecturer transcripts return {} for speaker_map."""
    from app.services.summarizer import _parse_diarization_response

    raw = json.dumps({
        "speaker_count": 1,
        "speaker_map": {},
        "diarized_transcript": "Speaker A: welcome to today's lecture on Bell's theorem.",
    })

    diarized, smap = _parse_diarization_response(raw)
    assert "Bell" in diarized
    assert smap == {}


def test_parse_diarization_strips_markdown_fences():
    from app.services.summarizer import _parse_diarization_response

    raw = (
        "```json\n"
        + json.dumps({
            "speaker_count": 1,
            "speaker_map": {},
            "diarized_transcript": "Speaker A: hello",
        })
        + "\n```"
    )
    diarized, smap = _parse_diarization_response(raw)
    assert diarized == "Speaker A: hello"
    assert smap == {}


def test_parse_diarization_invalid_json_raises():
    from app.services.summarizer import _parse_diarization_response

    with pytest.raises((ValueError, RuntimeError)):
        _parse_diarization_response("not actual json {{")


def test_parse_diarization_missing_keys_default():
    """If the model omits speaker_map, parser defaults to {}."""
    from app.services.summarizer import _parse_diarization_response

    raw = json.dumps({"diarized_transcript": "Speaker A: x"})
    diarized, smap = _parse_diarization_response(raw)
    assert diarized == "Speaker A: x"
    assert smap == {}


# ── Prompt content ────────────────────────────────────────────────────────────

def test_diarization_prompt_has_bilingual_few_shots():
    from app.services.summarizer import _DIARIZATION_PROMPT

    has_hebrew = any("֐" <= ch <= "׿" for ch in _DIARIZATION_PROMPT)
    assert has_hebrew, "diarization prompt should include a Hebrew example"
    has_english_word = any(
        word.isascii() and word.isalpha() and len(word) >= 4
        for word in _DIARIZATION_PROMPT.split()
    )
    assert has_english_word, "diarization prompt should include an English example"


def test_diarization_prompt_specifies_required_keys():
    from app.services.summarizer import _DIARIZATION_PROMPT

    for key in ("speaker_count", "speaker_map", "diarized_transcript"):
        assert key in _DIARIZATION_PROMPT, f"prompt missing {key}"


def test_diarization_temperature_is_01():
    from app.services.summarizer import _DIARIZATION_TEMPERATURE
    assert _DIARIZATION_TEMPERATURE == 0.1


def test_diarization_config_forces_json_response_mime_type():
    """
    Force Gemini into strict-JSON mode so the parser doesn't see malformed
    payloads like the 2026-05-01 prod failure
    ("Expecting ',' delimiter: line 12 column 4").
    """
    from app.services.summarizer import _diarization_config

    cfg = _diarization_config()
    assert cfg.response_mime_type == "application/json"


# ── Sync helper ───────────────────────────────────────────────────────────────

def test_diarize_transcript_sync_uses_temperature_01(monkeypatch):
    """The sync helper must invoke the LLM with temperature=0.1."""
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
        return FakeResponse(json.dumps({
            "speaker_count": 1,
            "speaker_map": {},
            "diarized_transcript": "Speaker A: x",
        }))

    monkeypatch.setattr(s, "_generate_with_retry", fake_generate)
    monkeypatch.setattr(s, "_get_client", lambda: object())

    diarized, smap = s._diarize_transcript_sync("transcript text")

    assert diarized == "Speaker A: x"
    assert smap == {}
    assert captured["config"].temperature == 0.1
