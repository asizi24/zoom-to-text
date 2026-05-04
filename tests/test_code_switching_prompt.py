"""
Tests for the synthesis system prompt:
  1. Code-switching / technical-term preservation rules are present.
  2. Language configuration (_system_prompt respects settings.lecture_language).

These are shape-tests over the prompt strings — no Gemini calls are made.
If someone rewrites the prompt and drops key guardrails, these tests fail loudly.
"""
import pytest
from app.services import summarizer
from app.services.summarizer import _system_prompt, _REVISE_PROMPT_HEADER


# ── Technical-term preservation ───────────────────────────────────────────────

def test_system_prompt_declares_preserve_rule():
    """The main prompt must have a dedicated section about preserving technical terms."""
    assert "preserve technical terms" in _system_prompt()


def test_system_prompt_lists_english_examples():
    """The prompt must teach Gemini with concrete English terms it should keep."""
    for term in ("React", "API", "JSON", "TCP", "useState"):
        assert term in _system_prompt(), f"expected example '{term}' in system prompt"


def test_system_prompt_forbids_translation():
    """Must instruct Gemini not to translate technical terms."""
    prompt = _system_prompt()
    assert "do NOT translate" in prompt or "לא לתרגם" in prompt


def test_system_prompt_covers_all_output_fields():
    """Preservation rule must apply to summary + chapters + quiz."""
    prompt = _system_prompt()
    for field in ("summary", "chapters", "quiz"):
        assert field in prompt


def test_revise_prompt_preserves_english_terms():
    """Revise prompt must not let the critique-revise pass strip English terms."""
    assert "מונחים טכניים באנגלית" in _REVISE_PROMPT_HEADER
    assert "React" in _REVISE_PROMPT_HEADER


# ── Language configuration ────────────────────────────────────────────────────

def test_auto_language_instructs_detection(monkeypatch):
    """Default 'auto' mode should tell Gemini to detect and match the lecture language."""
    monkeypatch.setattr(summarizer.settings, "lecture_language", "auto")
    prompt = _system_prompt()
    assert "זהה את שפת" in prompt


def test_hebrew_language_forces_hebrew_output(monkeypatch):
    """Explicit 'he' mode should keep the Hebrew output instruction."""
    monkeypatch.setattr(summarizer.settings, "lecture_language", "he")
    prompt = _system_prompt()
    assert "בעברית" in prompt


def test_english_language_forces_english_output(monkeypatch):
    """Explicit 'en' mode should produce an English-specific instruction."""
    monkeypatch.setattr(summarizer.settings, "lecture_language", "en")
    prompt = _system_prompt()
    assert "en" in prompt


def test_different_language_settings_produce_different_prompts(monkeypatch):
    """Changing lecture_language must actually change the prompt."""
    monkeypatch.setattr(summarizer.settings, "lecture_language", "auto")
    auto_prompt = _system_prompt()
    monkeypatch.setattr(summarizer.settings, "lecture_language", "he")
    he_prompt = _system_prompt()
    assert auto_prompt != he_prompt
