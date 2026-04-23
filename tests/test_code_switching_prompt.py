"""
Tests for the code-switching preservation rules in Gemini prompts.

These are shape-tests over the prompt strings: we don't call Gemini (too slow,
costs money, flaky), we assert the prompt *tells* Gemini to keep English
technical terms verbatim.  If someone later rewrites the prompt and drops the
code-switching clause, these tests fail loudly.
"""
from app.services.summarizer import _SYSTEM_PROMPT, _REVISE_PROMPT_HEADER


def test_system_prompt_declares_code_switching_rule():
    """The main prompt must have a dedicated Code-Switching section."""
    assert "Code-Switching" in _SYSTEM_PROMPT


def test_system_prompt_lists_english_examples():
    """The prompt must teach Gemini with concrete English terms it should keep."""
    for term in ("React", "API", "JSON", "TCP", "useState"):
        assert term in _SYSTEM_PROMPT, f"expected example '{term}' in system prompt"


def test_system_prompt_forbids_translation():
    """Must instruct Gemini not to translate/transliterate technical terms."""
    assert "לא לתרגם" in _SYSTEM_PROMPT
    assert "לא לתעתק" in _SYSTEM_PROMPT


def test_system_prompt_covers_all_output_fields():
    """Rule must apply to summary + chapters + quiz (all fields)."""
    for field in ("summary", "chapters", "quiz"):
        assert field in _SYSTEM_PROMPT


def test_revise_prompt_preserves_english_terms():
    """Revise prompt must not let the critique-revise pass strip English terms."""
    assert "מונחים טכניים באנגלית" in _REVISE_PROMPT_HEADER
    assert "React" in _REVISE_PROMPT_HEADER
