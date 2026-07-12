"""
Tests for Feature 6 — Flashcards + Anki export.

Gemini itself is NOT hit in tests (cost, flakiness). Since the structured-output
refactor there is no raw-JSON parsing to test — Gemini's constrained decoding
returns a parsed _FlashcardsSchema. We monkeypatch _generate_structured and assert:
  - generate_flashcards maps parsed cards to Flashcard objects
  - empty/invalid cards are dropped, failures return an empty list (soft-fail)
  - anki_export.create_apkg produces a non-trivial binary that looks like a zip
    (apkg is a zip containing sqlite + media), with the expected note count
  - CSV export is UTF-8 with BOM and has the right header
  - Deck IDs are deterministic per task_id (re-import updates, not duplicates)
"""
import io
import sqlite3
import zipfile

import pytest

from app.models import Flashcard
from app.services import anki_export, summarizer
from app.services.errors import PipelineError
from app.services.summarizer import _FlashcardSchema, _FlashcardsSchema


# ── Generation wiring ─────────────────────────────────────────────────────────

def _patch_structured(monkeypatch, parsed):
    async def fake_structured(contents, config, timeout=None, **kwargs):
        if isinstance(parsed, Exception):
            raise parsed
        return parsed
    monkeypatch.setattr(summarizer, "_generate_structured", fake_structured)


@pytest.mark.asyncio
async def test_generate_flashcards_returns_cards(monkeypatch):
    _patch_structured(monkeypatch, _FlashcardsSchema(flashcards=[
        _FlashcardSchema(front="מה תפקיד useState?", back="מחזיר [state, setState]", tags=["React"]),
        _FlashcardSchema(front="What is TCP?", back="Reliable delivery protocol.", tags=[]),
    ]))
    cards = await summarizer.generate_flashcards("some summary", "transcript")
    assert len(cards) == 2
    assert cards[0].front.startswith("מה תפקיד")
    assert cards[0].tags == ["React"]


@pytest.mark.asyncio
async def test_generate_flashcards_drops_empty_cards(monkeypatch):
    _patch_structured(monkeypatch, _FlashcardsSchema(flashcards=[
        _FlashcardSchema(front="", back="orphan back", tags=[]),
        _FlashcardSchema(front="good", back="good back", tags=[" React "]),
        _FlashcardSchema(front="no back", back="  ", tags=[]),
    ]))
    cards = await summarizer.generate_flashcards("some summary")
    assert len(cards) == 1
    assert cards[0].tags == ["React"]  # tags are stripped


@pytest.mark.asyncio
async def test_generate_flashcards_soft_fails_on_gemini_error(monkeypatch):
    """Flashcards are a bonus step — a Gemini failure returns [] instead of raising."""
    _patch_structured(monkeypatch, PipelineError("קצב", detail="429"))
    cards = await summarizer.generate_flashcards("some summary")
    assert cards == []


@pytest.mark.asyncio
async def test_generate_flashcards_empty_summary_short_circuits():
    cards = await summarizer.generate_flashcards("", "x")
    assert cards == []


# ── Anki export ───────────────────────────────────────────────────────────────

_SAMPLE_CARDS = [
    Flashcard(front="מה תפקיד useState?", back="מחזיר [state, setState]", tags=["React"]),
    Flashcard(front="What is TCP?", back="Reliable delivery protocol.", tags=["networking"]),
    Flashcard(front="מהי ההבדל בין let ל-const?", back="const אינו ניתן לשיוך מחדש.", tags=["JS"]),
]


def test_apkg_is_valid_zip_with_sqlite():
    data = anki_export.create_apkg(_SAMPLE_CARDS, "Unit Test Deck", task_id="t-1")
    assert len(data) > 1000  # at least a kB
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    # genanki packages contain 'collection.anki2' — the SQLite DB
    assert any(n.startswith("collection.anki2") for n in names)


def test_apkg_contains_all_notes():
    """Open the embedded SQLite DB and count notes — should equal our card count."""
    data = anki_export.create_apkg(_SAMPLE_CARDS, "X", task_id="t-2")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Extract the DB to /tmp-ish so sqlite3 can open it
        with zf.open("collection.anki2") as src:
            raw = src.read()
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".anki2") as tf:
        tf.write(raw)
        tf_path = tf.name
    try:
        conn = sqlite3.connect(tf_path)
        count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        conn.close()
        assert count == len(_SAMPLE_CARDS)
    finally:
        os.unlink(tf_path)


def test_deck_id_is_deterministic_per_task_id():
    a = anki_export._deck_id_for_task("task-xyz")
    b = anki_export._deck_id_for_task("task-xyz")
    c = anki_export._deck_id_for_task("task-other")
    assert a == b
    assert a != c
    assert 10**9 <= a < 2 * 10**9  # 10 digits, fits in positive int31


def test_csv_has_utf8_bom_and_header():
    data = anki_export.create_csv(_SAMPLE_CARDS)
    # UTF-8 BOM
    assert data[:3] == b"\xef\xbb\xbf"
    text = data[3:].decode("utf-8")
    lines = text.splitlines()
    assert lines[0] == '"front","back","tags"'
    assert len(lines) == 1 + len(_SAMPLE_CARDS)
    # Hebrew characters round-trip
    assert "useState" in text
    assert "מה תפקיד" in text


def test_csv_joins_tags_with_space():
    card = Flashcard(front="q", back="a", tags=["one", "two", "three"])
    data = anki_export.create_csv([card])
    assert b"one two three" in data
