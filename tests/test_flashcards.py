"""
Tests for Feature 6 — Flashcards + Anki export.

Gemini itself is NOT hit in tests (cost, flakiness). We monkeypatch the client
to return canned JSON and assert:
  - Parser handles clean JSON, fenced JSON, and thinking-preamble JSON
  - generate_flashcards returns the expected Flashcard objects
  - anki_export.create_apkg produces a non-trivial binary that looks like a zip
    (apkg is a zip containing sqlite + media), with the expected note count
  - CSV export is UTF-8 with BOM and has the right header
  - Deck IDs are deterministic per task_id (re-import updates, not duplicates)
"""
import io
import json
import sqlite3
import zipfile

import pytest

from app.models import Flashcard
from app.services import anki_export, summarizer


# ── Parser tests ──────────────────────────────────────────────────────────────

CANNED_JSON = {
    "flashcards": [
        {"front": "מה תפקיד useState?", "back": "מחזיר [state, setState]", "tags": ["React"]},
        {"front": "What is TCP?", "back": "Reliable delivery protocol.", "tags": ["networking"]},
    ]
}


def test_parse_clean_json():
    cards = summarizer._parse_flashcards_response(json.dumps(CANNED_JSON, ensure_ascii=False))
    assert len(cards) == 2
    assert cards[0].front.startswith("מה תפקיד")
    assert "React" in cards[0].tags


def test_parse_fenced_json():
    raw = "```json\n" + json.dumps(CANNED_JSON, ensure_ascii=False) + "\n```"
    cards = summarizer._parse_flashcards_response(raw)
    assert len(cards) == 2


def test_parse_thinking_preamble():
    """Gemini 2.5 sometimes emits {reasoning...} before the JSON."""
    raw = 'Thinking: {this is not the answer}\n' + json.dumps(CANNED_JSON, ensure_ascii=False)
    cards = summarizer._parse_flashcards_response(raw)
    assert len(cards) == 2


def test_parse_drops_empty_cards():
    raw = json.dumps({
        "flashcards": [
            {"front": "", "back": "orphan back", "tags": []},
            {"front": "good", "back": "good back", "tags": []},
            {"front": "no back", "back": "", "tags": []},
        ]
    })
    assert len(summarizer._parse_flashcards_response(raw)) == 1


def test_parse_returns_empty_on_garbage():
    assert summarizer._parse_flashcards_response("totally not json") == []


# ── Generation wiring ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_flashcards_returns_cards(monkeypatch):
    """Monkeypatch the sync helper so we don't hit Gemini."""
    def fake_sync(summary, transcript):
        return [
            Flashcard(front="a", back="b", tags=["t"]),
            Flashcard(front="c", back="d", tags=[]),
        ]
    monkeypatch.setattr(summarizer, "_generate_flashcards_sync", fake_sync)
    cards = await summarizer.generate_flashcards("some summary", "transcript")
    assert len(cards) == 2
    assert cards[0].tags == ["t"]


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
