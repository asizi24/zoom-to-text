"""
Anki .apkg export for Flashcards.

genanki builds SQLite-backed .apkg archives that Anki imports natively.
Every lesson gets a deterministic deck_id (derived from the task_id) so a
student who re-imports the file updates existing cards instead of duplicating
them — this is the intended genanki behavior when deck_id + note guid are
stable.

CSV export is a simple alternative for users who prefer not to install Anki.
"""
import csv
import hashlib
import io
import logging
from typing import Iterable

import genanki

from app.models import Flashcard

logger = logging.getLogger(__name__)


# Stable model_id for our card template. genanki requires a 10-digit int.
# Changing this invalidates all previously-imported cards, so treat as a schema.
_MODEL_ID = 1_610_000_001
_MODEL = genanki.Model(
    _MODEL_ID,
    "ZoomToText Flashcard",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": '<div dir="rtl" style="font-family:Arial,sans-serif;font-size:20px;">{{Front}}</div>',
            "afmt": (
                '<div dir="rtl" style="font-family:Arial,sans-serif;font-size:20px;">'
                "{{FrontSide}}<hr id=answer>{{Back}}"
                "</div>"
            ),
        }
    ],
    css=(
        ".card { font-family:Arial,sans-serif; font-size:20px; text-align:right; "
        "color:#111; background:#fafafa; direction:rtl; }"
    ),
)


def _deck_id_for_task(task_id: str) -> int:
    """
    Derive a stable positive 10-digit int deck_id from task_id.
    Anki deck_ids must be int; stable mapping means re-import updates, not duplicates.
    """
    h = hashlib.sha256(task_id.encode("utf-8")).digest()
    # Use the first 4 bytes as an unsigned int, stay under 2^31 for safety
    raw = int.from_bytes(h[:4], "big")
    return 1_000_000_000 + (raw % 1_000_000_000)


def _guid_for_card(task_id: str, card: Flashcard) -> str:
    """
    Stable GUID per (task, card.front) so re-imports update instead of duplicating.
    genanki exposes genanki.guid_for to build GUIDs from arbitrary input.
    """
    return genanki.guid_for(task_id, card.front)


def create_apkg(
    flashcards: Iterable[Flashcard],
    deck_name: str,
    task_id: str,
) -> bytes:
    """
    Build an in-memory .apkg for the given flashcards and return the bytes.
    Callers are responsible for setting Content-Disposition on the response.
    """
    cards = list(flashcards)
    deck = genanki.Deck(_deck_id_for_task(task_id), deck_name)

    for card in cards:
        note = genanki.Note(
            model=_MODEL,
            fields=[card.front, card.back],
            tags=list(card.tags or []),
            guid=_guid_for_card(task_id, card),
        )
        deck.add_note(note)

    package = genanki.Package(deck)
    buf = io.BytesIO()
    package.write_to_file(buf)
    buf.seek(0)
    data = buf.read()
    logger.info(
        f"Built Anki deck '{deck_name}' — {len(cards)} cards, "
        f"{len(data):,} bytes, deck_id={deck.deck_id}"
    )
    return data


def create_csv(flashcards: Iterable[Flashcard]) -> bytes:
    """
    Build a UTF-8 CSV (with BOM so Excel opens Hebrew correctly).
    Columns: front, back, tags (tags joined by spaces — Anki's default).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["front", "back", "tags"])
    for card in flashcards:
        writer.writerow([card.front, card.back, " ".join(card.tags or [])])
    # BOM so Excel (and other Windows tooling) detects UTF-8 automatically
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
