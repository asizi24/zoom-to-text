"""Tests for the topic-mastery dashboard (Batch B4).

GET /api/mastery aggregates a user's flashcard SM-2 state by tag.

Verifies:
  • empty library returns empty arrays and zero totals
  • tags from all cards bucket correctly
  • a card with repetitions>=3 counts as mastered
  • due cards (due_at <= now) count as due
  • cross-user isolation: only the calling user's cards count
"""
from datetime import datetime, timedelta, timezone

import pytest

from app import state
from app.api import deps
from app.main import app
from app.models import Flashcard, LessonResult


def _override_user(uid: str) -> None:
    async def _user():
        return uid
    app.dependency_overrides[deps.get_current_user] = _user


def _clear_override() -> None:
    app.dependency_overrides.pop(deps.get_current_user, None)


async def _seed_task(task_id: str, user_id: str, cards: list[Flashcard]):
    await state.create_task(task_id, f"https://x/{task_id}", user_id=user_id)
    await state.complete_task(task_id, LessonResult(summary="x", flashcards=cards))


async def test_mastery_empty_when_no_lectures(client):
    _override_user("m-user-empty")
    try:
        resp = client.get("/api/mastery")
        assert resp.status_code == 200
        body = resp.json()
        assert body["by_tag"] == []
        assert body["totals"] == {"cards": 0, "mastered": 0, "due": 0}
    finally:
        _clear_override()


async def test_mastery_buckets_cards_by_tag(client):
    cards = [
        Flashcard(front="q1", back="a1", tags=["א", "ב"]),
        Flashcard(front="q2", back="a2", tags=["ב"]),
        Flashcard(front="q3", back="a3", tags=[]),  # falls under "ללא תגית"
    ]
    await _seed_task("mast-1", "m-user-1", cards)

    _override_user("m-user-1")
    try:
        resp = client.get("/api/mastery")
        assert resp.status_code == 200
        body = resp.json()
        assert body["totals"]["cards"] == 3
        buckets = {b["tag"]: b for b in body["by_tag"]}
        assert buckets["א"]["total"] == 1
        assert buckets["ב"]["total"] == 2
        assert buckets["ללא תגית"]["total"] == 1
    finally:
        _clear_override()


async def test_mastery_counts_mastered_when_repetitions_at_least_3(client):
    cards = [Flashcard(front="q", back="a", tags=["טופוגרפיה"])]
    await _seed_task("mast-2", "m-user-2", cards)

    # Save a review row with repetitions=3 → mastered
    future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
    await state.save_card_review(
        user_id="m-user-2",
        task_id="mast-2",
        card_index=0,
        easiness=2.5,
        interval=10,
        repetitions=3,
        last_reviewed_at=datetime.now(timezone.utc).isoformat(),
        due_at=future,
    )

    _override_user("m-user-2")
    try:
        resp = client.get("/api/mastery")
        body = resp.json()
        assert body["totals"]["mastered"] == 1
        bucket = next(b for b in body["by_tag"] if b["tag"] == "טופוגרפיה")
        assert bucket["mastered"] == 1
        assert bucket["mastery_pct"] == 100
    finally:
        _clear_override()


async def test_mastery_counts_due_when_due_at_is_in_the_past(client):
    cards = [Flashcard(front="q", back="a", tags=["שפה"])]
    await _seed_task("mast-3", "m-user-3", cards)

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    await state.save_card_review(
        user_id="m-user-3",
        task_id="mast-3",
        card_index=0,
        easiness=2.5,
        interval=1,
        repetitions=1,
        last_reviewed_at=past,
        due_at=past,
    )

    _override_user("m-user-3")
    try:
        body = client.get("/api/mastery").json()
        assert body["totals"]["due"] == 1
        bucket = next(b for b in body["by_tag"] if b["tag"] == "שפה")
        assert bucket["due"] == 1
        # repetitions=1 < 3 — not mastered
        assert bucket["mastered"] == 0
    finally:
        _clear_override()


async def test_mastery_isolates_users(client):
    await _seed_task("mast-A", "m-user-A", [Flashcard(front="q", back="a", tags=["A"])])
    await _seed_task("mast-B", "m-user-B", [Flashcard(front="q", back="a", tags=["B"])])

    _override_user("m-user-B")
    try:
        body = client.get("/api/mastery").json()
        # Only B's cards must appear
        tags = {b["tag"] for b in body["by_tag"]}
        assert tags == {"B"}
        assert body["totals"]["cards"] == 1
    finally:
        _clear_override()
