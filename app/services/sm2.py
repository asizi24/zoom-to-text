"""SuperMemo-2 (SM-2) spaced repetition algorithm.

Reference: https://en.wikipedia.org/wiki/SuperMemo

We use a slightly simplified 4-grade UI mapping (again/hard/good/easy →
0/3/4/5) instead of full 0-5 — that's the standard Anki adaptation. The
core math stays identical:

    EF' = max(1.3, EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02)))
    If q < 3:           reset (repetitions=0, interval=1 day)
    Else if rep == 0:   interval=1 day
    Else if rep == 1:   interval=6 days
    Else:               interval=round(prev_interval * EF')

`due_at` is computed by adding `interval` days to the review time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Allowed UI grades and the SM-2 quality each maps to
GRADE_TO_Q = {
    "again": 0,
    "hard": 3,
    "good": 4,
    "easy": 5,
}

DEFAULT_EF = 2.5
MIN_EF = 1.3


@dataclass
class ReviewState:
    """SM-2 state for one flashcard for one user."""

    easiness: float = DEFAULT_EF
    interval: int = 0          # days until next review
    repetitions: int = 0       # successful reviews in a row
    last_reviewed_at: str | None = None  # ISO UTC
    due_at: str | None = None  # ISO UTC


def apply_review(state: ReviewState, grade: str, *, now: datetime | None = None) -> ReviewState:
    """Apply one review with a grade ('again' | 'hard' | 'good' | 'easy').

    Returns a NEW ReviewState — never mutates input.
    """
    if grade not in GRADE_TO_Q:
        raise ValueError(f"unknown SM-2 grade: {grade!r}")
    q = GRADE_TO_Q[grade]
    now = now or datetime.now(timezone.utc)

    new_ef = state.easiness + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if new_ef < MIN_EF:
        new_ef = MIN_EF

    if q < 3:
        new_reps = 0
        new_interval = 1
    else:
        new_reps = state.repetitions + 1
        if new_reps == 1:
            new_interval = 1
        elif new_reps == 2:
            new_interval = 6
        else:
            new_interval = max(1, round(state.interval * new_ef))

    due = now + timedelta(days=new_interval)
    return ReviewState(
        easiness=round(new_ef, 4),
        interval=new_interval,
        repetitions=new_reps,
        last_reviewed_at=now.isoformat(),
        due_at=due.isoformat(),
    )


def is_due(state: ReviewState, *, now: datetime | None = None) -> bool:
    """True if the card has never been reviewed OR its `due_at` has passed."""
    if not state.due_at:
        return True
    now = now or datetime.now(timezone.utc)
    try:
        due = datetime.fromisoformat(state.due_at)
    except ValueError:
        return True
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    return due <= now
