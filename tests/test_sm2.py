"""Tests for the SM-2 spaced repetition algorithm (Batch B3).

Pure-function tests: no DB, no client. Verifies the formulas, the reset on
'again', the 1d→6d→EF-driven interval progression, the EF floor, and the
ISO timestamps.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.services import sm2


_FIXED_NOW = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)


def test_default_state_is_due():
    assert sm2.is_due(sm2.ReviewState()) is True


def test_good_first_review_sets_1_day_interval():
    s = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    assert s.repetitions == 1
    assert s.interval == 1
    assert s.due_at == (_FIXED_NOW + timedelta(days=1)).isoformat()
    assert s.last_reviewed_at == _FIXED_NOW.isoformat()


def test_good_second_review_sets_6_day_interval():
    s1 = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    s2 = sm2.apply_review(s1, "good", now=_FIXED_NOW)
    assert s2.repetitions == 2
    assert s2.interval == 6


def test_good_third_review_multiplies_by_ef():
    s = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    s = sm2.apply_review(s, "good", now=_FIXED_NOW)
    s3 = sm2.apply_review(s, "good", now=_FIXED_NOW)
    # interval=6 * EF, EF after three q=4 reviews ≈ 2.5+3*(-0.02) ≈ 2.44
    assert s3.repetitions == 3
    assert s3.interval >= 12  # 6 * ~2.44 = ~14.6 → round to 15
    assert s3.interval <= 16


def test_again_resets_repetitions_and_interval():
    s = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    s = sm2.apply_review(s, "good", now=_FIXED_NOW)
    failed = sm2.apply_review(s, "again", now=_FIXED_NOW)
    assert failed.repetitions == 0
    assert failed.interval == 1
    # EF still drops but is clamped at 1.3
    assert failed.easiness >= sm2.MIN_EF


def test_ef_floor_after_many_again():
    s = sm2.ReviewState()
    for _ in range(20):
        s = sm2.apply_review(s, "again", now=_FIXED_NOW)
    assert s.easiness == pytest.approx(sm2.MIN_EF)


def test_easy_grows_ef():
    s = sm2.apply_review(sm2.ReviewState(), "easy", now=_FIXED_NOW)
    # +0.1 from default 2.5
    assert s.easiness == pytest.approx(2.6, abs=1e-3)


def test_hard_shrinks_ef_slightly():
    s = sm2.apply_review(sm2.ReviewState(), "hard", now=_FIXED_NOW)
    # q=3 → 0.1 - 2*(0.08 + 2*0.02) = 0.1 - 0.24 = -0.14
    assert s.easiness == pytest.approx(2.36, abs=1e-3)


def test_unknown_grade_raises():
    with pytest.raises(ValueError):
        sm2.apply_review(sm2.ReviewState(), "maybe")


def test_is_due_after_apply_is_false_immediately():
    s = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    assert sm2.is_due(s, now=_FIXED_NOW) is False


def test_is_due_after_interval_passes_is_true():
    s = sm2.apply_review(sm2.ReviewState(), "good", now=_FIXED_NOW)
    later = _FIXED_NOW + timedelta(days=2)
    assert sm2.is_due(s, now=later) is True
