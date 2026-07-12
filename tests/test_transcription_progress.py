"""Unit tests for live transcription progress reporting.

Transcription is the longest pipeline step (30-60 min for a lecture on CPU);
_transcribe_sync must report how far into the recording it has reached so the
task's progress bar moves instead of sitting frozen at 50%.
"""
import pytest

from app.services import transcriber


class _FakeSeg:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    duration = 100.0
    language = "he"


class _FakeModel:
    def transcribe(self, *args, **kwargs):
        segs = [_FakeSeg(i * 10.0, i * 10.0 + 10.0, f"seg{i}") for i in range(10)]
        return iter(segs), _FakeInfo()


def test_transcribe_sync_reports_audio_progress(monkeypatch):
    """progress_cb receives monotonically increasing fractions ending at 1.0."""
    # The callback is throttled to one call per 5 real seconds; advance a fake
    # clock by 10s per call so every segment passes the throttle in the test.
    t = [0.0]

    def fake_time():
        t[0] += 10.0
        return t[0]

    monkeypatch.setattr(transcriber.time, "time", fake_time)

    fractions: list[float] = []
    text, lang = transcriber._transcribe_sync(
        _FakeModel(), "x.mp3", "he", segment_cb=None, progress_cb=fractions.append
    )

    assert lang == "he"
    assert "seg0" in text and "seg9" in text
    assert fractions, "progress callback never fired"
    assert fractions == sorted(fractions), "progress must be monotonic"
    assert fractions[-1] == pytest.approx(1.0)


def test_transcribe_sync_survives_zero_duration(monkeypatch):
    """A missing/zero duration must disable progress, not divide by zero."""

    class _NoDurationInfo:
        duration = 0
        language = "he"

    class _Model:
        def transcribe(self, *args, **kwargs):
            return iter([_FakeSeg(0, 5, "hello")]), _NoDurationInfo()

    calls: list[float] = []
    text, lang = transcriber._transcribe_sync(
        _Model(), "x.mp3", "he", segment_cb=None, progress_cb=calls.append
    )
    assert "hello" in text
    assert calls == []
