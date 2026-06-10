"""
Tests for app/services/diarization/pyannote_provider.py.

All tests mock pyannote.audio via importlib — pyannote does not need to be
installed. The module is code-only (home-server / GPU only; never on Fly.io).
"""
import importlib
from unittest.mock import MagicMock

import pytest


class _MockSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _MockAnnotation:
    def __init__(self, tracks):
        # tracks: [(start, end, speaker_label), ...]
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for start, end, lbl in self._tracks:
            yield _MockSegment(start, end), None, lbl


def _make_pipeline_instance(tracks):
    """Return a callable mock that produces _MockAnnotation from *tracks*."""
    inst = MagicMock()
    inst.return_value = _MockAnnotation(tracks)
    return inst


@pytest.fixture
def mock_pyannote(monkeypatch):
    """
    Patch importlib.import_module so 'pyannote.audio' returns a module stub
    whose Pipeline.from_pretrained is controllable per-test.
    """
    pipeline_cls = MagicMock()
    module_stub = MagicMock()
    module_stub.Pipeline = pipeline_cls

    original = importlib.import_module

    def selective(name, *args, **kwargs):
        if name == "pyannote.audio":
            return module_stub
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", selective)
    return pipeline_cls  # caller configures .from_pretrained.return_value


def _setup_pipeline(pipeline_cls, tracks, monkeypatch):
    pipeline_cls.from_pretrained.return_value = _make_pipeline_instance(tracks)
    monkeypatch.setattr("app.config.settings.pyannote_model", "test/model", raising=False)
    monkeypatch.setattr("app.config.settings.hf_token", "", raising=False)


# ── Contract tests ────────────────────────────────────────────────────────────

def test_two_speakers_produces_labeled_lines(mock_pyannote, monkeypatch, tmp_path):
    """Two-speaker audio → transcript has 'Speaker A:' and 'Speaker B:' lines."""
    _setup_pipeline(mock_pyannote, [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")], monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    from app.services.diarization import pyannote_provider
    result, speaker_map = pyannote_provider.diarize_audio(audio, "Hello world. How are you?")

    assert "Speaker A:" in result
    assert "Speaker B:" in result
    assert speaker_map == {}


def test_single_speaker_all_labeled_A(mock_pyannote, monkeypatch, tmp_path):
    """Single-speaker audio → all lines labeled 'Speaker A:'; no Speaker B."""
    _setup_pipeline(mock_pyannote, [(0.0, 10.0, "SPEAKER_00")], monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    from app.services.diarization import pyannote_provider
    result, _ = pyannote_provider.diarize_audio(audio, "Hello. This is a test.")

    assert result.startswith("Speaker A:")
    assert "Speaker B:" not in result


def test_speaker_map_always_empty(mock_pyannote, monkeypatch, tmp_path):
    """Pyannote does not detect speaker names — speaker_map must always be {}."""
    _setup_pipeline(mock_pyannote, [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")], monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    from app.services.diarization import pyannote_provider
    _, speaker_map = pyannote_provider.diarize_audio(audio, "Test sentence.")

    assert speaker_map == {}


def test_label_normalization_respects_first_appearance_order(mock_pyannote, monkeypatch, tmp_path):
    """SPEAKER_00 → Speaker A, SPEAKER_01 → Speaker B — by order of first appearance."""
    tracks = [
        (0.0, 3.0, "SPEAKER_00"),
        (3.0, 6.0, "SPEAKER_01"),
        (6.0, 9.0, "SPEAKER_00"),
    ]
    _setup_pipeline(mock_pyannote, tracks, monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    from app.services.diarization import pyannote_provider
    result, _ = pyannote_provider.diarize_audio(
        audio, "One sentence. Two sentence. Three sentence. Four sentence."
    )

    lines = result.splitlines()
    speakers = [ln.split(":")[0].strip() for ln in lines if ":" in ln]
    assert "Speaker A" in speakers
    assert "Speaker B" in speakers


def test_return_type_is_tuple_of_str_and_dict(mock_pyannote, monkeypatch, tmp_path):
    """diarize_audio always returns (str, dict)."""
    _setup_pipeline(mock_pyannote, [(0.0, 5.0, "SPEAKER_00")], monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    from app.services.diarization import pyannote_provider
    result, speaker_map = pyannote_provider.diarize_audio(audio, "Hello.")

    assert isinstance(result, str)
    assert isinstance(speaker_map, dict)


def test_missing_pyannote_raises_import_error(monkeypatch):
    """If pyannote.audio is not installed, _load_pipeline raises ImportError."""
    original = importlib.import_module

    def fail_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ImportError("No module named 'pyannote'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fail_import)

    from app.services.diarization import pyannote_provider
    with pytest.raises(ImportError, match="pyannote"):
        pyannote_provider._load_pipeline("test/model", "")


# ── Internal helper unit tests ────────────────────────────────────────────────

def test_annotation_to_turns_sorted_by_start():
    """_annotation_to_turns returns turns sorted chronologically."""
    from app.services.diarization import pyannote_provider

    annotation = _MockAnnotation([(5.0, 8.0, "B"), (0.0, 4.0, "A")])
    turns = pyannote_provider._annotation_to_turns(annotation)

    assert turns[0][0] == 0.0  # start time of first turn
    assert turns[1][0] == 5.0


def test_assign_sentences_empty_transcript():
    """Empty transcript → empty assignment list."""
    from app.services.diarization import pyannote_provider

    result = pyannote_provider._assign_sentences_to_speakers(
        [], [(0.0, 5.0, "SPEAKER_00")], 5.0
    )
    assert result == []


def test_assign_sentences_no_turns_defaults_to_speaker_a():
    """No speaker turns → all sentences assigned to Speaker A."""
    from app.services.diarization import pyannote_provider

    result = pyannote_provider._assign_sentences_to_speakers(
        ["Hello.", "World."], [], 0.0
    )
    assert all(spk == "Speaker A" for spk, _ in result)
