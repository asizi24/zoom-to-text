"""
Tests for audio_preprocessor fail-fast validation AND preprocess() edge cases.

Covers four edge cases that waste resources if allowed into the pipeline:
  1. Empty file (0 bytes)
  2. Invalid / unrecognizable format (text file with .mp3 extension)
  3. Audio too short for a lecture (< MIN_LECTURE_SECONDS)
  4. Completely silent audio (mean volume far below threshold)

Plus one happy-path validate_audio test to confirm valid inputs are accepted.

preprocess() edge cases:
  5. Short recording (≤ CHUNK_SECONDS) — stays as a single chunk, no splitting
  6. All silence stripped (stripped file is very short) — falls back gracefully
     to the original file rather than crashing
  7. Recording already within one chunk — _split_chunks returns a single copy
     without looping through the multi-chunk path
  8. Multi-chunk recording — verifies the correct number of chunks is returned
"""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.audio_preprocessor import (
    CHUNK_SECONDS,
    MIN_LECTURE_SECONDS,
    SILENT_THRESHOLD_DB,
    cleanup_chunks,
    preprocess,
    validate_audio,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def empty_file(tmp_path):
    """Zero-byte file."""
    p = tmp_path / "empty.mp3"
    p.touch()
    return str(p)


@pytest.fixture()
def nonempty_file(tmp_path):
    """Non-empty file — passes the size check so format/duration/silence checks run."""
    p = tmp_path / "fake.mp3"
    p.write_bytes(b"\xff\xfb" + b"\x00" * 1024)   # plausible-looking binary blob
    return str(p)


# ── validate_audio edge cases ─────────────────────────────────────────────────

def test_empty_file_raises(empty_file):
    with pytest.raises(ValueError, match="[Ee]mpty"):
        validate_audio(empty_file)


def test_invalid_format_raises(nonempty_file):
    """ffprobe failure (CalledProcessError) → descriptive format error."""
    with patch(
        "app.services.audio_preprocessor._get_duration",
        side_effect=subprocess.CalledProcessError(1, "ffprobe"),
    ):
        with pytest.raises(ValueError, match="[Ii]nvalid|format|unrecognizable"):
            validate_audio(nonempty_file)


def test_too_short_raises(nonempty_file):
    with patch("app.services.audio_preprocessor._get_duration", return_value=10.0):
        with pytest.raises(ValueError, match="[Ss]hort"):
            validate_audio(nonempty_file)


def test_silent_file_raises(nonempty_file):
    with patch("app.services.audio_preprocessor._get_duration", return_value=600.0):
        with patch(
            "app.services.audio_preprocessor._get_mean_volume", return_value=-91.0
        ):
            with pytest.raises(ValueError, match="[Ss]ilent"):
                validate_audio(nonempty_file)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_audio_passes(nonempty_file):
    """A long, non-silent file must not raise."""
    with patch("app.services.audio_preprocessor._get_duration", return_value=3600.0):
        with patch(
            "app.services.audio_preprocessor._get_mean_volume", return_value=-20.0
        ):
            validate_audio(nonempty_file)  # no exception


# ── preprocess() edge cases ───────────────────────────────────────────────────

def _make_fake_temp(suffix=".mp3"):
    """Create a real temp file on disk so Path.unlink() doesn't fail."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="zoom_test_")
    os.close(fd)
    Path(path).write_bytes(b"\xff\xfb" + b"\x00" * 512)
    return path


def test_preprocess_short_recording_returns_single_chunk(nonempty_file):
    """
    A recording shorter than CHUNK_SECONDS (e.g. 4 minutes = 240 s) must
    come back as exactly one chunk — _split_chunks must NOT attempt multi-chunk
    splitting and instead calls _copy_to_temp.
    """
    fake_chunk = _make_fake_temp()
    try:
        with patch("app.services.audio_preprocessor._get_duration", return_value=240.0), \
             patch("app.services.audio_preprocessor._get_mean_volume", return_value=-20.0), \
             patch("app.services.audio_preprocessor._remove_silence", return_value=fake_chunk), \
             patch("app.services.audio_preprocessor._copy_to_temp", return_value=fake_chunk) as mock_copy:

            chunks = preprocess(nonempty_file)

        assert len(chunks) == 1
        # _copy_to_temp is called by _split_chunks when duration <= CHUNK_SECONDS
        mock_copy.assert_called_once()
    finally:
        Path(fake_chunk).unlink(missing_ok=True)


def test_preprocess_no_chunking_needed_when_within_limit(nonempty_file):
    """
    A recording exactly at CHUNK_SECONDS must NOT produce multiple chunks.
    The single-chunk fast path in _split_chunks must be taken.
    """
    fake_chunk = _make_fake_temp()
    try:
        with patch("app.services.audio_preprocessor._get_duration", return_value=float(CHUNK_SECONDS)), \
             patch("app.services.audio_preprocessor._get_mean_volume", return_value=-20.0), \
             patch("app.services.audio_preprocessor._remove_silence", return_value=fake_chunk), \
             patch("app.services.audio_preprocessor._copy_to_temp", return_value=fake_chunk):

            chunks = preprocess(nonempty_file)

        assert len(chunks) == 1
    finally:
        Path(fake_chunk).unlink(missing_ok=True)


def test_preprocess_multi_chunk_recording_returns_correct_count(nonempty_file):
    """
    A 30-minute recording (1800 s) with CHUNK_SECONDS=780 (13 min) should
    produce 3 chunks: 0-780, 780-1560, 1560-1800.
    """
    # Build three fake temp files — one per expected chunk
    fake_chunks = [_make_fake_temp() for _ in range(3)]
    try:
        # validate_audio passes, _remove_silence returns the same file
        # _get_duration is called multiple times: once for validation, once for
        # the original duration, once for the stripped file, once inside _split_chunks.
        # We give all calls the same duration (1800 s) for simplicity.
        duration = 1800.0
        call_count = [0]

        def duration_side_effect(path):
            call_count[0] += 1
            return duration

        # Simulate subprocess.run for chunking — write fake content to the output path
        original_subprocess_run = subprocess.run

        def fake_subprocess_run(cmd, **kwargs):
            if "ffmpeg" in cmd[0] and "-ss" in cmd:
                # Chunking call — write some bytes to the output path (-y flag, path is last arg)
                out_path = cmd[-1]
                Path(out_path).write_bytes(b"\xff\xfb" + b"\x00" * 64)
                mock = MagicMock()
                mock.returncode = 0
                return mock
            # For other calls (silence removal, copy), delegate normally or return mock
            mock = MagicMock()
            mock.returncode = 0
            return mock

        stripped_file = _make_fake_temp()
        try:
            with patch("app.services.audio_preprocessor._get_duration", side_effect=duration_side_effect), \
                 patch("app.services.audio_preprocessor._get_mean_volume", return_value=-20.0), \
                 patch("app.services.audio_preprocessor._remove_silence", return_value=stripped_file), \
                 patch("subprocess.run", side_effect=fake_subprocess_run):

                chunks = preprocess(nonempty_file)

            # 1800 / 780 = 2 full chunks + 1 remainder = 3 chunks
            assert len(chunks) == 3
        finally:
            Path(stripped_file).unlink(missing_ok=True)
    finally:
        for c in fake_chunks:
            Path(c).unlink(missing_ok=True)


def test_preprocess_fallback_when_silence_removal_fails(nonempty_file):
    """
    When _remove_silence raises an exception (e.g. ffmpeg crash), preprocess()
    must NOT propagate the exception — it must fall back gracefully and return
    the original file as a single chunk via _copy_to_temp.
    """
    fake_chunk = _make_fake_temp()
    try:
        with patch("app.services.audio_preprocessor._get_duration", return_value=600.0), \
             patch("app.services.audio_preprocessor._get_mean_volume", return_value=-20.0), \
             patch(
                 "app.services.audio_preprocessor._remove_silence",
                 side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
             ), \
             patch("app.services.audio_preprocessor._copy_to_temp", return_value=fake_chunk) as mock_copy:

            chunks = preprocess(nonempty_file)

        # Fallback path: _copy_to_temp is called with the original file
        mock_copy.assert_called_once_with(nonempty_file)
        assert len(chunks) == 1
        assert chunks[0] == fake_chunk
    finally:
        Path(fake_chunk).unlink(missing_ok=True)


def test_preprocess_raises_value_error_for_silent_audio(nonempty_file):
    """
    validate_audio is called first inside preprocess(). A completely silent
    file must still raise ValueError — the fallback only covers post-validation
    processing failures, not pre-flight rejections.
    """
    with patch("app.services.audio_preprocessor._get_duration", return_value=600.0), \
         patch("app.services.audio_preprocessor._get_mean_volume", return_value=SILENT_THRESHOLD_DB - 1.0):
        with pytest.raises(ValueError, match="[Ss]ilent"):
            preprocess(nonempty_file)


def test_preprocess_raises_value_error_for_too_short_audio(nonempty_file):
    """
    Recordings shorter than MIN_LECTURE_SECONDS must be rejected even when
    called via the top-level preprocess() entry point.
    """
    with patch("app.services.audio_preprocessor._get_duration", return_value=MIN_LECTURE_SECONDS - 1.0):
        with pytest.raises(ValueError, match="[Ss]hort"):
            preprocess(nonempty_file)


def test_cleanup_chunks_deletes_files(tmp_path):
    """cleanup_chunks() must delete all files in the list."""
    files = []
    for i in range(3):
        p = tmp_path / f"chunk_{i}.mp3"
        p.write_bytes(b"data")
        files.append(str(p))

    cleanup_chunks(files)

    for f in files:
        assert not Path(f).exists()


def test_cleanup_chunks_tolerates_missing_files():
    """cleanup_chunks() must not raise when files are already gone."""
    fake_paths = ["/tmp/does_not_exist_abc123.mp3"]
    cleanup_chunks(fake_paths)  # must not raise
