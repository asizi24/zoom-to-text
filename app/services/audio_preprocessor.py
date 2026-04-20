"""
Audio preprocessor for Whisper transcription.

Two steps applied before transcription:
  1. Silence removal — strips segments below -40 dBFS using ffmpeg's silenceremove
     filter, so Whisper doesn't waste time on dead air between topics.
  2. Chunking — splits the result into ≤13-minute pieces. This is the practical
     accuracy/memory sweet spot discovered through use.

All operations use ffmpeg (already available via yt-dlp). No extra dependencies.

The caller always receives NEW temp files it owns and must delete. The original
audio path is never modified.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path

# Maximum time (seconds) for any single ffmpeg/ffprobe call.
# A 10-hour recording at fast read speeds should finish in < 5 minutes.
_FFMPEG_TIMEOUT = 600

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 13 * 60   # 13 minutes per chunk
SILENCE_DB    = -40        # dBFS threshold — below this is treated as silence
SILENCE_MIN_S = 1.0        # minimum silence duration to remove (seconds)
PAD_S         = 0.2        # seconds of silence to keep around speech (natural transitions)


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(audio_path: str) -> list[str]:
    """
    Prepare an audio file for Whisper transcription.

    Steps:
      1. Remove silence (segments below SILENCE_DB for ≥ SILENCE_MIN_S seconds)
      2. Split into CHUNK_SECONDS-long segments

    Returns a list of new temp file paths. The caller must delete all of them.
    If any step fails, falls back gracefully so transcription can still proceed.
    """
    try:
        original_duration = _get_duration(audio_path)
        logger.info(
            f"[Preprocessor] Input: {original_duration:.0f}s "
            f"({original_duration/60:.1f} min) — removing silence..."
        )

        stripped_path = _remove_silence(audio_path)

        stripped_duration = _get_duration(stripped_path)
        saved_s = original_duration - stripped_duration
        pct     = saved_s / original_duration * 100 if original_duration else 0
        logger.info(
            f"[Preprocessor] Silence removed: {saved_s:.0f}s stripped "
            f"({pct:.0f}% of original). Remaining: {stripped_duration:.0f}s "
            f"({stripped_duration/60:.1f} min)"
        )

        chunks = _split_chunks(stripped_path)
        Path(stripped_path).unlink(missing_ok=True)

        logger.info(
            f"[Preprocessor] Split into {len(chunks)} chunk(s) "
            f"of ≤{CHUNK_SECONDS // 60} min each"
        )
        return chunks

    except Exception as exc:
        logger.warning(
            f"[Preprocessor] Failed ({exc}) — "
            "falling back to original file as single chunk"
        )
        return [_copy_to_temp(audio_path)]


def cleanup_chunks(chunk_paths: list[str]) -> None:
    """Delete all temp chunk files created by preprocess()."""
    for path in chunk_paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"[Preprocessor] Could not delete chunk {path}: {exc}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(result.stdout.strip())


def _remove_silence(audio_path: str) -> str:
    """
    Strip silence using ffmpeg's silenceremove filter.

    Uses the 'areverse' trick to handle both leading AND trailing silence:
      pass 1 (forward)  → removes leading + internal silence
      areverse           → flip audio
      pass 2 (forward on flipped) → removes what was trailing silence
      areverse           → flip back

    Returns path to a new temp .mp3 file.
    Uses mkstemp (not deprecated mktemp) to atomically create the temp file.
    """
    fd, out = tempfile.mkstemp(suffix=".mp3", prefix="zoom_stripped_")
    os.close(fd)  # ffmpeg will write to the path; close the OS fd we got from mkstemp

    silence_filter = (
        f"silenceremove="
        f"start_periods=1:"
        f"start_silence={PAD_S}:"
        f"start_threshold={SILENCE_DB}dB:"
        f"stop_periods=-1:"
        f"stop_silence={PAD_S}:"
        f"stop_threshold={SILENCE_DB}dB,"
        f"areverse,"
        f"silenceremove="
        f"start_periods=1:"
        f"start_silence={PAD_S}:"
        f"start_threshold={SILENCE_DB}dB,"
        f"areverse"
    )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-af", silence_filter,
            "-c:a", "libmp3lame", "-q:a", "4",
            "-loglevel", "error",
            out,
        ],
        check=True,
        timeout=_FFMPEG_TIMEOUT,
    )
    return out


def _split_chunks(audio_path: str) -> list[str]:
    """
    Split audio into segments of at most CHUNK_SECONDS each.
    Returns a list of new temp .mp3 file paths.
    """
    duration = _get_duration(audio_path)

    if duration <= CHUNK_SECONDS:
        return [_copy_to_temp(audio_path)]

    n_chunks = int(duration // CHUNK_SECONDS) + (1 if duration % CHUNK_SECONDS else 0)
    chunks   = []

    for i in range(n_chunks):
        start = i * CHUNK_SECONDS
        fd, out = tempfile.mkstemp(suffix=".mp3", prefix=f"zoom_chunk{i:02d}_")
        os.close(fd)  # ffmpeg will overwrite via -y; close the OS fd
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t",  str(CHUNK_SECONDS),
                "-i",  audio_path,
                "-c",  "copy",
                "-loglevel", "error",
                out,
            ],
            check=True,
            timeout=_FFMPEG_TIMEOUT,
        )
        chunks.append(out)

    return chunks


def _copy_to_temp(audio_path: str) -> str:
    """Copy a file to a new temp path (so the caller always owns the returned paths)."""
    fd, out = tempfile.mkstemp(suffix=Path(audio_path).suffix, prefix="zoom_chunk_")
    os.close(fd)  # ffmpeg will overwrite via -y; close the OS fd
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-c", "copy", "-loglevel", "error", out],
        check=True,
        timeout=_FFMPEG_TIMEOUT,
    )
    return out
