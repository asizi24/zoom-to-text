"""
Audio preprocessor for Whisper transcription.

Two steps applied before transcription (WHISPER_API mode):
  1. Silence removal — strips segments below -40 dBFS using ffmpeg's silenceremove
     filter, so Whisper doesn't waste time on dead air between topics.
  2. Chunking — splits the result into ≤13-minute pieces. This is the practical
     accuracy/memory sweet spot discovered through use.

Plus one standalone async helper used by the upload pipeline (all non-GEMINI
modes): extract_audio_track() — swaps a heavy uploaded video for a lean MP3.

All operations use ffmpeg (already available via yt-dlp). No extra dependencies.

The caller always receives NEW temp files it owns and must delete. The original
audio path is never modified (except extract_audio_track, which deletes its
source by design — see its docstring).
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

# Maximum time (seconds) for any single ffmpeg/ffprobe call.
# A 10-hour recording at fast read speeds should finish in < 5 minutes.
_FFMPEG_TIMEOUT = 600

# Extraction re-encodes the full audio track; on a shared Fly.io CPU a 3-hour
# lecture can take 15-20 min (same reason zoom_downloader skips it for
# GEMINI_DIRECT), so this timeout is deliberately generous.
_EXTRACT_TIMEOUT = 1800

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 13 * 60   # 13 minutes per chunk
SILENCE_DB    = -40        # dBFS threshold — below this is treated as silence
SILENCE_MIN_S = 1.0        # minimum silence duration to remove (seconds)
PAD_S         = 0.2        # seconds of silence to keep around speech (natural transitions)

# Speech normalization applied before Whisper sees the audio:
#   highpass=80   — cuts HVAC/handling rumble below the speech band
#   dynaudnorm    — adaptive per-window gain: lifts quiet passages (lecturer
#                   walking away from the mic, student questions from the back
#                   of the room) without crushing the loud ones the way a
#                   single global gain (loudnorm one-pass) would.
#                   f=250ms frames + g=15 window ≈ responsive but not pumping.
_NORMALIZE_FILTER = "highpass=f=80,dynaudnorm=f=250:g=15"


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


async def extract_audio_track(src_path: str, delete_source: bool = True) -> str:
    """
    Extract the audio track from a local media file into a 96 kbps MP3 —
    the same codec/bitrate the URL download path produces via yt-dlp, so
    everything downstream (Whisper, playback persistence) sees identical input.

    Runs ffmpeg via asyncio.create_subprocess_exec: the event loop stays free
    while a 1 GB lecture .mp4 is re-encoded down to ~45 MB of MP3.

    On success: returns the new .mp3 path; if delete_source (default), the
    original heavy file is removed immediately to free container disk.
    On failure: removes any partial output and returns the ORIGINAL path
    untouched — extraction is an optimization, not a hard requirement, so
    the pipeline still gets a chance to process the raw upload.
    """
    src = Path(src_path)
    if src.suffix.lower() == ".mp3":
        # Already an MP3 — re-encoding (even to normalize) would stack a
        # second generation of lossy compression; Whisper's VAD + temperature
        # fallbacks cope with unnormalized MP3s well enough.
        return src_path

    dest = src.with_suffix(".mp3")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",                                # drop the video stream entirely
        "-af", _NORMALIZE_FILTER,             # rumble cut + adaptive loudness
        "-c:a", "libmp3lame", "-b:a", "96k",  # matches yt-dlp preferredquality=96
        "-loglevel", "error",
        str(dest),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_EXTRACT_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"ffmpeg timed out after {_EXTRACT_TIMEOUT}s")
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited {proc.returncode}: "
                f"{stderr.decode(errors='replace')[:500]}"
            )
        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output file")
    except Exception as exc:
        logger.warning(
            f"[Preprocessor] Audio extraction failed ({exc}) — using original file"
        )
        dest.unlink(missing_ok=True)
        return src_path

    freed_mb = (src.stat().st_size - dest.stat().st_size) / 1024 / 1024
    if delete_source:
        src.unlink(missing_ok=True)
    logger.info(
        f"[Preprocessor] Extracted audio track: {dest.name} "
        f"(freed {freed_mb:.0f} MB of disk)"
    )
    return str(dest)


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

    # Normalization runs AFTER silence removal — boosting quiet passages first
    # would lift room noise above SILENCE_DB and defeat the silence detector.
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
        f"areverse,"
        f"{_NORMALIZE_FILTER}"
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
