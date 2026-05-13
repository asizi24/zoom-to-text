"""Audio-clip extractor (B4).

Uses ffmpeg to slice a recording's source audio between two timestamps and
returns the bytes of an mp3-encoded clip. The slice is computed on demand
(no caching on disk yet) — keeps the share endpoint stateless and avoids
clip-file cleanup logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard cap so a runaway request can't pin the CPU on the single-machine deployment.
_MAX_DURATION_SEC = 5 * 60  # 5 minutes
_FFMPEG_TIMEOUT_SEC = 60.0


class ClipExtractionError(Exception):
    """Raised when ffmpeg fails or the requested range is invalid."""


async def extract_clip_bytes(audio_path: str, start_sec: float, end_sec: float) -> bytes:
    """Slice ``audio_path`` from ``start_sec`` to ``end_sec`` and return mp3 bytes."""
    if not audio_path or not Path(audio_path).exists():
        raise ClipExtractionError("Source audio is missing on disk")
    if end_sec <= start_sec:
        raise ClipExtractionError("end_sec must be greater than start_sec")
    duration = end_sec - start_sec
    if duration > _MAX_DURATION_SEC:
        raise ClipExtractionError(
            f"Clip duration {duration:.0f}s exceeds max {_MAX_DURATION_SEC}s"
        )

    fd, out_path = tempfile.mkstemp(suffix=".mp3", prefix="clip-")
    os.close(fd)
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", audio_path,
            "-t", f"{duration:.3f}",
            "-vn",
            "-acodec", "libmp3lame",
            "-b:a", "96k",
            "-loglevel", "error",
            out_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=_FFMPEG_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ClipExtractionError("ffmpeg timed out while extracting clip")
        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="ignore").strip()
            raise ClipExtractionError(f"ffmpeg failed: {err[:200]}")
        return Path(out_path).read_bytes()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
