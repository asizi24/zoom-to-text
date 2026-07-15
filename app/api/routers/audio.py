"""
Audio streaming (Feature 7): serve a task's persisted audio with HTTP Range
support — required for HTML5 <audio> seeking.
"""
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import state
from app.api.deps import get_current_user
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Persistent audio root — files live here after processing so the UI player can
# seek back into them. Matches processor._audio_root() exactly.
_AUDIO_ROOT = (settings.data_dir / "audio").resolve()


def _path_under_audio_root(p: Path) -> bool:
    """
    Return True iff p resolves to a location strictly inside _AUDIO_ROOT.

    Prevents directory traversal — even if somebody managed to inject a
    symlink or a ../../-laden path into the DB, we refuse to serve it.
    """
    try:
        return p.resolve().is_relative_to(_AUDIO_ROOT)
    except Exception:
        return False


# Content-type map: keep small, known-safe list (no arbitrary mimetypes.guess_type)
_AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def _parse_range(header: str | None, file_size: int) -> tuple[int, int] | None:
    """
    Parse a single-range 'Range: bytes=a-b' request.
    Returns (start, end) inclusive, or None if the header is missing/invalid.
    Multi-range requests are not supported — we serve the first range only.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[6:].split(",", 1)[0].strip()
    try:
        start_s, end_s = spec.split("-", 1)
        if start_s == "":
            # suffix: "bytes=-500" → last 500 bytes
            length = int(end_s)
            if length <= 0:
                return None
            start = max(file_size - length, 0)
            end   = file_size - 1
        else:
            start = int(start_s)
            end   = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start < 0 or end >= file_size or start > end:
        return None
    return start, end


@router.get("/tasks/{task_id}/audio")
async def stream_audio(
    task_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """
    Stream the persistent audio file for a task with HTTP Range support.

    Range support is required for HTML5 <audio> seek to work — without it,
    the browser can only play from byte 0 and seekbar drags are no-ops.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    audio_path = await state.get_audio_path(task_id)
    if not audio_path:
        raise HTTPException(status_code=404, detail="No audio stored for this task")

    p = Path(audio_path)
    if not p.exists() or not _path_under_audio_root(p):
        # Either the file was wiped or the path is suspicious — both 404
        raise HTTPException(status_code=404, detail="Audio file unavailable")

    file_size = p.stat().st_size
    media_type = _AUDIO_TYPES.get(p.suffix.lower(), "application/octet-stream")

    range_header = request.headers.get("range")
    rng = _parse_range(range_header, file_size)

    async def _iter(start: int, length: int, chunk: int = 64 * 1024):
        # aiofiles for async I/O; small per-chunk read keeps memory bounded
        async with aiofiles.open(p, "rb") as f:
            await f.seek(start)
            remaining = length
            while remaining > 0:
                data = await f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if rng is None:
        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": "inline",
        }
        return StreamingResponse(_iter(0, file_size), media_type=media_type, headers=headers)

    start, end = rng
    length = end - start + 1
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": "inline",
    }
    return StreamingResponse(
        _iter(start, length), status_code=206, media_type=media_type, headers=headers
    )
