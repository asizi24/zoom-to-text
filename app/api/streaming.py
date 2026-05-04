"""
WebSocket streaming transcription endpoint — code-only, home-server only.

Never activated in production (ENABLE_STREAMING=False by default).
Activate on a home server with enough RAM: ENABLE_STREAMING=true in .env

Protocol
--------
Client → Server:
  1. Optional JSON: {"type": "config", "language": "he"}
  2. Binary frames: raw audio bytes (accumulated; any number of chunks)
  3. JSON: {"type": "done"}

Server → Client:
  - JSON: {"type": "done", "transcript": "..."} — assembled transcript
  - JSON: {"type": "error", "message": "..."} — on transcription failure

Auth: session_id cookie (same as REST API).
Gate: settings.enable_streaming must be True.

Close codes (sent after accept):
  1008 — Policy Violation: streaming is disabled in this deployment
  3401 — Unauthorized: missing or expired session
"""
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, WebSocket, WebSocketDisconnect

from app import state
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/transcribe")
async def ws_transcribe(
    websocket: WebSocket,
    session_id: Optional[str] = Cookie(default=None),
):
    """
    Accept a stream of binary audio chunks and return a full transcript.

    The client may send as many binary frames as needed while recording,
    then signals end-of-stream with {"type": "done"}. All chunks are
    buffered to a temp file, transcribed via Faster-Whisper, and the
    result is returned in a single {"type": "done", "transcript": "..."}
    message before the connection closes.
    """
    # Accept first so close codes are sent inside a proper WS frame.
    await websocket.accept()

    if not settings.enable_streaming:
        await websocket.close(code=1008)
        return

    user_id = await state.get_session_user(session_id) if session_id else None
    if not user_id:
        await websocket.close(code=3401)
        return

    language = "he"
    chunks: list[bytes] = []

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            raw_bytes = message.get("bytes")
            if raw_bytes:
                chunks.append(raw_bytes)
                continue

            raw_text = message.get("text")
            if raw_text:
                data = json.loads(raw_text)
                msg_type = data.get("type")

                if msg_type == "config":
                    language = data.get("language", language)

                elif msg_type == "done":
                    await _finalize(websocket, chunks, language)
                    break

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected during streaming")
    except Exception as exc:
        logger.exception(f"WebSocket streaming error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


async def _finalize(websocket: WebSocket, chunks: list[bytes], language: str) -> None:
    """Write buffered audio to a temp file, transcribe, and send the result."""
    if not chunks:
        await websocket.send_json({"type": "done", "transcript": ""})
        return

    from app.services import transcriber

    audio_bytes = b"".join(chunks)
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        transcript, _lang = await transcriber.transcribe(tmp_path, language)
        await websocket.send_json({"type": "done", "transcript": transcript})
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        Path(tmp_path).unlink(missing_ok=True)
