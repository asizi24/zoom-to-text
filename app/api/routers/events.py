"""
Live task updates: the SSE event stream and the transcript-delta endpoint the
frontend uses while a task is transcribing (and as its polling fallback).
"""
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app import state
from app.api.deps import get_current_user
from app.events import hub as events_hub
from app.models import TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Live transcript preview ───────────────────────────────────────────────────────

@router.get("/tasks/{task_id}/transcript")
async def get_partial_transcript(
    task_id: str,
    offset: int = Query(0, ge=0, description="Character offset — return only text after this position"),
    user_id: str = Depends(get_current_user),
):
    """
    Return the live partial transcript delta for WHISPER-mode tasks.

    Poll this endpoint while status == 'transcribing' to stream transcript
    text as it is produced. Use ?offset=N to get only the new characters
    since the last poll — the frontend tracks the offset locally and sends
    it on each request so only the delta is transferred.

    Response: {"text": "<new chars>", "total": <total length so far>}
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    delta, total = await state.get_partial_transcript(task_id, from_offset=offset)
    return {"text": delta, "total": total}


# ── Live task events (SSE) ────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/tasks/{task_id}/events")
async def task_events(
    task_id: str,
    request: Request,
    user_id: str = Depends(get_current_user),
):
    """
    Server-Sent Events stream of live task updates — replaces 2s polling.

    Events (each a JSON object in the `data:` field):
      {"type":"snapshot", status, progress, message, transcript_total}  — on connect
      {"type":"status", status, progress, message}                     — task row changed
      {"type":"transcript", text, total}                               — live transcript delta
      {"type":"done", status}                                          — terminal; stream closes

    A `: ping` comment is sent every 15s so proxies don't kill the idle
    connection. The frontend falls back to polling when this endpoint is
    unavailable, and keeps a 20s watchdog poll as a safety net.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # Subscribe BEFORE reading the snapshot so no event can slip between them.
    queue = events_hub.subscribe(task_id)

    async def generate():
        try:
            # Snapshot: current row + transcript length (delta fetched by
            # the client through GET /transcript with its own offset).
            _, transcript_total = await state.get_partial_transcript(
                task_id, from_offset=2**31
            )
            snapshot_task = await state.get_task(task_id)
            yield _sse({
                "type": "snapshot",
                "status": snapshot_task.status.value,
                "progress": snapshot_task.progress,
                "message": snapshot_task.message,
                "transcript_total": transcript_total,
            })
            if snapshot_task.status in (
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ):
                yield _sse({"type": "done", "status": snapshot_task.status.value})
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        return
                    yield ": ping\n\n"
                    continue
                yield _sse(event)
                if event.get("type") == "done":
                    return
        finally:
            events_hub.unsubscribe(task_id, queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
