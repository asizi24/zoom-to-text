"""
Flashcards: JSON listing + Anki (.apkg) and CSV export.
"""
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app import state
from app.api.deps import get_current_user
from app.services import anki_export

logger = logging.getLogger(__name__)
router = APIRouter()


def _sanitize_deck_name(task_id: str, url: str | None) -> str:
    """Build a readable deck name from the task. Stripped of filesystem-nasty chars."""
    base = (url or "").strip() or task_id
    # Collapse to a short label — Anki truncates long deck names awkwardly
    base = re.sub(r"https?://", "", base)[:60]
    base = re.sub(r"[^\w\-֐-׿ .]", " ", base).strip()
    return f"ZoomToText — {base or task_id[:8]}"


@router.get("/tasks/{task_id}/flashcards")
async def get_flashcards(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Return the generated flashcards for a completed task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet")
    return {"flashcards": [c.model_dump() for c in task.result.flashcards]}


@router.get("/tasks/{task_id}/flashcards/export.apkg")
async def export_flashcards_apkg(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Download the task's flashcards as an Anki .apkg package."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None or not task.result.flashcards:
        raise HTTPException(status_code=400, detail="No flashcards to export")

    deck_name = _sanitize_deck_name(task_id, task.url)
    data = anki_export.create_apkg(task.result.flashcards, deck_name, task_id)
    filename = f"flashcards-{task_id[:8]}.apkg"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/tasks/{task_id}/flashcards/export.csv")
async def export_flashcards_csv(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Download the task's flashcards as UTF-8 CSV (for users who don't use Anki)."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None or not task.result.flashcards:
        raise HTTPException(status_code=400, detail="No flashcards to export")

    data = anki_export.create_csv(task.result.flashcards)
    filename = f"flashcards-{task_id[:8]}.csv"
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
