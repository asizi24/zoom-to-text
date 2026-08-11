"""
Smart Summary repository — the three task columns that back the on-demand
Map-Reduce Obsidian summary (smart_summary / smart_summary_status /
smart_summary_error).

Kept separate from the transcription result_json: re-generating a Smart Summary
must never mutate the lesson (summary/chapters/quiz/flashcards). Status values:
  idle | pending | running | completed | failed
"""
import logging
from typing import Optional

from app import state
from app.state import get_write_lock

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = ("pending", "running")


async def get_smart_summary(task_id: str) -> Optional[dict]:
    """Return {status, markdown, error} for a task, or None if the task is gone.

    A task that has never generated a Smart Summary reports status 'idle'.
    """
    db = await state._get_db()
    async with db.execute(
        "SELECT smart_summary, smart_summary_status, smart_summary_error "
        "FROM tasks WHERE id=?",
        [task_id],
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "status": row["smart_summary_status"] or "idle",
        "markdown": row["smart_summary"],
        "error": row["smart_summary_error"],
    }


async def set_smart_summary_status(
    task_id: str, status: str, error: Optional[str] = None
) -> None:
    """Update the generation status. Clears any prior error unless one is given."""
    async with get_write_lock():
        db = await state._get_db()
        await db.execute(
            "UPDATE tasks SET smart_summary_status=?, smart_summary_error=? WHERE id=?",
            [status, error, task_id],
        )
        await db.commit()


async def save_smart_summary(task_id: str, markdown: str) -> None:
    """Persist the finished markdown and flip status to completed."""
    async with get_write_lock():
        db = await state._get_db()
        await db.execute(
            "UPDATE tasks SET smart_summary=?, smart_summary_status='completed', "
            "smart_summary_error=NULL WHERE id=?",
            [markdown, task_id],
        )
        await db.commit()


async def reset_running_smart_summaries() -> int:
    """Startup cleanup: a Smart Summary job runs as an in-process background task
    (not the durable pipeline queue), so a restart orphans any in-flight run.
    Flip pending/running back to failed so the UI offers a re-run instead of
    spinning forever. Returns the number of rows reset."""
    async with get_write_lock():
        db = await state._get_db()
        placeholders = ",".join("?" * len(_ACTIVE_STATUSES))
        cursor = await db.execute(
            f"UPDATE tasks SET smart_summary_status='failed', "
            f"smart_summary_error=? WHERE smart_summary_status IN ({placeholders})",
            ["השרת הופעל מחדש בזמן היצירה — נסה שוב", *_ACTIVE_STATUSES],
        )
        await db.commit()
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
