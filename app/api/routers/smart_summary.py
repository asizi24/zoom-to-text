"""
On-demand Smart Summary endpoints.

POST /tasks/{id}/smart-summary            — start Map-Reduce generation (202)
GET  /tasks/{id}/smart-summary            — status + markdown (poll)
GET  /tasks/{id}/smart-summary/export.md  — download the Obsidian note
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app import state
from app.api.deps import get_current_user
from app.models import TaskStatus
from app.ratelimit import rate_limit
from app.services import smart_summary

logger = logging.getLogger(__name__)
router = APIRouter()

# Generation queues heavy LLM work — share the task submission budget.
_rate_limit = Depends(rate_limit("tasks", "rate_limit_tasks_per_minute"))


@router.post("/tasks/{task_id}/smart-summary", status_code=202,
             dependencies=[_rate_limit])
async def start_smart_summary(task_id: str, user_id: str = Depends(get_current_user)):
    """Kick off Map-Reduce Smart Summary generation for a completed task.

    Idempotent: if a run is already pending/running, returns its current status
    instead of launching a second one.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail="ניתן ליצור סיכום חכם רק למשימה שהושלמה",
        )
    if task.result is None:
        raise HTTPException(status_code=400, detail="למשימה אין תוצאה לסיכום")

    current = await state.get_smart_summary(task_id)
    if current and current["status"] in ("pending", "running"):
        return {"status": current["status"]}

    await state.set_smart_summary_status(task_id, "pending")
    smart_summary.enqueue(task_id)
    return {"status": "pending"}


@router.get("/tasks/{task_id}/smart-summary")
async def get_smart_summary(task_id: str, user_id: str = Depends(get_current_user)):
    """Poll generation status and fetch the markdown once complete.

    Response: {"status": idle|pending|running|completed|failed,
               "markdown": str|null, "error": str|null}
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    data = await state.get_smart_summary(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return data


@router.get("/tasks/{task_id}/smart-summary/export.md")
async def export_smart_summary(task_id: str, user_id: str = Depends(get_current_user)):
    """Download the generated Obsidian note as a .md file."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    data = await state.get_smart_summary(task_id)
    if not data or not data.get("markdown"):
        raise HTTPException(status_code=400, detail="אין סיכום חכם להורדה")
    filename = f"smart-summary-{task_id[:8]}.md"
    return Response(
        content=data["markdown"],
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
