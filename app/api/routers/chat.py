"""
Lesson Q&A: single-shot ask, multi-turn streaming chat, and chat history.
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import state
from app.api.deps import get_current_user
from app.services import summarizer

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., description="Question about the lesson content")


def _build_lesson_context(result) -> str:
    """Build a rich context string from a completed lesson result."""
    parts = []
    if result.summary:
        parts.append(f"סיכום:\n{result.summary}")
    for ch in result.chapters:
        parts.append(f"\nפרק: {ch.title}\n{ch.content}")
        if ch.key_points:
            parts.append("נקודות מרכזיות: " + ", ".join(ch.key_points))
    if result.transcript:
        # Include up to 30 k chars of transcript for richer answers
        parts.append(f"\nתמלול:\n{result.transcript[:30_000]}")
    return "\n".join(parts)


@router.post("/tasks/{task_id}/ask")
async def ask_question(task_id: str, body: AskRequest, user_id: str = Depends(get_current_user)):
    """
    Ask a question about a completed lesson.
    Uses the stored summary + chapters as context for a Gemini-powered answer.
    Returns 404 if not found or owned by a different user.
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(status_code=400, detail="Task has no result yet — wait for processing to complete")

    context = _build_lesson_context(task.result)

    try:
        answer = await summarizer.ask_about_lesson(context, body.question)
        return {"answer": answer}
    except Exception as exc:
        logger.error(f"Ask failed for task {task_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tasks/{task_id}/chat")
async def chat_with_recording(
    task_id: str,
    body: AskRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Multi-turn streaming chat about a completed lesson.

    Returns a Server-Sent Events stream where each event carries a JSON payload:
      {"text": "<chunk>"}   — partial model response
      {"done": true}        — stream finished (no more events)
      {"error": "<msg>"}    — error occurred

    The user message and final model response are stored in SQLite so the
    history survives page reloads. History is capped at 40 messages
    (app/repositories/chat.py).
    """
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.result is None:
        raise HTTPException(
            status_code=400, detail="Task has no result yet — wait for processing to complete"
        )

    context = _build_lesson_context(task.result)
    history = await state.get_chat_history(task_id)

    # Persist the user message before streaming starts
    await state.append_chat_message(task_id, "user", body.question)

    async def generate():
        full_response: list[str] = []
        try:
            async for chunk in summarizer.stream_chat_response(context, history, body.question):
                full_response.append(chunk)
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"Chat stream failed for task {task_id}: {exc}")
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            if full_response:
                await state.append_chat_message(task_id, "model", "".join(full_response))
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Cache-Control": "no-cache",
        },
    )


@router.get("/tasks/{task_id}/chat")
async def get_chat_history(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Return the stored chat history for a completed task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    history = await state.get_chat_history(task_id)
    return {"history": history}


@router.delete("/tasks/{task_id}/chat", status_code=204)
async def clear_chat_history(
    task_id: str,
    user_id: str = Depends(get_current_user),
):
    """Clear the chat history for a task."""
    task = await state.get_task_for_user(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    await state.clear_chat_history(task_id)
