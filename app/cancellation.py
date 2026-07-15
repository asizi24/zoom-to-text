"""
In-process cooperative-cancellation registry.

A cancelled task id lives in a module-level set. Two consumers poll it:

  - the pipeline orchestrator (processor.py) between major steps, and
  - the Whisper segment loop (transcriber.py), which runs in a worker THREAD.

The thread is exactly why this is a plain set + threading.Lock rather than an
asyncio primitive: the transcription loop cannot await, and reaching back into
the event loop on every segment just to check a flag would be absurd. Set
membership is atomic under the GIL, so the hot-path check (`is_cancelled`) takes
no lock; the lock only guards the rare mutations.

In-process is sufficient for the same reason the SSE hub is (app/events.py):
there is exactly one uvicorn worker (see Dockerfile). If the worker is ever
split into its own service, this becomes a shared flag in the DB/Redis — the
`request_cancel`/`is_cancelled`/`clear` contract stays identical.
"""
import threading

_lock = threading.Lock()
_cancelled: set[str] = set()


def request_cancel(task_id: str) -> None:
    """Flag a task for cancellation. Idempotent."""
    with _lock:
        _cancelled.add(task_id)


def is_cancelled(task_id: str) -> bool:
    """Hot path — called once per transcription segment. Lock-free by design."""
    return task_id in _cancelled


def clear(task_id: str) -> None:
    """Drop the flag once the task has fully unwound. Idempotent.

    Called from the worker's finally block for every terminal outcome, so a
    later retry of the same id starts with a clean slate.
    """
    with _lock:
        _cancelled.discard(task_id)
