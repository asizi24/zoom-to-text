"""
In-process pipeline worker pool.

SQLite is the durable queue (tasks.payload_json holds the job parameters);
an asyncio.Queue is only the wake-up signal. This gives:
  - A hard concurrency cap (settings.pipeline_concurrency, default 1) so N
    simultaneous submissions can no longer run N parallel Whisper
    transcriptions and OOM the container.
  - Restart-resume: on startup, tasks that were mid-flight when the server
    died are re-enqueued instead of failed (state.reset_interrupted_tasks).

Scale-out path: when one container is no longer enough, swap this module's
queue for arq/Redis and run the worker loop as a second compose service —
the enqueue()/payload contract stays identical.
"""
import asyncio
import logging

from app import state
from app.config import settings
from app.models import ProcessingMode
from app.services import processor

logger = logging.getLogger(__name__)

# Created in start() rather than at import time: asyncio.Queue binds to the
# running event loop on first use, and the app (or a test client) may start
# more than one loop over the process lifetime.
_queue: asyncio.Queue[str] | None = None
_workers: list[asyncio.Task] = []


async def start() -> None:
    """Start the worker coroutines and re-enqueue jobs interrupted by a restart.
    Called from the FastAPI lifespan after state.init_db()."""
    global _queue
    _queue = asyncio.Queue()
    for task_id in await state.reset_interrupted_tasks():
        _queue.put_nowait(task_id)
    for i in range(settings.pipeline_concurrency):
        _workers.append(asyncio.create_task(_worker_loop(i), name=f"pipeline-worker-{i}"))
    logger.info(f"{settings.pipeline_concurrency} pipeline worker(s) started")


async def stop() -> None:
    for w in _workers:
        w.cancel()
    await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()


def enqueue(task_id: str) -> None:
    """Signal the workers that a task (already persisted with its payload) is ready."""
    if _queue is None:
        raise RuntimeError("worker.start() has not been called")
    _queue.put_nowait(task_id)


async def _worker_loop(idx: int) -> None:
    queue = _queue
    assert queue is not None  # start() assigns before spawning loops
    while True:
        task_id = await queue.get()
        try:
            job = await state.get_job_payload(task_id)
            if job is None:
                logger.warning(f"worker[{idx}]: task {task_id} has no payload — skipping")
                continue
            if job.get("file_path"):
                await processor.run_pipeline_from_file(
                    task_id=task_id,
                    file_path=job["file_path"],
                    mode=ProcessingMode(job["mode"]),
                    language=job.get("language", "he"),
                )
            else:
                await processor.run_pipeline(
                    task_id=task_id,
                    url=job["url"],
                    mode=ProcessingMode(job["mode"]),
                    cookies=job.get("cookies"),
                    language=job.get("language", "he"),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            # run_pipeline handles its own failures; this guards the loop itself.
            logger.exception(f"worker[{idx}] crashed on task {task_id}")
        finally:
            # Wipe the payload whether the task succeeded or failed — it may
            # contain the user's Zoom session cookies.
            try:
                await state.clear_job_payload(task_id)
            except Exception:
                logger.exception(f"worker[{idx}]: failed to clear payload for {task_id}")
            queue.task_done()
