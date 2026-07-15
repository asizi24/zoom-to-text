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

from app import cancellation, state
from app.config import settings
from app.logging_config import task_id_var
from app.models import ProcessingMode
from app.services import processor

logger = logging.getLogger(__name__)

# Created in start() rather than at import time: asyncio.Queue binds to the
# running event loop on first use, and the app (or a test client) may start
# more than one loop over the process lifetime.
_queue: asyncio.Queue[str] | None = None
_workers: list[asyncio.Task] = []
# Task ids currently being processed — consulted by stop() so shutdown can
# flag them for cooperative cancellation (see stop() for why that matters).
_in_flight: set[str] = set()


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
    # Flag in-flight tasks BEFORE cancelling the worker coroutines: the Whisper
    # transcription runs in a non-daemon executor thread that asyncio.Task
    # cancellation cannot reach — without the flag, interpreter shutdown would
    # block on that thread until the entire lecture finished transcribing (or
    # Docker's stop grace period SIGKILLed us). With it, the thread exits at
    # its next per-segment checkpoint (~seconds). The task row stays in-flight
    # in the DB with its payload intact, so the next startup re-queues it.
    for task_id in tuple(_in_flight):
        cancellation.request_cancel(task_id)
    for w in _workers:
        w.cancel()
    await asyncio.gather(*_workers, return_exceptions=True)
    _workers.clear()


def enqueue(task_id: str) -> None:
    """Signal the workers that a task (already persisted with its payload) is ready."""
    if _queue is None:
        raise RuntimeError("worker.start() has not been called")
    _queue.put_nowait(task_id)


def is_running() -> bool:
    """Readiness check: the pool has started and at least one worker is alive."""
    return _queue is not None and any(not w.done() for w in _workers)


async def _worker_loop(idx: int) -> None:
    queue = _queue
    assert queue is not None  # start() assigns before spawning loops
    while True:
        task_id = await queue.get()
        _in_flight.add(task_id)
        # Bind the id for the whole pipeline run: every log line it emits —
        # download, transcription progress, Gemini calls — carries task_id in
        # the structured output without touching the individual call sites.
        ctx_token = task_id_var.set(task_id)
        shutting_down = False
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
            shutting_down = True
            raise
        except Exception:
            # run_pipeline handles its own failures; this guards the loop itself.
            logger.exception(f"worker[{idx}] crashed on task {task_id}")
        finally:
            task_id_var.reset(ctx_token)
            _in_flight.discard(task_id)
            if not shutting_down:
                # Drop any cancellation flag so a later retry of this id starts
                # clean. Skipped during shutdown — stop() just SET that flag so
                # the transcription thread (which outlives this coroutine) can
                # see it and exit; clearing it here would strand the thread.
                cancellation.clear(task_id)
                # Scrub the cookies from the payload but KEEP the rest so the
                # task stays retryable — cookies must not linger, everything
                # else is harmless job metadata that /retry replays.
                try:
                    await state.finalize_job_payload(task_id)
                except Exception:
                    logger.exception(f"worker[{idx}]: failed to finalize payload for {task_id}")
            queue.task_done()
