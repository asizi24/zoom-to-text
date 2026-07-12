"""
In-process pub/sub hub for task progress events.

state.py publishes an event every time a task row changes (status/progress
update, transcript delta, completion, failure); the SSE endpoint in
routes.py subscribes per-task and streams the events to the browser.

This replaces 2-second HTTP polling with push updates. It is deliberately
in-process (a dict of asyncio.Queues, no Redis): the app runs a single
uvicorn worker by design (see Dockerfile CMD), so every subscriber lives
on the same event loop as every publisher.

All publishers run on the event loop (state.py functions are async; the
transcriber thread reaches them via run_coroutine_threadsafe), so publish()
can be a plain sync call using Queue.put_nowait.
"""
import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Per-subscriber buffer. A browser that stops reading for this many events
# gets its queue dropped rather than blocking the pipeline.
_QUEUE_SIZE = 256


class TaskEventHub:
    def __init__(self):
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._subs[task_id].add(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(task_id)
        if subs is None:
            return
        subs.discard(q)
        if not subs:
            self._subs.pop(task_id, None)

    def publish(self, task_id: str, event: dict) -> None:
        for q in tuple(self._subs.get(task_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow/stuck client: drop it. The frontend's watchdog poll
                # (every 20s) resyncs anything a dropped stream missed.
                self._subs[task_id].discard(q)
                logger.warning("Dropped slow SSE subscriber for task %s", task_id)


hub = TaskEventHub()
