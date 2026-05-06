/**
 * SSE progress client — ES module.
 *
 * Wraps the GET /api/tasks/{id}/events stream and translates events into
 * callbacks so callers stay decoupled from the EventSource API.
 *
 * Usage (from the main script):
 *   import { TaskStream } from '/static/js/sse.js';
 *   const stream = new TaskStream(taskId, {
 *     onProgress(pct, msg) { … },
 *     onComplete(result, hasAudio) { … },
 *     onFailed(msg, details)  { … },
 *     onCancelled()           { … },
 *   });
 *   stream.start();   // opens the EventSource
 *   stream.stop();    // closes it (e.g. on user cancel or component teardown)
 */

export class TaskStream {
  /** @param {string} taskId  @param {object} callbacks */
  constructor(taskId, callbacks = {}) {
    this._taskId   = taskId;
    this._cb       = callbacks;
    this._es       = null;
  }

  start() {
    if (this._es) return;   // already running
    const url = `/api/tasks/${this._taskId}/events`;
    const es  = new EventSource(url, { withCredentials: true });
    this._es  = es;

    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch { return; }

      this._cb.onProgress?.(d.progress ?? 0, d.message ?? '');

      if (!d.done) return;
      es.close(); this._es = null;

      switch (d.status) {
        case 'completed':
          this._cb.onComplete?.(d.result, !!d.has_audio);
          break;
        case 'failed':
          this._cb.onFailed?.(d.message ?? 'עיבוד נכשל', null);
          break;
        case 'cancelled':
          this._cb.onCancelled?.();
          break;
      }
    };

    es.onerror = async () => {
      es.close(); this._es = null;
      // Fall back to a single REST poll after a stream error
      try {
        const r    = await fetch(`/api/tasks/${this._taskId}`);
        if (!r.ok) return;
        const task = await r.json();
        this._cb.onProgress?.(task.progress ?? 0, task.message ?? '');
        if (task.status === 'completed') this._cb.onComplete?.(task.result, !!task.has_audio);
        else if (task.status === 'failed')    this._cb.onFailed?.(task.error ?? '', task.error_details);
        else if (task.status === 'cancelled') this._cb.onCancelled?.();
      } catch { /* swallow — caller will timeout or retry */ }
    };
  }

  stop() {
    if (this._es) { this._es.close(); this._es = null; }
  }
}
