// Live-updates orchestration for a running task.
//
// Transport strategy:
//   1. SSE via ReconnectingEventSource — exponential backoff (1s → 30s cap,
//      jitter); every reconnect gets a fresh server snapshot, and transcript
//      gaps are resynced through the delta endpoint.
//   2. After 5 consecutive SSE failures: degrade to 2s polling, then attempt
//      to upgrade back to SSE once a minute — the first SSE message stops the
//      polling again.
//   3. A 20s watchdog poll runs regardless, catching a stalled stream or a
//      missed 'done'.
import { store } from './store.js';
import { apiFetch, ApiError } from './api.js';
import { ReconnectingEventSource } from './sse.js';
import { setCancelEnabled, showProgress, updateProgress, hideProgress } from './progress.js';
import {
  applyTranscriptEvent, backfillTranscript, hideLiveTranscript,
  resetLiveTranscript, showLiveTranscriptCard,
  startTranscriptDeltaPolling, stopTranscriptPolling,
} from './transcript.js';
import { showResults } from './results.js';
import { showError, toast } from './ui.js';

let pollTimer     = null;
let watchdogTimer = null;
let stream        = null;   // ReconnectingEventSource | null
let sseUpgradeTimer = null;

const SSE_UPGRADE_RETRY_MS = 60000;

export function startLiveUpdates(taskId) {
  stopLiveUpdates();
  store.taskFinished = false;
  store.currentTaskId = taskId;
  setCancelEnabled(true);

  // Safety net: a slow poll catches a stalled stream or a missed 'done'
  watchdogTimer = setInterval(() => pollOnce(taskId), 20000);

  if (!window.EventSource) { startPollingBundle(taskId); return; }
  connectStream(taskId);
}

function connectStream(taskId) {
  stream = new ReconnectingEventSource(`/api/tasks/${taskId}/events`, {
    onMessage: (e) => {
      // SSE is healthy — if we had degraded to polling, upgrade back
      stopPollingBundle();
      let payload;
      try { payload = JSON.parse(e.data); } catch { return; }
      handleLiveEvent(taskId, payload);
    },
    onPermanentFailure: () => {
      stream = null;
      startPollingBundle(taskId);   // degrade — the UI keeps working
      sseUpgradeTimer = setTimeout(() => {
        if (!store.taskFinished && store.currentTaskId === taskId) connectStream(taskId);
      }, SSE_UPGRADE_RETRY_MS);
    },
  });
  stream.start();
}

export function stopLiveUpdates() {
  stopPollingBundle();
  if (watchdogTimer)   { clearInterval(watchdogTimer); watchdogTimer = null; }
  if (sseUpgradeTimer) { clearTimeout(sseUpgradeTimer); sseUpgradeTimer = null; }
  if (stream)          { stream.stop(); stream = null; }
}

function startPollingBundle(taskId) {
  if (!pollTimer) pollTimer = setInterval(() => pollOnce(taskId), 2000);
  startTranscriptDeltaPolling(taskId);
}

function stopPollingBundle() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  stopTranscriptPolling();
}

async function handleLiveEvent(taskId, p) {
  if (store.taskFinished) return;
  if (p.type === 'snapshot') {
    updateProgress(p.progress, p.message, p.status);
    if (p.transcript_total > 0) {
      showLiveTranscriptCard();
      await backfillTranscript(taskId);
    }
    if (p.status === 'completed' || p.status === 'failed') pollOnce(taskId);
  } else if (p.type === 'status') {
    updateProgress(p.progress, p.message, p.status);
  } else if (p.type === 'transcript') {
    await applyTranscriptEvent(taskId, p);
  } else if (p.type === 'done') {
    pollOnce(taskId);  // fetch the full task (result JSON) exactly once
  }
}

export async function pollOnce(taskId) {
  try {
    const r = await apiFetch(`/api/tasks/${taskId}`);
    applyTaskState(await r.json(), taskId);
  } catch { /* retry next tick (401 already redirected in apiFetch) */ }
}

function applyTaskState(task, taskId) {
  if (store.taskFinished) return;
  if (task.status === 'completed') {
    store.taskFinished = true;
    stopLiveUpdates();
    showResults(task.result, taskId, !!task.has_audio);
  } else if (task.status === 'failed') {
    store.taskFinished = true;
    stopLiveUpdates();
    hideLiveTranscript();
    hideProgress();
    showError(task.error || 'העיבוד נכשל');
  } else if (task.status === 'cancelled') {
    store.taskFinished = true;
    stopLiveUpdates();
    hideLiveTranscript();
    hideProgress();
    showError('העיבוד בוטל. ניתן להריץ אותו מחדש מלשונית ההיסטוריה.');
  } else {
    updateProgress(task.progress, task.message, task.status);
  }
}

// ── Cancel the running task ──────────────────────────────────────────────────

export async function cancelCurrentTask() {
  if (!store.currentTaskId) return;
  if (!confirm('לבטל את העיבוד הנוכחי? התהליך ייעצר ותוכל להריץ אותו מחדש מההיסטוריה.')) return;
  setCancelEnabled(false);
  try {
    await apiFetch(`/api/tasks/${store.currentTaskId}/cancel`, { method: 'POST' });
    // The SSE 'done'(cancelled) event (or the watchdog poll) tears down the
    // view; force one poll so the UI updates immediately on the fallback path.
    pollOnce(store.currentTaskId);
  } catch (err) {
    // 409 = already finished between click and request — treat as done.
    if (err instanceof ApiError && err.status === 409) {
      pollOnce(store.currentTaskId);
      return;
    }
    toast(err.message || 'לא ניתן לבטל את המשימה');
    setCancelEnabled(true);
  }
}

// ── Task resume — a reload must never orphan a running job ───────────────────

// Re-attach the progress UI to a task that is already running server-side.
// Accepts either a bare id (extension ?task= flow) or a task object from
// the history/list API ({id, url, progress, message, created_at}).
export function resumeTask(t) {
  const taskId = typeof t === 'string' ? t : t.id;
  if (typeof t === 'object') {
    store.currentSource = (t.url || '').replace(/^upload:/, '');
    store.processingStartTime = t.created_at ? Date.parse(t.created_at) : Date.now();
  } else {
    store.processingStartTime = Date.now();
  }
  showProgress();
  resetLiveTranscript();
  if (typeof t === 'object' && t.progress) {
    updateProgress(t.progress, t.message || 'ממשיך עיבוד...', t.status);
  }
  // Live transcript card appears lazily on the first delta — the resumed
  // task's mode is unknown client-side, so don't show an empty panel.
  startLiveUpdates(taskId);
}

// On page load: pick up the newest task that is still processing, so an
// upload keeps showing progress even after a refresh / browser restart.
export async function resumeLatestInFlight() {
  try {
    const r = await apiFetch('/api/tasks?limit=10');
    const tasks = await r.json();
    const terminal = ['completed', 'failed', 'cancelled'];
    const inflight = tasks.find(t => !terminal.includes(t.status));
    if (inflight) resumeTask(inflight);
  } catch { /* server unreachable — leave the normal input UI */ }
}
