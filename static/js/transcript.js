// Transcript concerns: the live-transcript card (buffer, pin, delta polling,
// SSE event application, backfill resync) and the finished-transcript search.
import { store } from './store.js';
import { hideChatPanel } from './chat.js';
import { linkifyTimestamps } from './player.js';

let transcriptTimer    = null;
let transcriptOffset   = 0;
let transcriptPinned   = false;
let liveTranscriptText = '';   // full text buffer — enables client-side search
let liveSearchQuery    = '';

function makeCursor() {
  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';
  return cursor;
}

export function resetLiveTranscript() {
  transcriptOffset   = 0;
  transcriptPinned   = false;
  liveTranscriptText = '';
  liveSearchQuery    = '';

  const input = document.getElementById('live-search-input');
  if (input) input.value = '';
  document.getElementById('live-search-count').textContent = '';

  const box = document.getElementById('live-transcript-box');
  box.textContent = '';
  box.appendChild(makeCursor());

  const pinBtn = document.getElementById('transcript-pin-btn');
  pinBtn.textContent = '📌 נעל';
  pinBtn.classList.remove('pinned');
  document.getElementById('live-transcript-status').textContent = 'ממתין לטקסט...';
}

export function showLiveTranscriptCard() {
  document.getElementById('live-transcript-card').style.display = 'block';
}

export function appendLiveTranscript(delta) {
  if (!delta) return;
  liveTranscriptText += delta;
  showLiveTranscriptCard();
  document.getElementById('live-transcript-status').textContent =
    liveTranscriptText.length.toLocaleString('he-IL') + ' תווים';

  const box = document.getElementById('live-transcript-box');
  if (liveSearchQuery) {
    // Search active — keep the filtered view in sync with fresh text
    renderTranscriptSearch(box, liveTranscriptText, liveSearchQuery,
      document.getElementById('live-search-count'));
    return;
  }
  const cur = box.querySelector('.typing-cursor');
  if (cur) cur.remove();
  box.appendChild(document.createTextNode(delta));
  box.appendChild(makeCursor());
  if (!transcriptPinned) box.scrollTop = box.scrollHeight;
}

// Apply an SSE 'transcript' event: append in-order deltas directly; on a gap
// (missed events across a reconnect) resync through the delta endpoint.
export async function applyTranscriptEvent(taskId, event) {
  showLiveTranscriptCard();
  if (event.total - event.text.length === transcriptOffset) {
    transcriptOffset = event.total;
    appendLiveTranscript(event.text);
  } else if (event.total > transcriptOffset) {
    await backfillTranscript(taskId);
  }
}

export function transcriptTotalSoFar() {
  return transcriptOffset;
}

export async function backfillTranscript(taskId) {
  try {
    const r = await fetch(`/api/tasks/${taskId}/transcript?offset=${transcriptOffset}`);
    if (!r.ok) return;
    const data = await r.json();
    transcriptOffset = data.total;
    if (data.text) appendLiveTranscript(data.text);
  } catch { /* next event or watchdog will retry */ }
}

// Fallback transport: poll the delta endpoint every 2s (SSE path pushes
// deltas instead). resetLiveTranscript() must run before this.
export function startTranscriptDeltaPolling(taskId) {
  if (transcriptTimer) clearInterval(transcriptTimer);
  transcriptTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/tasks/${taskId}/transcript?offset=${transcriptOffset}`);
      if (!r.ok) return;
      const data = await r.json();
      if (data.text) {
        transcriptOffset = data.total;
        appendLiveTranscript(data.text);
      }
    } catch { /* retry on next tick */ }
  }, 2000);
}

export function stopTranscriptPolling() {
  if (transcriptTimer) { clearInterval(transcriptTimer); transcriptTimer = null; }
}

export function hideLiveTranscript() {
  stopTranscriptPolling();
  document.getElementById('live-transcript-card').style.display = 'none';
  hideChatPanel();
}

export function toggleTranscriptPin() {
  transcriptPinned = !transcriptPinned;
  const btn = document.getElementById('transcript-pin-btn');
  btn.classList.toggle('pinned', transcriptPinned);
  btn.textContent = transcriptPinned ? '📌 נעול' : '📌 נעל';
}

// ── Transcript search (shared by live + finished views) ──
// Splits on [MM:SS] boundaries so every hit is shown with its timestamp
// context line, then highlights each occurrence. DOM-built — no innerHTML.
export function renderTranscriptSearch(box, text, query, countEl) {
  box.textContent = '';
  const q = query.toLowerCase();
  const segs = text.split(/(?=\[\d{1,2}:\d{2}\])/);
  let totalHits = 0;

  for (const seg of segs) {
    const lower = seg.toLowerCase();
    let idx = lower.indexOf(q);
    if (idx === -1) continue;

    const el = document.createElement('span');
    el.className = 't-seg';
    let last = 0;
    while (idx !== -1) {
      totalHits++;
      if (idx > last) el.appendChild(document.createTextNode(seg.slice(last, idx)));
      const mark = document.createElement('mark');
      mark.className = 't-hit';
      mark.textContent = seg.slice(idx, idx + query.length);
      el.appendChild(mark);
      last = idx + query.length;
      idx = lower.indexOf(q, last);
    }
    if (last < seg.length) el.appendChild(document.createTextNode(seg.slice(last)));
    box.appendChild(el);
  }

  if (!totalHits) {
    const none = document.createElement('div');
    none.className = 't-no-hits';
    none.textContent = `אין תוצאות עבור "${query}"`;
    box.appendChild(none);
    countEl.textContent = '';
  } else {
    countEl.textContent = totalHits.toLocaleString('he-IL') + ' תוצאות';
  }
  box.scrollTop = 0;
  linkifyTimestamps(box);  // no-op unless an audio player is active
}

export function onLiveSearch(value) {
  liveSearchQuery = value.trim();
  const box     = document.getElementById('live-transcript-box');
  const countEl = document.getElementById('live-search-count');
  if (!liveSearchQuery) {
    countEl.textContent = '';
    box.textContent = liveTranscriptText;
    box.appendChild(makeCursor());
    if (!transcriptPinned) box.scrollTop = box.scrollHeight;
    return;
  }
  renderTranscriptSearch(box, liveTranscriptText, liveSearchQuery, countEl);
}

export function onFinalSearch(value) {
  const q       = value.trim();
  const box     = document.getElementById('transcript-box');
  const countEl = document.getElementById('final-search-count');
  const full    = (store.currentResult && store.currentResult.transcript) || '';
  if (!q) {
    countEl.textContent = '';
    box.textContent = full;
    linkifyTimestamps(box);
    return;
  }
  if (box.style.display !== 'block') toggleTranscript();  // auto-open results
  renderTranscriptSearch(box, full, q, countEl);
}

export function toggleTranscript() {
  const box = document.getElementById('transcript-box');
  const tog = document.querySelector('.transcript-toggle');
  const open = box.style.display === 'block';
  box.style.display = open ? 'none' : 'block';
  tog.textContent = (open ? '▶' : '▼') + ' ' + (open ? 'הצג תמלול גולמי' : 'הסתר תמלול');
}
