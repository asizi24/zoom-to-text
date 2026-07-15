// Progress card: phase stepper, bar + creep animation, ETA estimate,
// batch note, and the cancel-button enabled state.
import { store } from './store.js';

let creepTimer       = null;
let lastRealProgress = 0;
let lastProgressTime = 0;
let visualProgress   = 0;

export function setCancelEnabled(on) {
  const btn = document.getElementById('cancel-btn');
  if (btn) btn.disabled = !on;
}

function applyProgressBar(pct) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-pct').textContent  = Math.round(pct) + '%';
}

// Typical total processing time in seconds per mode (2-hour recording baseline)
const MODE_TYPICAL_SEC = {
  gemini_direct: 240, whisper_api: 600, whisper_local: 1800,
  ivrit_ai: 1500, transcription_only: 1400,
};

function estimateRemaining(pct) {
  if (!store.processingStartTime || pct <= 5) return null;
  const elapsed = (Date.now() - store.processingStartTime) / 1000;

  // Linear interpolation from actual elapsed + current progress
  const linearRemaining = elapsed / (pct / 100) - elapsed;

  // Mode-based estimate: typical total minus elapsed
  const typical = MODE_TYPICAL_SEC[store.selectedMode] || 300;
  const modeRemaining = Math.max(0, typical - elapsed);

  // Early on (pct < 30) trust the mode estimate more; later trust linear more
  const weight = Math.min(1, pct / 60);
  return Math.max(0, linearRemaining * weight + modeRemaining * (1 - weight));
}

function updateEta(pct) {
  const etaEl = document.getElementById('progress-eta');
  const remaining = estimateRemaining(pct);
  if (remaining === null) { etaEl.textContent = ''; return; }

  if (remaining <= 8) {
    etaEl.textContent = '⏱️ כמעט מוכן...';
  } else {
    const mins = Math.floor(remaining / 60);
    const secs = Math.round(remaining % 60);
    const timeStr = mins > 0
      ? `${mins}:${String(secs).padStart(2, '0')} דקות`
      : `${secs} שניות`;
    etaEl.textContent = `⏱️ זמן משוער לסיכום: ${timeStr}`;
  }
}

// ── Phase stepper ──
const STEP_ORDER = ['ingest', 'transcribe', 'summarize', 'done'];

function stepFromStatus(status, pct) {
  if (status === 'completed' || pct >= 100) return 'done';
  if (status === 'summarizing') return 'summarize';
  if (status === 'transcribing') return pct >= 15 ? 'transcribe' : 'ingest';
  if (status === 'downloading' || status === 'pending') return 'ingest';
  return null;  // unknown status → keep current highlight
}

function updateSteps(status, pct) {
  const current = stepFromStatus(status, pct);
  if (!current) return;
  const curIdx = STEP_ORDER.indexOf(current);
  document.querySelectorAll('#progress-steps .p-step').forEach(el => {
    const idx = STEP_ORDER.indexOf(el.dataset.step);
    el.classList.toggle('done',   idx < curIdx || current === 'done');
    el.classList.toggle('active', idx === curIdx && current !== 'done');
  });
}

export function stopCreep() {
  if (creepTimer) { clearInterval(creepTimer); creepTimer = null; }
}

export function updateProgress(pct, msg, status) {
  // Stop existing creep — a real update arrived
  stopCreep();

  lastRealProgress = pct;
  lastProgressTime = Date.now();
  visualProgress   = pct;

  updateSteps(status, pct);
  applyProgressBar(pct);
  const label = document.getElementById('status-label');
  label.textContent = '';
  const spinner = document.createElement('span');
  spinner.className = 'spinner';
  label.appendChild(spinner);
  label.appendChild(document.createTextNode(msg || 'מעבד...'));

  updateEta(pct);

  // Start creep: after 8s without a real update, inch bar forward slowly
  if (pct < 100) {
    creepTimer = setInterval(() => {
      const stale = (Date.now() - lastProgressTime) / 1000;
      if (stale > 8) {
        const cap = Math.min(lastRealProgress + 30, 95);
        if (visualProgress < cap) {
          visualProgress = Math.min(visualProgress + 0.15, cap);
          applyProgressBar(visualProgress);
        }
        updateEta(visualProgress); // refresh ETA every second
      }
    }, 1000);
  }
}

// Small banner under the progress bar for batch uploads.
export function showBatchNote(count) {
  const el = document.getElementById('batch-note');
  if (!el) return;
  if (count > 1) {
    el.textContent = `📚 ${count} קבצים בתור — הראשון מעובד כעת, השאר ממתינים (ראה היסטוריה)`;
    el.style.display = 'block';
  } else {
    el.style.display = 'none';
  }
}

export function showProgress() {
  document.getElementById('input-card').style.display    = 'none';
  document.getElementById('progress-card').style.display = 'block';
  document.getElementById('progress-eta').textContent    = '';
  const note = document.getElementById('batch-note');
  if (note) note.style.display = 'none';
  setCancelEnabled(true);
  document.querySelectorAll('#progress-steps .p-step')
    .forEach(el => el.classList.remove('active', 'done'));
  updateProgress(5, 'מתחיל עיבוד...', 'pending');
}

export function hideProgress() {
  stopCreep();
  document.getElementById('progress-card').style.display = 'none';
}
