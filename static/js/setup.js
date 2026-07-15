// First-boot Setup Wizard controller.
//
// Standalone ES module (loaded only by static/setup.html, which GET / serves
// until setup is complete). Flow: probe hardware → recommend a backend →
// either pull a local Ollama model (SSE progress) or save a Gemini key →
// POST /api/setup/complete → reload into the app.
//
// No imports from the main app bundle: the wizard runs before login and must
// stand entirely on its own.

const $ = (id) => document.getElementById(id);

let selectedBackend = null;
let installedModels = [];
let defaultModel = 'gemma2:9b';
let pullSource = null;   // active EventSource during a model pull
let pullCompleted = false;

// ── Boot ───────────────────────────────────────────────────────────────────────
init();

async function init() {
  // If somehow reached after setup, bounce to the app.
  try {
    const status = await fetchJson('/api/setup/status');
    if (status.setup_complete) { location.href = '/'; return; }
  } catch { /* keep going — the wizard is still usable */ }

  wireOptionCards();
  await loadHardware();
}

// ── Hardware probe ──────────────────────────────────────────────────────────────
async function loadHardware() {
  let hw;
  try {
    hw = await fetchJson('/api/setup/hardware');
  } catch (err) {
    // Probe failed — let the user choose manually, default to the cloud path.
    renderHardware({
      cpu_only: true, strong_gpu: false, recommended_backend: 'gemini',
      detail: 'לא הצלחנו לזהות את החומרה אוטומטית — אפשר לבחור ידנית.',
      gpu_name: null, ollama_available: false, installed_models: [],
    });
    return;
  }

  defaultModel = hw.default_model || hw.recommended_model || defaultModel;
  installedModels = hw.installed_models || [];
  renderHardware(hw);
}

function renderHardware(hw) {
  const el = $('hw');
  el.classList.remove('detecting');

  if (hw.strong_gpu) {
    $('hw-icon').textContent = '⚡';
    $('hw-title').textContent = `זוהה GPU חזק${hw.gpu_name ? ' · ' + hw.gpu_name : ''}`;
  } else if (hw.cpu_only) {
    $('hw-icon').textContent = '💻';
    $('hw-title').textContent = 'לא זוהה GPU חזק — מצב CPU';
  } else {
    $('hw-icon').textContent = '🖥️';
    $('hw-title').textContent = `זוהה GPU${hw.gpu_name ? ' · ' + hw.gpu_name : ''}`;
  }
  $('hw-note').textContent = hw.detail || '';

  // Reflect the model name + install state on the Ollama card.
  $('ollama-desc').textContent =
    `רץ כולו על המחשב שלך — פרטי, ללא עלות ענן. מודל מומלץ: ${defaultModel}.`;
  const isInstalled = installedModels.includes(defaultModel);
  $('ollama-installed').hidden = !isInstalled;
  $('ollama-cta').textContent = isInstalled ? 'השתמש במודל המותקן וסיים' : 'הורד את המודל וסיים';

  // Mark + pre-select the recommended backend.
  const rec = hw.recommended_backend === 'gemini' ? 'gemini' : 'ollama';
  $(rec === 'gemini' ? 'gemini-rec' : 'ollama-rec').hidden = false;
  selectBackend(rec);
}

// ── Option selection ────────────────────────────────────────────────────────────
function wireOptionCards() {
  for (const id of ['opt-ollama', 'opt-gemini']) {
    const card = $(id);
    card.addEventListener('click', () => selectBackend(card.dataset.backend));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectBackend(card.dataset.backend); }
    });
  }
  $('ollama-cta').addEventListener('click', startOllama);
  $('gemini-cta').addEventListener('click', saveGemini);
  $('gemini-key').addEventListener('keydown', (e) => { if (e.key === 'Enter') saveGemini(); });
}

function selectBackend(backend) {
  if (pullSource) return;   // don't switch mid-download
  selectedBackend = backend;
  $('opt-ollama').classList.toggle('selected', backend === 'ollama');
  $('opt-gemini').classList.toggle('selected', backend === 'gemini');
  $('detail-ollama').classList.toggle('show', backend === 'ollama');
  $('detail-gemini').classList.toggle('show', backend === 'gemini');
  hideMsg();
  if (backend === 'gemini') $('gemini-key').focus();
}

// ── Ollama path: pull model (SSE), then finish ──────────────────────────────────
async function startOllama() {
  hideMsg();
  pullCompleted = false;

  if (installedModels.includes(defaultModel)) {
    // Already pulled — skip straight to finishing.
    await finish({ backend: 'ollama', ollama_model: defaultModel });
    return;
  }

  setBusy('ollama-cta', true, 'מוריד…');
  $('pull-progress').classList.add('show');
  setBar(null, 'מתחבר ל-Ollama…');

  const url = `/api/setup/ollama/pull/stream?model=${encodeURIComponent(defaultModel)}`;
  pullSource = new EventSource(url);

  pullSource.onmessage = async (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }

    if (ev.type === 'progress') {
      setBar(ev.percent, translateStatus(ev.status, ev.percent));
    } else if (ev.type === 'done') {
      pullCompleted = true;
      closePull();
      setBar(100, 'הורדה הושלמה — מסיים…');
      await finish({ backend: 'ollama', ollama_model: defaultModel });
    } else if (ev.type === 'error') {
      closePull();
      failPull(ev.message || 'הורדת המודל נכשלה');
    }
  };

  // EventSource fires onerror on network drop AND on normal stream close.
  // Only treat it as a failure if we never received a terminal 'done'.
  pullSource.onerror = () => {
    if (pullCompleted) return;
    closePull();
    failPull('החיבור ל-Ollama נכשל. ודא שהשירות פועל ונסה שוב.');
  };
}

function closePull() {
  if (pullSource) { pullSource.close(); pullSource = null; }
}

function failPull(message) {
  $('pull-progress').classList.remove('show');
  setBusy('ollama-cta', false, 'נסה שוב');
  showError(message);
}

// ── Gemini path: save key, then finish ──────────────────────────────────────────
async function saveGemini() {
  hideMsg();
  const key = $('gemini-key').value.trim();
  if (!key) { showError('אנא הכנס מפתח Gemini API'); return; }
  setBusy('gemini-cta', true, 'שומר…');
  try {
    await finish({ backend: 'gemini', gemini_api_key: key });
  } catch {
    setBusy('gemini-cta', false, 'שמור וסיים');
  }
}

// ── Finish: persist + reload ────────────────────────────────────────────────────
async function finish(payload) {
  try {
    const resp = await fetch('/api/setup/complete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || 'ההגדרה נכשלה');
    }
    showSuccess('מצוין! ההגדרה הושלמה. טוען את האפליקציה…');
    setTimeout(() => { location.href = '/'; }, 900);
  } catch (err) {
    showError(err.message || 'ההגדרה נכשלה — נסה שוב');
    throw err;
  }
}

// ── Progress bar helpers ─────────────────────────────────────────────────────────
function setBar(percent, statusText) {
  const fill = $('pull-bar');
  if (percent == null) {
    fill.classList.add('indeterminate');
  } else {
    fill.classList.remove('indeterminate');
    fill.style.width = `${percent}%`;
  }
  if (statusText != null) $('pull-status').textContent = statusText;
}

function translateStatus(status, percent) {
  const map = {
    'pulling manifest': 'מוריד מניפסט…',
    'verifying sha256 digest': 'מאמת קובץ…',
    'writing manifest': 'כותב מניפסט…',
    'removing any unused layers': 'מנקה שכבות…',
    'success': 'הושלם',
  };
  if (map[status]) return map[status];
  if (status && status.startsWith('downloading')) {
    return percent != null ? `מוריד את המודל… ${percent}%` : 'מוריד את המודל…';
  }
  return status || 'עובד…';
}

// ── Small utilities ─────────────────────────────────────────────────────────────
async function fetchJson(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function setBusy(btnId, busy, label) {
  const btn = $(btnId);
  btn.disabled = busy;
  if (label != null) btn.textContent = label;
}

function showError(text)   { const m = $('msg'); m.className = 'msg error';   m.textContent = text; }
function showSuccess(text) { const m = $('msg'); m.className = 'msg success'; m.textContent = text; }
function hideMsg()         { const m = $('msg'); m.className = 'msg';          m.textContent = ''; }
