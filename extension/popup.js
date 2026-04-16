/**
 * Zoom Transcriber — Chrome Extension Popup
 *
 * Flow:
 *  1. User opens a Zoom recording page (already authenticated)
 *  2. User clicks the extension icon → this popup opens
 *  3. content.js (injected in the Zoom page) extracts the CDN video URL
 *     and the session cookies from the authenticated browser context
 *  4. popup.js sends URL + cookies to the backend → receives a task_id
 *  5. Polls GET /api/tasks/{task_id} every 2.5s and shows live progress
 *  6. When done → shows a "View Results" button that opens the web UI
 *
 * The server URL is stored in chrome.storage.local.
 * Default: http://localhost:8000 (local Docker)
 * Change to your EC2 public IP/domain in the settings panel.
 */

const DEFAULT_SERVER = 'http://localhost:8000';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusEl     = document.getElementById('status');
const sendBtn      = document.getElementById('send-btn');
const settingsLink = document.getElementById('settings-link');
const serverInput  = document.getElementById('server-url');
const settingsDiv  = document.getElementById('settings-panel');
const saveBtn      = document.getElementById('save-settings');
const progressWrap = document.getElementById('progress-wrap');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const resultLink   = document.getElementById('result-link');

let serverUrl    = DEFAULT_SERVER;
let pollInterval = null;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  const stored = await chrome.storage.local.get(['serverUrl']);
  serverUrl = stored.serverUrl || DEFAULT_SERVER;
  serverInput.value = serverUrl;

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const isZoomPage = tab?.url?.includes('zoom.us');

  if (!isZoomPage) {
    setStatus('⚠️ פתח דף הקלטת Zoom תחילה', 'warn');
    sendBtn.disabled = true;
    return;
  }

  setStatus('✅ דף Zoom זוהה — לחץ לשליחה', 'ok');
  sendBtn.disabled = false;
});

// ── Settings ──────────────────────────────────────────────────────────────────
settingsLink.addEventListener('click', () => {
  settingsDiv.style.display = settingsDiv.style.display === 'none' ? 'block' : 'none';
});

saveBtn.addEventListener('click', async () => {
  serverUrl = serverInput.value.trim().replace(/\/$/, '');
  await chrome.storage.local.set({ serverUrl });
  settingsDiv.style.display = 'none';
  setStatus(`💾 שמור — שרת: ${serverUrl}`, 'ok');
});

// ── Send to server ────────────────────────────────────────────────────────────
sendBtn.addEventListener('click', async () => {
  sendBtn.disabled = true;
  resultLink.style.display = 'none';
  setStatus('📡 מחלץ מידע מהדף...', 'info');

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // Ask the content script for the CDN video URL + cookies
    let pageData = {};
    try {
      pageData = await chrome.tabs.sendMessage(tab.id, { action: 'getZoomData' });
    } catch {
      // Content script not yet active — inject it on demand
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
      // Small delay to let it initialize
      await new Promise(r => setTimeout(r, 400));
      pageData = await chrome.tabs.sendMessage(tab.id, { action: 'getZoomData' });
    }

    const recordingUrl = pageData?.videoUrl || tab.url;
    const cookies      = pageData?.cookies  || null;

    if (!cookies) {
      setStatus('⚠️ לא נמצאו cookies — ייתכן שהסרטון ייכשל אם הוא פרטי', 'warn');
    } else {
      setStatus('🚀 שולח לשרת לעיבוד...', 'info');
    }

    const response = await fetch(`${serverUrl}/api/tasks`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url:      recordingUrl,
        mode:     'gemini_direct',
        cookies:  cookies,
        language: 'he',
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `שגיאת שרת ${response.status}`);
    }

    const task = await response.json();
    setStatus(`⏳ עיבוד החל...`, 'info');
    showProgress(5, 'מתחיל...');
    startPolling(task.task_id);

  } catch (err) {
    setStatus(`❌ ${err.message}`, 'error');
    sendBtn.disabled = false;
  }
});

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling(taskId) {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const r = await fetch(`${serverUrl}/api/tasks/${taskId}`);
      if (!r.ok) return;
      const task = await r.json();

      showProgress(task.progress, task.message || '');

      if (task.status === 'completed') {
        clearInterval(pollInterval);
        setStatus('✅ הסיכום מוכן!', 'ok');
        showProgress(100, 'הושלם!');
        resultLink.href          = `${serverUrl}/?task=${taskId}`;
        resultLink.style.display = 'block';
        sendBtn.disabled         = false;

      } else if (task.status === 'failed') {
        clearInterval(pollInterval);
        setStatus(`❌ ${task.error || 'העיבוד נכשל'}`, 'error');
        sendBtn.disabled = false;
      }
    } catch { /* retry on next tick */ }
  }, 2500);
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(msg, type = 'info') {
  statusEl.textContent = msg;
  statusEl.className   = `status status-${type}`;
}

function showProgress(pct, msg) {
  progressWrap.style.display = 'block';
  progressFill.style.width   = pct + '%';
  const label = msg.replace(/[⬇️✅🤖🎙️]/gu, '').trim();
  progressText.textContent   = `${label} (${pct}%)`;
}
