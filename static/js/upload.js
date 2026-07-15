// Input card: tabs, mode selector, file drag & drop, and job submission.
import { store } from './store.js';
import { apiFetch, postJson } from './api.js';
import { showError, hideError } from './ui.js';
import { showBatchNote, showProgress } from './progress.js';
import { resetLiveTranscript, showLiveTranscriptCard } from './transcript.js';
import { startLiveUpdates } from './live.js';

let selectedFiles = [];   // batch upload — one or more files

export function clearSelectedFiles() {
  selectedFiles = [];
  document.getElementById('file-badge').style.display = 'none';
}

export function switchTab(tab) {
  store.currentTab = tab;
  document.querySelectorAll('.tab').forEach((b, i) => {
    b.classList.toggle('active', (tab === 'url' && i === 0) || (tab === 'file' && i === 1));
  });
  document.getElementById('tab-url').classList.toggle('active', tab === 'url');
  document.getElementById('tab-file').classList.toggle('active', tab === 'file');
}

export function selectMode(el) {
  document.querySelectorAll('.mode-opt').forEach(m => m.classList.remove('selected'));
  el.classList.add('selected');
  store.selectedMode = el.dataset.mode;
}

function setFiles(fileList) {
  selectedFiles = Array.from(fileList);
  if (!selectedFiles.length) return;
  document.getElementById('file-name').textContent = selectedFiles.length === 1
    ? selectedFiles[0].name
    : `${selectedFiles.length} קבצים נבחרו`;
  document.getElementById('file-badge').style.display = 'flex';
}

export function initUpload() {
  const dropZone  = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('over');
    if (e.dataTransfer.files.length) setFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', e => { if (e.target.files.length) setFiles(e.target.files); });
}

export async function submit() {
  hideError();
  let taskId;
  let batchCount = 0;

  try {
    if (store.currentTab === 'url') {
      const url = document.getElementById('url-input').value.trim();
      if (!url) { showError('אנא הכנס קישור'); return; }
      store.currentSource = url;
      taskId = await submitUrl(url);
    } else {
      if (!selectedFiles.length) { showError('אנא בחר קובץ'); return; }
      // Upload every file sequentially. The queue (pipeline_concurrency=1)
      // processes them one by one; the live view attaches to the first.
      const ids = await submitFiles();
      taskId = ids[0];
      batchCount = ids.length;
      store.currentSource = batchCount > 1
        ? `${batchCount} קבצים`
        : selectedFiles[0].name;
    }
  } catch (err) {
    showError(err.message); return;
  }

  store.processingStartTime = Date.now();
  showProgress();
  resetLiveTranscript();
  // GEMINI_DIRECT produces no live transcript — its card appears lazily
  // (on first delta) for every other mode too, but showing it upfront
  // when we know text is coming feels more responsive.
  if (store.selectedMode !== 'gemini_direct') showLiveTranscriptCard();
  // For a batch, tell the user the rest are queued (they show in History).
  showBatchNote(batchCount);
  startLiveUpdates(taskId);
}

async function submitUrl(url) {
  const data = await postJson('/api/tasks', { url, mode: store.selectedMode, language: 'he' });
  return data.task_id;
}

// Upload all selected files sequentially. Returns the created task ids in
// submission order (ids[0] is the one that runs first under the FIFO queue).
async function submitFiles() {
  const ids = [];
  for (const f of selectedFiles) {
    const fd = new FormData();
    fd.append('file', f);
    fd.append('mode', store.selectedMode);
    fd.append('language', 'he');
    const r = await apiFetch('/api/tasks/upload', { method: 'POST', body: fd });
    ids.push((await r.json()).task_id);
  }
  return ids;
}
