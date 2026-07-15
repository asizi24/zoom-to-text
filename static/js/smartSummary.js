// On-demand Smart Summary: trigger Map-Reduce generation for the current
// completed task, poll to completion, and render the Obsidian markdown safely
// (DOM APIs only — the markdown is LLM-generated, so never innerHTML).
import { store } from './store.js';
import { apiFetch, apiJson } from './api.js';
import { toast } from './ui.js';

let taskId = null;      // owned here (store.currentTaskId isn't set for history loads)
let markdown = '';
let pollTimer = null;

const $ = (id) => document.getElementById(id);

// ── Lifecycle hooks (called from results.js / main.js) ──────────────────────────

export function resetSmartSummary() {
  stopPoll();
  taskId = null;
  markdown = '';
  const section = $('smart-summary-section');
  if (section) section.style.display = 'none';
  hide('ss-actions'); hide('ss-status'); hide('ss-error');
  const render = $('ss-render');
  if (render) { render.textContent = ''; render.style.display = 'none'; }
  setBtn(false, '✨ צור סיכום חכם');
}

// Called when a completed task is shown — surfaces an existing summary (so a
// reload doesn't lose it) or resumes polling a run that's still in flight.
export async function hydrateSmartSummary(id) {
  resetSmartSummary();
  taskId = id;
  if (!taskId) return;
  try {
    const data = await apiJson(`/api/tasks/${taskId}/smart-summary`);
    if (data.status === 'completed' && data.markdown) {
      showResult(data.markdown);
    } else if (data.status === 'pending' || data.status === 'running') {
      showSection();
      showStatus('ממשיך ליצור סיכום חכם…');
      setBtn(true, '✨ יוצר…');
      startPoll();
    }
    // idle / failed → leave the panel hidden; the button starts generation.
  } catch { /* no summary yet — expected */ }
}

// ── Trigger ─────────────────────────────────────────────────────────────────────

export async function generateSmartSummary() {
  if (!taskId) { toast('אין משימה פעילה'); return; }
  showSection();
  showStatus('יוצר סיכום חכם… זה עשוי לקחת דקה.');
  const render = $('ss-render');
  render.style.display = 'none';
  hide('ss-actions'); hide('ss-error');
  setBtn(true, '✨ יוצר…');

  try {
    const resp = await apiFetch(`/api/tasks/${taskId}/smart-summary`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'completed' && data.markdown) { showResult(data.markdown); return; }
    startPoll();
  } catch (err) {
    showError(err.message || 'לא ניתן ליצור סיכום חכם');
  }
}

function startPoll() {
  stopPoll();
  pollTimer = setInterval(async () => {
    let data;
    try { data = await apiJson(`/api/tasks/${taskId}/smart-summary`); }
    catch { return; }   // transient — try again next tick
    if (data.status === 'completed' && data.markdown) showResult(data.markdown);
    else if (data.status === 'failed') showError(data.error || 'יצירת הסיכום נכשלה');
  }, 1500);
}

function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

// ── Actions ─────────────────────────────────────────────────────────────────────

export function copySmartSummary(btn) {
  if (!markdown) return;
  navigator.clipboard.writeText(markdown).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✅ הועתק';
    setTimeout(() => { btn.textContent = orig; }, 1800);
  });
}

export function downloadSmartSummary() {
  if (!markdown) return;
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filenameFor(markdown);
  a.click();
  URL.revokeObjectURL(a.href);
}

function filenameFor(md) {
  const m = md.match(/^title:\s*"?(.+?)"?\s*$/m);
  const base = (m ? m[1] : (store.currentSource || 'smart-summary'))
    .replace(/[\\/:*?"<>|#^[\]]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80);
  return `${base || 'smart-summary'} — ${new Date().toISOString().slice(0, 10)}.md`;
}

// ── View state ──────────────────────────────────────────────────────────────────

function showSection() { $('smart-summary-section').style.display = 'block'; }
function hide(id) { const el = $(id); if (el) el.style.display = 'none'; }

function showStatus(text) {
  $('ss-status').style.display = 'flex';
  $('ss-status-text').textContent = text;
  hide('ss-error');
}

function showError(text) {
  stopPoll();
  hide('ss-status');
  const e = $('ss-error');
  e.style.display = 'block';
  e.textContent = '⚠️ ' + text;
  setBtn(false, markdown ? '✨ צור מחדש' : '✨ צור סיכום חכם');
  $('ss-actions').style.display = markdown ? 'flex' : 'none';
}

function showResult(md) {
  markdown = md;
  stopPoll();
  showSection();
  hide('ss-status'); hide('ss-error');
  const render = $('ss-render');
  render.style.display = 'block';
  renderMarkdown(md, render);
  $('ss-actions').style.display = 'flex';
  setBtn(false, '✨ צור מחדש');
}

function setBtn(disabled, label) {
  const btn = $('smart-summary-btn');
  if (!btn) return;
  btn.disabled = disabled;
  if (label) btn.textContent = label;
}

// ── Safe markdown renderer (DOM only, no innerHTML) ──────────────────────────────

function renderMarkdown(md, container) {
  container.textContent = '';
  const lines = stripFrontmatter(md).split('\n');
  let list = null;
  const flushList = () => { if (list) { container.appendChild(list); list = null; } };

  for (let i = 0; i < lines.length;) {
    const line = lines[i];

    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {                              // fenced code block (LTR)
      flushList();
      const code = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) { code.push(lines[i]); i++; }
      i++;                                    // skip closing fence
      const pre = document.createElement('pre');
      pre.className = 'ss-code';
      pre.dir = 'ltr';
      const codeEl = document.createElement('code');
      if (fence[1]) codeEl.className = `language-${fence[1]}`;
      codeEl.textContent = code.join('\n');
      pre.appendChild(codeEl);
      container.appendChild(pre);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushList();
      const el = document.createElement(heading[1].length <= 2 ? 'h3' : 'h4');
      el.className = 'ss-h';
      appendInline(el, heading[2]);
      container.appendChild(el);
      i++; continue;
    }

    const item = line.match(/^\s*[-*]\s+(.*)$/);
    if (item) {
      if (!list) { list = document.createElement('ul'); list.className = 'ss-ul'; }
      const li = document.createElement('li');
      appendInline(li, item[1]);
      list.appendChild(li);
      i++; continue;
    }

    if (!line.trim()) { flushList(); i++; continue; }

    // paragraph: gather consecutive plain lines
    flushList();
    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim()
           && !/^(#{1,6}\s|```|\s*[-*]\s)/.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    const p = document.createElement('p');
    p.className = 'ss-p';
    p.dir = 'auto';
    appendInline(p, para.join(' '));
    container.appendChild(p);
  }
  flushList();
}

// Inline **bold** and `code`; everything else is plain text (textContent only).
function appendInline(el, text) {
  const re = /(\*\*([^*]+)\*\*|`([^`]+)`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) el.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[2] !== undefined) {
      const strong = document.createElement('strong');
      strong.textContent = m[2];
      el.appendChild(strong);
    } else {
      const code = document.createElement('code');
      code.className = 'ss-inline-code';
      code.dir = 'ltr';
      code.textContent = m[3];
      el.appendChild(code);
    }
    last = re.lastIndex;
  }
  if (last < text.length) el.appendChild(document.createTextNode(text.slice(last)));
}

function stripFrontmatter(md) {
  if (md.startsWith('---')) {
    const end = md.indexOf('\n---', 3);
    if (end !== -1) {
      const nl = md.indexOf('\n', end + 1);
      return nl !== -1 ? md.slice(nl + 1).trim() : '';
    }
  }
  return md;
}
