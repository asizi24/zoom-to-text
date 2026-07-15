// History card: recent tasks, reload completed results, retry failed ones,
// re-attach to running ones.
import { store } from './store.js';
import { apiJson, apiFetch } from './api.js';
import { showResults } from './results.js';
import { resumeTask } from './live.js';
import { toast } from './ui.js';

let historyVisible = false;

export async function toggleHistory() {
  historyVisible = !historyVisible;
  const card = document.getElementById('history-card');
  card.style.display = historyVisible ? 'block' : 'none';
  if (historyVisible) await loadHistory();
}

function emptyMessage(text) {
  const div = document.createElement('div');
  div.className = 'history-empty';
  div.textContent = text;
  return div;
}

async function loadHistory() {
  const container = document.getElementById('history-content');
  container.textContent = '';
  container.appendChild(emptyMessage('טוען...'));
  let tasks;
  try {
    tasks = await apiJson('/api/tasks?limit=30');
  } catch {
    container.textContent = '';
    container.appendChild(emptyMessage('שגיאה בטעינת ההיסטוריה'));
    return;
  }
  if (!tasks.length) {
    container.textContent = '';
    container.appendChild(emptyMessage('אין היסטוריה עדיין'));
    return;
  }
  const list = document.createElement('div');
  list.className = 'history-list';
  tasks.forEach(t => {
    const item = document.createElement('div');
    const isCompleted = t.status === 'completed';
    const isFailed    = t.status === 'failed';
    const isCancelled = t.status === 'cancelled';
    const isRetryable = isFailed || isCancelled;  // terminal but re-runnable
    item.className = `history-item${isRetryable ? ' dimmed' : ''}`;
    const statusIcon = isCompleted ? '✅' : isFailed ? '❌' : isCancelled ? '🚫' : '⏳';
    const badgeClass = isCompleted ? 'completed' : isFailed ? 'failed' : isCancelled ? 'cancelled' : 'running';
    const badgeText  = isCompleted ? 'הושלם' : isFailed ? 'נכשל' : isCancelled ? 'בוטל' : 'בתהליך';
    const dt = new Date(t.created_at).toLocaleString('he-IL', { dateStyle: 'short', timeStyle: 'short' });
    const shortUrl = (t.url || '').replace(/^upload:/, '📁 ').replace('https://', '').slice(0, 60);

    // Build history item with DOM APIs to prevent XSS from stored URLs
    const statusSpan = document.createElement('span');
    statusSpan.className = 'h-status';
    statusSpan.textContent = statusIcon;
    item.appendChild(statusSpan);

    const infoDiv = document.createElement('div');
    infoDiv.className = 'h-info';
    const urlDiv = document.createElement('div');
    urlDiv.className = 'h-url';
    urlDiv.textContent = shortUrl || 'לא ידוע';
    infoDiv.appendChild(urlDiv);
    const metaDiv = document.createElement('div');
    metaDiv.className = 'h-meta';
    metaDiv.textContent = dt;
    infoDiv.appendChild(metaDiv);
    item.appendChild(infoDiv);

    const badge = document.createElement('span');
    badge.className = `h-badge ${badgeClass}`;
    badge.textContent = badgeText;
    item.appendChild(badge);

    // Failed / cancelled tasks get a Retry button that re-queues the job.
    if (isRetryable) {
      const retryBtn = document.createElement('button');
      retryBtn.className = 'h-retry-btn';
      retryBtn.textContent = '↻ נסה שוב';
      retryBtn.onclick = (ev) => retryTask(t.id, ev);
      item.appendChild(retryBtn);
    }

    if (isCompleted) {
      item.onclick = () => loadHistoryTask(t.id);
    } else if (!isRetryable) {
      // Running task — clicking re-attaches the live progress view
      item.onclick = () => { toggleHistory(); resumeTask(t); };
    }
    list.appendChild(item);
  });
  container.textContent = '';
  container.appendChild(list);
}

async function loadHistoryTask(taskId) {
  try {
    const task = await apiJson(`/api/tasks/${taskId}`);
    if (task.result) {
      store.currentSource = task.url || '';
      toggleHistory();
      showResults(task.result, taskId, !!task.has_audio);
    }
  } catch (err) {
    toast(err.message || 'שגיאה בטעינת המשימה');
  }
}

// Re-queue a failed / cancelled task and attach the live progress view to it.
export async function retryTask(taskId, ev) {
  if (ev) ev.stopPropagation();  // don't trigger the row's own click
  try {
    const r = await apiFetch(`/api/tasks/${taskId}/retry`, { method: 'POST' });
    const task = await r.json();
    if (historyVisible) toggleHistory();
    resumeTask(task);
  } catch (err) {
    toast(err.message || 'לא ניתן להריץ מחדש את המשימה');
  }
}
