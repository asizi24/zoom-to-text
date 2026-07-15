// Chat panel: multi-turn streaming chat about a completed lesson.
import { apiFetch, apiJson } from './api.js';
import { hideAudioPlayer, linkifyTimestamps } from './player.js';

let currentChatTaskId = null;
let chatStreaming = false;

export function showChatPanel(taskId) {
  currentChatTaskId = taskId;
  document.getElementById('chat-card').style.display = 'block';
  loadChatHistory();
}

export function hideChatPanel() {
  document.getElementById('chat-card').style.display = 'none';
  currentChatTaskId = null;
  const msgs = document.getElementById('chat-messages');
  if (msgs) msgs.textContent = '';
  const clearBtn = document.getElementById('chat-clear-btn');
  if (clearBtn) clearBtn.style.display = 'none';
  hideAudioPlayer();
}

async function loadChatHistory() {
  if (!currentChatTaskId) return;
  try {
    const data = await apiJson(`/api/tasks/${currentChatTaskId}/chat`);
    const msgs = document.getElementById('chat-messages');
    msgs.textContent = '';
    for (const msg of (data.history || [])) {
      appendChatBubble(msg.role, msg.content, false);
    }
    if (data.history && data.history.length > 0) {
      document.getElementById('chat-clear-btn').style.display = '';
      msgs.scrollTop = msgs.scrollHeight;
    }
  } catch (e) { console.warn('Failed to load chat history', e); }
}

function appendChatBubble(role, text, streaming = false) {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `chat-bubble ${role}${streaming ? ' streaming' : ''}`;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  // Only linkify finalized bubbles — during streaming text mutates and we
  // re-linkify in sendChat() when the stream ends.
  if (!streaming) linkifyTimestamps(div);
  return div;
}

export async function sendChat() {
  if (chatStreaming || !currentChatTaskId) return;
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  chatStreaming = true;
  document.getElementById('chat-send-btn').disabled = true;

  appendChatBubble('user', question, false);

  const modelBubble = appendChatBubble('model', '', true);
  let fullText = '';

  try {
    const r = await apiFetch(`/api/tasks/${currentChatTaskId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    if (!r.body) throw new Error('No response body');

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop(); // keep incomplete line
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload) continue;
        try {
          const evt = JSON.parse(payload);
          if (evt.text) {
            fullText += evt.text;
            modelBubble.textContent = fullText;
            const m = document.getElementById('chat-messages');
            m.scrollTop = m.scrollHeight;
          }
          if (evt.done) break;
          if (evt.error) throw new Error(evt.error);
        } catch { /* malformed chunk — skip */ }
      }
    }
  } catch (err) {
    modelBubble.textContent = `שגיאה: ${err.message}`;
  } finally {
    modelBubble.classList.remove('streaming');
    linkifyTimestamps(modelBubble);
    chatStreaming = false;
    document.getElementById('chat-send-btn').disabled = false;
    document.getElementById('chat-clear-btn').style.display = '';
  }
}

export async function clearChat() {
  if (!currentChatTaskId) return;
  try {
    await apiFetch(`/api/tasks/${currentChatTaskId}/chat`, { method: 'DELETE' });
  } catch { /* best-effort */ }
  document.getElementById('chat-messages').textContent = '';
  document.getElementById('chat-clear-btn').style.display = 'none';
}
