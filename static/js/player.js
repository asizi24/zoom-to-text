// Audio player (Feature 7): streams /api/tasks/{id}/audio with Range seek,
// plus the [MM:SS] linkifier that turns timestamps into seek anchors.
let currentAudioTaskId = null;

export function showAudioPlayer(taskId) {
  currentAudioTaskId = taskId;
  const audio = document.getElementById('audio-el');
  audio.src = `/api/tasks/${taskId}/audio`;
  audio.load();
  document.getElementById('audio-player-bar').classList.add('visible');
  audio.onplay  = () => setPlayIcon(true);
  audio.onpause = () => setPlayIcon(false);
  audio.ontimeupdate = updateScrubber;
  audio.onloadedmetadata = updateScrubber;
  audio.onerror = () => {
    // If the audio 404s (old task, file wiped) just hide the player silently
    hideAudioPlayer();
  };
}

export function hideAudioPlayer() {
  const audio = document.getElementById('audio-el');
  try { audio.pause(); } catch { /* not playing */ }
  audio.removeAttribute('src');
  try { audio.load(); } catch { /* no source */ }
  document.getElementById('audio-player-bar').classList.remove('visible');
  currentAudioTaskId = null;
}

function setPlayIcon(playing) {
  document.getElementById('ap-play-icon').textContent = playing ? '⏸' : '▶';
}

export function togglePlayback() {
  const audio = document.getElementById('audio-el');
  if (!audio.src) return;
  if (audio.paused) audio.play(); else audio.pause();
}

function updateScrubber() {
  const audio = document.getElementById('audio-el');
  const cur   = audio.currentTime || 0;
  const dur   = audio.duration || 0;
  const pct   = dur > 0 ? (cur / dur * 100) : 0;
  document.getElementById('ap-scrubber-fill').style.width = pct + '%';
  document.getElementById('ap-time').textContent = `${fmtSec(cur)} / ${fmtSec(dur)}`;
}

export function scrubAudio(evt) {
  const audio = document.getElementById('audio-el');
  const dur = audio.duration || 0;
  if (!dur) return;
  const bar = evt.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (evt.clientX - bar.left) / bar.width));
  audio.currentTime = ratio * dur;
}

function fmtSec(s) {
  s = Math.max(0, Math.floor(s || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m.toString().padStart(2, '0')}:${r.toString().padStart(2, '0')}`;
}

// Seek to a [MM:SS] string (called from ts-link clicks).
export function seekTo(tsStr) {
  const audio = document.getElementById('audio-el');
  if (!audio.src) return;
  const m = /^(\d+):(\d{2})$/.exec(tsStr);
  if (!m) return;
  audio.currentTime = parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
  audio.play().catch(() => {/* autoplay may be blocked — user clicks play */});
}

// Replace [MM:SS] occurrences inside an element's text with clickable spans.
// Uses text-node walking so we don't accidentally strip HTML from children.
const TS_RE = /\[(\d{1,2}):(\d{2})\]/g;

export function linkifyTimestamps(root) {
  if (!root || !currentAudioTaskId) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
  const targets = [];
  let n;
  while ((n = walker.nextNode())) {
    if (n.parentElement && n.parentElement.classList.contains('ts-link')) continue;
    if (TS_RE.test(n.nodeValue)) { targets.push(n); TS_RE.lastIndex = 0; }
  }
  for (const node of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    text.replace(TS_RE, (match, mm, ss, idx) => {
      if (idx > last) frag.appendChild(document.createTextNode(text.slice(last, idx)));
      const span = document.createElement('span');
      span.className = 'ts-link';
      span.textContent = `[${mm}:${ss}]`;
      const ts = `${mm}:${ss}`;
      span.addEventListener('click', () => seekTo(ts));
      frag.appendChild(span);
      last = idx + match.length;
      return match;
    });
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
}

// Spacebar = play/pause when player is visible and user isn't typing
export function initPlayerKeys() {
  document.addEventListener('keydown', e => {
    if (e.code !== 'Space') return;
    if (!document.getElementById('audio-player-bar').classList.contains('visible')) return;
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    e.preventDefault();
    togglePlayback();
  });
}
