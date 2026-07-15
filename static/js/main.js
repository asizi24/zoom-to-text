// Application entry point: wires the modules together, owns the global
// keyboard shortcuts and the full-view reset, and exposes the handful of
// functions still referenced by inline onclick= attributes in index.html.
import { store } from './store.js';
import { openAbout, closeAbout, hideError, initUiEffects } from './ui.js';
import {
  initUpload, switchTab, selectMode, submit, clearSelectedFiles,
} from './upload.js';
import { stopCreep } from './progress.js';
import {
  startLiveUpdates, stopLiveUpdates, cancelCurrentTask,
  resumeTask, resumeLatestInFlight,
} from './live.js';
import {
  toggleTranscriptPin, onLiveSearch, onFinalSearch, toggleTranscript,
  hideLiveTranscript,
} from './transcript.js';
import { showResults, checkAnswer, copySection, copyAll, resetQuizScore } from './results.js';
import { exportMarkdown, exportObsidian } from './exports.js';
import {
  renderFlashcards, resetFlashcards, flipFlashcard,
  nextFlashcard, prevFlashcard, markFlashcard,
} from './flashcards.js';
import { sendChat, clearChat, hideChatPanel } from './chat.js';
import { togglePlayback, scrubAudio, hideAudioPlayer, initPlayerKeys } from './player.js';
import { toggleHistory } from './history.js';

// ── Full view reset ("process another lesson") ────────────────────────────────
function reset() {
  stopLiveUpdates();
  stopCreep();
  store.taskFinished = false;
  hideLiveTranscript();
  hideChatPanel();
  hideAudioPlayer();
  document.getElementById('results-card').style.display = 'none';
  document.getElementById('input-card').style.display   = 'block';
  document.querySelector('.brand').style.display        = 'block';
  document.querySelector('.stats').style.display        = 'flex';
  document.getElementById('url-input').value = '';
  clearSelectedFiles();
  store.currentTaskId = null;
  store.currentResult = null;
  store.currentSource = '';
  store.processingStartTime = null;
  const batchNote = document.getElementById('batch-note');
  if (batchNote) batchNote.style.display = 'none';
  resetQuizScore();
  document.getElementById('score-badge').style.display = 'none';
  document.getElementById('chapters-section').style.display = 'block';
  document.getElementById('transcript-section').style.display = 'none';
  document.getElementById('flashcards-section').style.display = 'none';
  resetFlashcards();
  hideError();
  switchTab('url');
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && document.getElementById('input-card').style.display !== 'none') {
    const active = document.activeElement;
    // Only fire when the URL input is focused OR nothing specific is focused
    if (!active || active === document.body || active.id === 'url-input') {
      e.preventDefault();
      submit();
    }
  }
  if (e.key === 'Escape' && document.getElementById('results-card').style.display === 'block') {
    reset();
  }
});

// ── Wiring ────────────────────────────────────────────────────────────────────
initUpload();
initUiEffects();
initPlayerKeys();

// ── Transitional bridge for inline onclick= handlers ──────────────────────────
// The markup still uses inline handlers; they resolve against window. As
// sections get eventized (addEventListener in their module), remove their
// entry here. New code must NOT rely on these globals — import instead.
Object.assign(window, {
  openAbout, closeAbout,
  switchTab, selectMode, submit,
  cancelCurrentTask,
  toggleTranscriptPin, onLiveSearch, onFinalSearch, toggleTranscript,
  togglePlayback, scrubAudio,
  exportObsidian, exportMarkdown, copyAll, copySection,
  flipFlashcard, prevFlashcard, nextFlashcard, markFlashcard,
  sendChat, clearChat,
  toggleHistory,
  reset,
  checkAnswer,
});

// ── Startup ───────────────────────────────────────────────────────────────────
// Auto-load a specific task when arriving from the Chrome extension
// (?task=<id>); otherwise re-attach to the newest in-flight task so a page
// refresh never orphans a running job.
const taskFromUrl = new URLSearchParams(location.search).get('task');
if (taskFromUrl) {
  resumeTask(taskFromUrl);
} else {
  resumeLatestInFlight();
}
