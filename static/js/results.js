// Results card: render the completed lesson (summary, chapters, transcript,
// quiz) and the copy helpers. All rendering uses DOM APIs — no innerHTML —
// to keep model output XSS-safe.
import { store } from './store.js';
import { hideProgress } from './progress.js';
import { hideLiveTranscript } from './transcript.js';
import { linkifyTimestamps, showAudioPlayer } from './player.js';
import { renderFlashcards } from './flashcards.js';
import { showChatPanel } from './chat.js';
import { buildMarkdown } from './exports.js';
import { hydrateSmartSummary } from './smartSummary.js';

let quizScore = { correct: 0, total: 0 };

export function resetQuizScore() {
  quizScore = { correct: 0, total: 0 };
}

export function showResults(result, taskId, hasAudio) {
  hideLiveTranscript();
  hideProgress();
  store.currentResult = result;
  document.getElementById('input-card').style.display = 'none';
  document.querySelector('.brand').style.display = 'none';
  document.querySelector('.stats').style.display = 'none';

  // Show the audio player first so linkifyTimestamps knows a player is active.
  // linkifyTimestamps is a no-op when no player is active, so calling it on
  // text BEFORE showAudioPlayer would render inert tokens.
  if (hasAudio && taskId) showAudioPlayer(taskId);

  // Summary
  document.getElementById('summary-text').textContent = result.summary || '';
  linkifyTimestamps(document.getElementById('summary-text'));

  // Chapters
  const cl = document.getElementById('chapters-list');
  cl.textContent = '';
  (result.chapters || []).forEach(ch => {
    const chDiv = document.createElement('div');
    chDiv.className = 'chapter';

    const titleDiv = document.createElement('div');
    titleDiv.className = 'chapter-title';
    titleDiv.textContent = ch.title || '';
    chDiv.appendChild(titleDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'chapter-content';
    contentDiv.textContent = ch.content || '';
    chDiv.appendChild(contentDiv);

    if ((ch.key_points || []).length) {
      const ul = document.createElement('ul');
      ul.className = 'key-points';
      (ch.key_points || []).forEach(p => {
        const li = document.createElement('li');
        li.textContent = p;
        ul.appendChild(li);
      });
      chDiv.appendChild(ul);
    }

    cl.appendChild(chDiv);
  });
  if (!(result.chapters || []).length)
    document.getElementById('chapters-section').style.display = 'none';
  else
    linkifyTimestamps(cl);

  // Transcript (Whisper mode only) — with search
  if (result.transcript) {
    document.getElementById('transcript-box').textContent = result.transcript;
    document.getElementById('transcript-section').style.display = 'block';
    document.getElementById('final-search-wrap').style.display = 'flex';
    document.getElementById('final-search-input').value = '';
    document.getElementById('final-search-count').textContent = '';
    linkifyTimestamps(document.getElementById('transcript-box'));
  }

  // Quiz
  quizScore = { correct: 0, total: 0 };
  const ql = document.getElementById('quiz-list');
  ql.textContent = '';
  (result.quiz || []).forEach((q, i) => {
    quizScore.total++;

    const card = document.createElement('div');
    card.className = 'q-card';

    const numDiv = document.createElement('div');
    numDiv.className = 'q-num';
    numDiv.textContent = `שאלה ${i + 1}`;
    card.appendChild(numDiv);

    const qText = document.createElement('div');
    qText.className = 'q-text';
    qText.textContent = q.question || '';
    card.appendChild(qText);

    const optsDiv = document.createElement('div');
    optsDiv.className = 'options';
    (q.options || []).forEach(o => {
      const btn = document.createElement('button');
      btn.className = 'opt-btn';
      btn.textContent = o;
      btn.addEventListener('click', function () { checkAnswer(this, o, q.correct_answer); });
      optsDiv.appendChild(btn);
    });
    card.appendChild(optsDiv);

    const expDiv = document.createElement('div');
    expDiv.className = 'explanation';
    expDiv.textContent = `💡 ${q.explanation || ''}`;
    card.appendChild(expDiv);

    ql.appendChild(card);
  });
  linkifyTimestamps(ql);

  document.getElementById('results-card').style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });

  renderFlashcards(result.flashcards || [], taskId);
  if (taskId) showChatPanel(taskId);
  hydrateSmartSummary(taskId);
}

export function checkAnswer(btn, selected, correct) {
  const card    = btn.closest('.q-card');
  const allBtns = card.querySelectorAll('.opt-btn');
  const expDiv  = card.querySelector('.explanation');

  allBtns.forEach(b => {
    b.disabled = true;
    if (b.textContent.trim() === correct.trim()) b.classList.add('correct');
  });

  const isCorrect = selected.trim() === correct.trim();
  if (!isCorrect) btn.classList.add('wrong');
  else { quizScore.correct++; }

  if (expDiv) expDiv.style.display = 'block';

  // Update score badge
  const badge = document.getElementById('score-badge');
  badge.style.display = 'inline-block';
  badge.textContent = `ציון: ${quizScore.correct} / ${quizScore.total}`;

  const pct = Math.round((quizScore.correct / quizScore.total) * 100);
  badge.style.background =
    pct >= 80 ? 'linear-gradient(135deg,#10B981,#059669)' :
    pct >= 60 ? 'linear-gradient(135deg,#F59E0B,#D97706)' :
                'linear-gradient(135deg,#EF4444,#DC2626)';
}

// ── Copy helpers ──────────────────────────────────────────────────────────────

export function copySection(elementId, btn) {
  const el = document.getElementById(elementId);
  const text = el ? el.innerText : '';
  navigator.clipboard.writeText(text).then(() => {
    btn.classList.add('copied');
    const orig = btn.innerHTML;
    btn.innerHTML = '✅ הועתק';
    setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = orig; }, 2000);
  });
}

export function copyAll() {
  if (!store.currentResult) return;
  const md = buildMarkdown(store.currentResult);
  navigator.clipboard.writeText(md).then(() => {
    const btn = document.getElementById('copy-all-btn');
    btn.classList.add('success-flash');
    const orig = btn.innerHTML;
    btn.innerHTML = '✅ הועתק';
    setTimeout(() => { btn.classList.remove('success-flash'); btn.innerHTML = orig; }, 2000);
  });
}
