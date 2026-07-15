// Flashcards deck: render, flip, navigate, self-scoring.
let flashcards = [];
let fcIndex = 0;
let fcKnew = 0;
let fcMissed = 0;
let fcSeen = new Set();

export function resetFlashcards() {
  flashcards = []; fcIndex = 0; fcKnew = 0; fcMissed = 0; fcSeen = new Set();
}

export function renderFlashcards(cards, taskId) {
  flashcards = cards || [];
  fcIndex = 0;
  fcKnew = 0;
  fcMissed = 0;
  fcSeen = new Set();
  const section = document.getElementById('flashcards-section');
  if (!flashcards.length) {
    section.style.display = 'none';
    return;
  }
  section.style.display = 'block';
  if (taskId) {
    document.getElementById('fc-apkg-btn').href =
      `/api/tasks/${taskId}/flashcards/export.apkg`;
    document.getElementById('fc-csv-btn').href =
      `/api/tasks/${taskId}/flashcards/export.csv`;
  }
  showFlashcard();
}

function showFlashcard() {
  if (!flashcards.length) return;
  const card = flashcards[fcIndex];
  document.getElementById('fc-front-text').textContent = card.front || '';
  document.getElementById('fc-back-text').textContent = card.back || '';
  // Tags — built with DOM APIs (no innerHTML) to prevent XSS
  const tagBox = document.getElementById('fc-front-tags');
  tagBox.textContent = '';
  (card.tags || []).forEach(t => {
    const span = document.createElement('span');
    span.className = 'fc-tag';
    span.textContent = t;
    tagBox.appendChild(span);
  });
  document.getElementById('fc-card').classList.remove('flipped');
  document.getElementById('fc-counter').textContent =
    `${fcIndex + 1} מתוך ${flashcards.length}`;
  document.getElementById('fc-prev-btn').disabled = fcIndex === 0;
  document.getElementById('fc-next-btn').disabled = fcIndex === flashcards.length - 1;
  updateFcScore();
}

export function flipFlashcard() {
  document.getElementById('fc-card').classList.toggle('flipped');
}

export function nextFlashcard() {
  if (fcIndex < flashcards.length - 1) { fcIndex++; showFlashcard(); }
}

export function prevFlashcard() {
  if (fcIndex > 0) { fcIndex--; showFlashcard(); }
}

export function markFlashcard(correct) {
  if (!flashcards.length) return;
  if (!fcSeen.has(fcIndex)) {
    fcSeen.add(fcIndex);
    if (correct) fcKnew++; else fcMissed++;
  }
  updateFcScore();
  if (fcIndex < flashcards.length - 1) {
    fcIndex++;
    showFlashcard();
  } else {
    // Final card: just re-render to update disabled state
    document.getElementById('fc-card').classList.remove('flipped');
  }
}

function updateFcScore() {
  const el = document.getElementById('fc-score');
  if (fcSeen.size === 0) { el.textContent = ''; return; }
  el.textContent = `סימנת ${fcSeen.size} מתוך ${flashcards.length} · ✓ ${fcKnew} · ✗ ${fcMissed}`;
}
