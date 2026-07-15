// Global UI chrome: the input-card error box, toast notifications (replacing
// alert()), the About modal, and the cosmetic pointer effects.

// ── Input-card error box ──────────────────────────────────────────────────────
export function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = '❌ ' + msg;
  box.style.display = 'block';
  document.getElementById('input-card').style.display = 'block';
}

export function hideError() {
  document.getElementById('error-box').style.display = 'none';
}

// ── Toasts ────────────────────────────────────────────────────────────────────
// Non-blocking replacement for alert(): stacked in the corner, auto-dismissed,
// click to dismiss early. kind: 'error' | 'success' | 'info'.
let toastContainer = null;

function ensureToastContainer() {
  if (toastContainer) return toastContainer;
  toastContainer = document.createElement('div');
  toastContainer.className = 'toast-container';
  toastContainer.setAttribute('dir', 'rtl');
  document.body.appendChild(toastContainer);
  return toastContainer;
}

export function toast(msg, kind = 'error', ttlMs = 5000) {
  const container = ensureToastContainer();
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.textContent = (kind === 'error' ? '❌ ' : kind === 'success' ? '✅ ' : 'ℹ️ ') + msg;
  el.addEventListener('click', () => dismiss());
  container.appendChild(el);

  let timer = setTimeout(dismiss, ttlMs);
  function dismiss() {
    if (!el.parentNode) return;
    clearTimeout(timer);
    el.classList.add('toast-out');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    // Fallback removal if the transition doesn't fire (reduced-motion)
    setTimeout(() => el.remove(), 400);
  }
  return dismiss;
}

// ── About modal (profile card) ────────────────────────────────────────────────
export function openAbout() {
  document.getElementById('about-overlay').classList.add('open');
}

export function closeAbout() {
  document.getElementById('about-overlay').classList.remove('open');
}

// ── Cosmetic pointer effects + modal escape key ───────────────────────────────
export function initUiEffects() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' &&
        document.getElementById('about-overlay').classList.contains('open')) {
      closeAbout();
    }
  });

  // Click pulse (universal click feedback): single delegated listener — any
  // click on an interactive element spawns a pulse at the cursor. Cheap (one
  // DOM node per click, auto-removed). Respects prefers-reduced-motion via CSS.
  const INTERACTIVE_SELECTOR = [
    'button', 'a[href]', '[role="button"]', 'input[type="submit"]',
    'input[type="button"]', '.tab', '.mode-opt', '.fc-card',
    '.history-item', '.opt-btn', '.profile-social-btn', '.ts-link',
    '.ap-play-btn', '.ap-scrubber', '.about-toggle', '.about-close',
  ].join(',');

  document.addEventListener('click', (e) => {
    const hit = e.target.closest(INTERACTIVE_SELECTOR);
    if (!hit || hit.disabled) return;
    const pulse = document.createElement('span');
    pulse.className = 'click-pulse';
    pulse.style.left = `${e.clientX}px`;
    pulse.style.top  = `${e.clientY}px`;
    document.body.appendChild(pulse);
    pulse.addEventListener('animationend', () => pulse.remove(), { once: true });
  }, true); // capture phase — fires even if the target stops propagation

  // Spotlight tracking (cursor → CSS vars → card glow). rAF-throttled so
  // pointermove at 120Hz doesn't thrash style recalculation.
  const root = document.documentElement;
  let pending = false;
  let px = 0, py = 0;
  function apply() {
    pending = false;
    root.style.setProperty('--mx', px.toFixed(0));
    root.style.setProperty('--my', py.toFixed(0));
    root.style.setProperty('--mxp', (px / window.innerWidth).toFixed(3));
  }
  document.addEventListener('pointermove', e => {
    px = e.clientX; py = e.clientY;
    if (!pending) { pending = true; requestAnimationFrame(apply); }
  }, { passive: true });
}
