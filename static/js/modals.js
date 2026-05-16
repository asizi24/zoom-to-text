// Modal accessibility + mobile drag-to-dismiss.
// The four .about-overlay modals (about, webhooks, insights, recipes) get:
//   - focus trap while open (Tab/Shift+Tab cycle)
//   - swipe-down to dismiss on touch devices ≤540px
// ESC + backdrop click + focus restoration are already handled by the
// _openModal/_closeModal helpers in index.html — we only add what's missing.

(function () {
  const FOCUSABLE = [
    'a[href]', 'area[href]',
    'button:not([disabled])', 'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])', 'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',');

  function focusables(modal) {
    return [...modal.querySelectorAll(FOCUSABLE)].filter(el => {
      // skip elements inside hidden subtrees
      if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return false;
      return !el.hasAttribute('inert');
    });
  }

  function openModal(modal) {
    if (!modal) return;
    const trap = e => {
      if (e.key !== 'Tab') return;
      const list = focusables(modal);
      if (!list.length) { e.preventDefault(); modal.focus(); return; }
      const first = list[0];
      const last = list[list.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !modal.contains(active))) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && (active === last || !modal.contains(active))) {
        e.preventDefault(); first.focus();
      }
    };
    modal._focusTrap = trap;
    modal.addEventListener('keydown', trap);
  }

  function closeModal(modal) {
    if (!modal || !modal._focusTrap) return;
    modal.removeEventListener('keydown', modal._focusTrap);
    modal._focusTrap = null;
  }

  // Watch aria-hidden changes — that's the canonical "is the modal open" flag
  // (the existing _openModal/_closeModal in index.html toggle it).
  function observe(modal) {
    const mo = new MutationObserver(() => {
      const hidden = modal.getAttribute('aria-hidden') !== 'false';
      if (hidden) closeModal(modal);
      else openModal(modal);
    });
    mo.observe(modal, { attributes: true, attributeFilter: ['aria-hidden'] });
    // Sync initial state
    if (modal.getAttribute('aria-hidden') === 'false') openModal(modal);
  }

  // ── Drag-to-dismiss on bottom-sheet (mobile only) ─────────────────────
  function isPhone() { return window.matchMedia('(max-width: 540px)').matches; }

  function attachDrag(modal) {
    const wrap = modal.querySelector('.profile-card-wrap');
    if (!wrap) return;

    let startY = null;
    let lastY = null;
    let dragging = false;

    function getCloseFn() {
      // The existing close helpers are global functions named close<Modal>Modal
      const map = {
        'about-overlay':   typeof closeAbout         === 'function' ? closeAbout         : null,
        'webhooks-modal':  typeof closeWebhooksModal === 'function' ? closeWebhooksModal : null,
        'insights-modal':  typeof closeInsightsModal === 'function' ? closeInsightsModal : null,
        'recipes-modal':   typeof closeRecipesModal  === 'function' ? closeRecipesModal  : null,
      };
      return map[modal.id] || null;
    }

    function onStart(e) {
      if (!isPhone()) return;
      const t = e.touches ? e.touches[0] : e;
      // Only start drag near the top of the sheet (where the handle lives)
      const card = modal.querySelector('.profile-card');
      if (!card) return;
      const rect = card.getBoundingClientRect();
      if (t.clientY - rect.top > 56) return; // require grip near top
      startY = lastY = t.clientY;
      dragging = true;
      wrap.classList.add('dragging');
    }

    function onMove(e) {
      if (!dragging) return;
      const t = e.touches ? e.touches[0] : e;
      lastY = t.clientY;
      const dy = Math.max(0, lastY - startY);
      wrap.style.transform = `translateY(${dy}px)`;
    }

    function onEnd() {
      if (!dragging) return;
      const dy = Math.max(0, (lastY ?? startY) - startY);
      wrap.classList.remove('dragging');
      wrap.style.transform = '';
      dragging = false;
      if (dy > 110) {
        const close = getCloseFn();
        if (close) close();
      }
    }

    modal.addEventListener('touchstart', onStart, { passive: true });
    modal.addEventListener('touchmove',  onMove,  { passive: true });
    modal.addEventListener('touchend',   onEnd);
    modal.addEventListener('touchcancel',onEnd);
  }

  function init() {
    document.querySelectorAll('[role="dialog"]').forEach(m => {
      observe(m);
      attachDrag(m);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
