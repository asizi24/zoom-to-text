// WAI-ARIA tablist keyboard navigation.
// Reads `direction` to flip Left/Right arrows in RTL.
// Calls the existing legacy `switchTab(key)` function for any side-effects.

function _initTablist(list) {
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  if (!tabs.length) return;
  const isRTL = getComputedStyle(list).direction === 'rtl';

  function activate(tab, opts = {}) {
    const key = tab.dataset.tabKey;
    if (key && typeof window.switchTab === 'function') {
      window.switchTab(key);
    } else {
      tabs.forEach(t => {
        const sel = t === tab;
        t.setAttribute('aria-selected', String(sel));
        t.setAttribute('tabindex', sel ? '0' : '-1');
        t.classList.toggle('active', sel);
        const panel = document.getElementById(t.getAttribute('aria-controls'));
        if (panel) panel.classList.toggle('active', sel);
      });
    }
    if (opts.focus !== false) tab.focus();
  }

  list.addEventListener('keydown', e => {
    const cur = tabs.indexOf(document.activeElement);
    if (cur < 0) return;
    const fwd = isRTL ? 'ArrowLeft' : 'ArrowRight';
    const bwd = isRTL ? 'ArrowRight' : 'ArrowLeft';
    let next = null;
    if (e.key === fwd)        next = (cur + 1) % tabs.length;
    else if (e.key === bwd)   next = (cur - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End')  next = tabs.length - 1;
    if (next !== null) {
      e.preventDefault();
      activate(tabs[next]);
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[role="tablist"]').forEach(_initTablist);
});
