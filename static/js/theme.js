// Light/dark theme controller.
// Pre-paint resolver lives inline in <head> to avoid FOUC; this module
// owns the user-facing toggle and persistence.

(function () {
  const STORAGE_KEY = 'z2t-theme';
  const root = document.documentElement;

  function currentTheme() {
    const explicit = root.dataset.theme;
    if (explicit === 'light' || explicit === 'dark') return explicit;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }

  function applyTheme(next) {
    root.dataset.theme = next;
    try { localStorage.setItem(STORAGE_KEY, next); } catch (e) { /* ignore */ }
    syncToggle(next);
  }

  function syncToggle(theme) {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const icon = document.getElementById('theme-toggle-icon');
    const isLight = theme === 'light';
    btn.setAttribute('aria-pressed', String(isLight));
    btn.setAttribute('title', isLight ? 'עבור למצב כהה' : 'עבור למצב בהיר');
    btn.setAttribute('aria-label', isLight ? 'עבור למצב כהה' : 'עבור למצב בהיר');
    if (icon) icon.textContent = isLight ? '☀️' : '🌙';
  }

  function init() {
    syncToggle(currentTheme());
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const next = currentTheme() === 'light' ? 'dark' : 'light';
      applyTheme(next);
    });

    // React to OS theme changes only when user has not made an explicit choice.
    const mql = window.matchMedia('(prefers-color-scheme: light)');
    const onOsChange = () => {
      let saved = null;
      try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) { /* ignore */ }
      if (saved !== 'light' && saved !== 'dark') {
        syncToggle(mql.matches ? 'light' : 'dark');
      }
    };
    if (mql.addEventListener) mql.addEventListener('change', onOsChange);
    else if (mql.addListener) mql.addListener(onOsChange);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
