/**
 * Zoom Transcriber service worker.
 *
 * Goal: make the app installable as a PWA + speed up cold reloads by caching
 * static assets. We deliberately do NOT serve a stale shell while offline —
 * the app's value (transcription + chat) requires a live backend, so a stale
 * UI that can't talk to the server is worse than an explicit "you're offline"
 * page in the browser.
 *
 * Caching policy:
 *   • Static assets under /static/* and /manifest.webmanifest → cache-first
 *   • Everything else (the SPA root, /login, /api/*, /ws/*, /share/*, /clips/*)
 *     → bypass the SW entirely, default network behavior.
 */
const CACHE_NAME = 'z2t-shell-v1';
const SHELL = [
  '/static/style.css',
  '/static/js/sse.js',
  '/static/js/export-utils.js',
  '/static/icon.svg',
  '/manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      Promise.allSettled(SHELL.map((url) => cache.add(url)))
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Bypass for the SPA root, login page, and every dynamic backend path.
  if (
    url.pathname === '/' ||
    url.pathname === '/login' ||
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/ws/') ||
    url.pathname.startsWith('/auth/') ||
    url.pathname.startsWith('/share/') ||
    url.pathname.startsWith('/clips/')
  ) {
    return;
  }
  // Cache-first for static assets + manifest.
  if (
    url.pathname.startsWith('/static/') ||
    url.pathname === '/manifest.webmanifest'
  ) {
    event.respondWith(
      caches.match(req).then((cached) => {
        if (cached) return cached;
        return fetch(req).then((resp) => {
          if (resp.ok) {
            const respClone = resp.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, respClone));
          }
          return resp;
        });
      })
    );
  }
});
