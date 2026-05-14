"""Tests for the B6.1 PWA scaffolding.

Covers:
  • /manifest.webmanifest is served with the correct MIME type and parses as
    JSON with the fields a browser needs for Add-to-Home-Screen
  • /service-worker.js is served at the root with the
    `Service-Worker-Allowed: /` header so its registration scope can be `/`
  • The SPA HTML actually references the manifest + registers the SW so a
    template-edit regression would fail this test
  • /static/icon.svg is reachable (Chromium requires at least one icon)
"""
import json


def test_manifest_endpoint_returns_json(client):
    resp = client.get("/manifest.webmanifest")
    assert resp.status_code == 200
    assert "application/manifest+json" in resp.headers["content-type"]
    payload = json.loads(resp.content)
    # The keys Chromium/Edge check before showing the install prompt:
    assert payload["name"]
    assert payload["short_name"]
    assert payload["start_url"] == "/"
    assert payload["scope"] == "/"
    assert payload["display"] in {"standalone", "fullscreen", "minimal-ui"}
    assert payload["icons"], "at least one icon is required for installability"
    assert payload["icons"][0]["src"].startswith("/static/")


def test_service_worker_served_at_root_with_allowed_header(client):
    resp = client.get("/service-worker.js")
    assert resp.status_code == 200
    # MIME must be a JavaScript type or the browser will refuse to register.
    assert "javascript" in resp.headers["content-type"]
    # We need this header so the SW can claim scope = "/" even though some
    # deployments may rewrite paths.
    assert resp.headers.get("service-worker-allowed") == "/"
    body = resp.text
    # Sanity: the file actually contains a service worker, not an HTML fallback.
    assert "addEventListener" in body
    assert "fetch" in body


def test_index_html_wires_manifest_and_registers_sw(client):
    resp = client.get("/static/index.html")
    assert resp.status_code == 200
    html = resp.text
    assert 'rel="manifest"' in html
    assert "/manifest.webmanifest" in html
    assert "serviceWorker.register" in html.replace(" ", "") or \
        "serviceWorker" in html and "register" in html


def test_pwa_icon_is_reachable(client):
    resp = client.get("/static/icon.svg")
    assert resp.status_code == 200
    assert "svg" in resp.headers["content-type"]
    assert "<svg" in resp.text
