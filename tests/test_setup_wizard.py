"""
Tests for the first-boot Setup Wizard: GET / interception, the unauthenticated
but self-disabling /api/setup/* endpoints, and the model-pull SSE stream.

Uses the `wizard_client` fixture (setup_complete starts false on a fresh DB).
The OllamaProvider is stubbed so nothing depends on a live Ollama server.
"""
import app.api.routers.setup as setup_api
import app.services.hardware as hardware


def _patch_strong_gpu(monkeypatch):
    monkeypatch.setattr(
        hardware, "_query_nvidia_smi",
        lambda: [("NVIDIA GeForce RTX 4070 Ti", 12282)],
    )


class _FakeOllama:
    """Configurable stand-in for OllamaProvider (set class attrs per test)."""
    available = False
    models: list = []
    pull_events: list = []

    def __init__(self, host, model, timeout=None):
        self.model = model

    async def is_available(self):
        return type(self).available

    async def list_models(self):
        return list(type(self).models)

    async def pull(self, model=None):
        for event in type(self).pull_events:
            yield event


# ── GET / interception ─────────────────────────────────────────────────────────

def test_root_serves_wizard_when_not_setup(wizard_client):
    resp = wizard_client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    assert "setup.js" in resp.text


def test_status_initially_incomplete(wizard_client):
    data = wizard_client.get("/api/setup/status").json()
    assert data["setup_complete"] is False


# ── Hardware probe endpoint ──────────────────────────────────────────────────────

def test_hardware_probe_reports_gpu(wizard_client, monkeypatch):
    _patch_strong_gpu(monkeypatch)
    _FakeOllama.available = False
    _FakeOllama.models = []
    monkeypatch.setattr(setup_api, "OllamaProvider", _FakeOllama)

    data = wizard_client.get("/api/setup/hardware").json()
    assert data["strong_gpu"] is True
    assert data["recommended_backend"] == "ollama"
    assert data["default_model"] == "gemma2:9b"
    assert data["ollama_available"] is False
    assert data["installed_models"] == []


# ── Completion + self-disable ────────────────────────────────────────────────────

def test_complete_ollama_then_endpoints_disable(wizard_client):
    resp = wizard_client.post(
        "/api/setup/complete", json={"backend": "ollama", "ollama_model": "gemma2:9b"}
    )
    assert resp.status_code == 200
    assert resp.json()["backend"] == "ollama"

    status = wizard_client.get("/api/setup/status").json()
    assert status["setup_complete"] is True
    assert status["summary_backend"] == "ollama"
    assert status["ollama_model"] == "gemma2:9b"

    # Setup endpoints now self-disable (403), and GET / stops serving the wizard.
    assert wizard_client.get("/api/setup/hardware").status_code == 403
    assert wizard_client.post(
        "/api/setup/complete", json={"backend": "gemini", "gemini_api_key": "x"}
    ).status_code == 403

    root = wizard_client.get("/", follow_redirects=False)
    assert root.status_code == 302
    assert root.headers["location"] == "/login"


def test_complete_gemini_success(wizard_client):
    resp = wizard_client.post(
        "/api/setup/complete", json={"backend": "gemini", "gemini_api_key": "AIzaTESTKEY"}
    )
    assert resp.status_code == 200
    status = wizard_client.get("/api/setup/status").json()
    assert status["setup_complete"] is True
    assert status["summary_backend"] == "gemini"


def test_complete_gemini_requires_key(wizard_client):
    assert wizard_client.post(
        "/api/setup/complete", json={"backend": "gemini"}
    ).status_code == 400
    assert wizard_client.post(
        "/api/setup/complete", json={"backend": "gemini", "gemini_api_key": "   "}
    ).status_code == 400


def test_complete_rejects_unknown_backend(wizard_client):
    assert wizard_client.post(
        "/api/setup/complete", json={"backend": "openai"}
    ).status_code == 400


# ── Model-pull SSE stream ────────────────────────────────────────────────────────

def test_pull_stream_emits_progress_and_done(wizard_client, monkeypatch):
    _FakeOllama.pull_events = [
        {"status": "pulling manifest"},
        {"status": "downloading", "total": 100, "completed": 40},
        {"status": "success"},
    ]
    monkeypatch.setattr(setup_api, "OllamaProvider", _FakeOllama)

    resp = wizard_client.get("/api/setup/ollama/pull/stream?model=gemma2:9b")
    assert resp.status_code == 200
    body = resp.text
    assert '"type": "progress"' in body
    assert '"percent": 40' in body
    assert '"type": "done"' in body
