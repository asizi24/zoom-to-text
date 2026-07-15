"""
Tests for the on-demand Map-Reduce Smart Summary.

Unit level: chunking, the map→reduce call pattern, and Obsidian document
assembly (frontmatter, tag extraction, code-fence unwrap, 3 flashcards).
Full stack: trigger on a completed task → poll → markdown → export, with the
LLM provider stubbed so nothing hits a live Ollama/Gemini.
"""
import asyncio
import time

import pytest

from app.config import settings
from app.models import LessonResult
from app.services import smart_summary
from app.services.smart_summary import (
    _build_obsidian_doc,
    chunk_by_words,
    generate_smart_summary,
)

REDUCE_BODY = (
    "tags: python, networking\n\n"
    "## 📝 סיכום\n"
    "זהו סיכום השיעור.\n\n"
    "## 🗂️ נושאים עיקריים\n"
    "### רשתות\n"
    "הסבר על `TCP`.\n\n"
    "```python\nprint('hello')\n```\n\n"
    "## 🎴 כרטיסיות לחזרה\n"
    "**שאלה:** מה זה TCP?\n**תשובה:** פרוטוקול אמין.\n\n"
    "**שאלה:** מה זה UDP?\n**תשובה:** פרוטוקול מהיר.\n\n"
    "**שאלה:** מה ההבדל?\n**תשובה:** אמינות מול מהירות.\n"
)


class FakeProvider:
    name = "ollama"

    def __init__(self):
        self.calls = []

    async def complete(self, prompt, *, system=None, temperature=0.3, timeout=None):
        self.calls.append(system)
        if system == smart_summary._REDUCE_SYSTEM:
            return REDUCE_BODY
        return f"MAP partial: {prompt[:16]}"


def _run(coro):
    try:
        return asyncio.run(coro)
    finally:
        asyncio.set_event_loop(asyncio.new_event_loop())


# ── Chunking ───────────────────────────────────────────────────────────────────

def test_chunk_by_words_splits_on_word_boundary():
    text = " ".join(str(i) for i in range(9000))
    chunks = chunk_by_words(text, 4000)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 4000
    assert len(chunks[2].split()) == 1000


def test_chunk_by_words_empty():
    assert chunk_by_words("") == []
    assert chunk_by_words("   ") == []


# ── Map-Reduce call pattern ──────────────────────────────────────────────────────

def test_single_chunk_runs_reduce_only():
    provider = FakeProvider()
    md = _run(generate_smart_summary("short source text", provider, title="שיעור"))
    assert provider.calls == [smart_summary._REDUCE_SYSTEM]
    assert md.startswith("---")


def test_multi_chunk_runs_map_then_reduce():
    provider = FakeProvider()
    text = " ".join(str(i) for i in range(10))  # 10 words
    _run(generate_smart_summary(text, provider, title="שיעור", words_per_chunk=4))
    maps = [c for c in provider.calls if c == smart_summary._MAP_SYSTEM]
    reduces = [c for c in provider.calls if c == smart_summary._REDUCE_SYSTEM]
    assert len(maps) == 3          # 10 words / 4 → 3 chunks
    assert len(reduces) == 1


# ── Obsidian assembly ────────────────────────────────────────────────────────────

def test_obsidian_document_structure():
    provider = FakeProvider()
    md = _run(generate_smart_summary(
        "x", provider, title="שיעור רשתות", source="upload:lec.mp4"
    ))
    assert md.startswith("---\n")
    assert 'title: "שיעור רשתות"' in md
    assert "date: " in md
    assert 'source: "upload:lec.mp4"' in md
    # base tags + tags extracted from the leading `tags:` line
    assert "  - שיעור" in md
    assert "  - zoom-to-text" in md
    assert "  - python" in md
    assert "  - networking" in md
    assert "generated_by: ollama" in md
    # the raw tags line was lifted out of the body
    assert "tags: python, networking" not in md
    # body preserved with code fence and exactly 3 flashcards
    assert "## 📝 סיכום" in md
    assert "```python" in md
    assert md.count("**שאלה:**") == 3


def test_build_obsidian_doc_unwraps_outer_fence():
    body = "```markdown\n## 📝 סיכום\nתוכן\n```"
    md = _build_obsidian_doc(body, title="T", source="", backend="gemini")
    assert "```markdown" not in md
    assert "## 📝 סיכום" in md
    assert "generated_by: gemini" in md


# ── Full-stack endpoint flow ─────────────────────────────────────────────────────

def _login(client, mock_email):
    client.post("/api/auth/request", json={"email": "allowed@example.com"})
    token = mock_email[0]["token"]
    resp = client.get(f"/api/auth/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 302
    client.cookies.set("session_id", resp.cookies["session_id"])


def _upload(client):
    resp = client.post(
        "/api/tasks/upload",
        files={"file": ("lecture.mp4", b"fake media bytes", "video/mp4")},
        data={"mode": "gemini_direct", "language": "he"},
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["task_id"]


def _wait_for_terminal(client, task_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.05)
    pytest.fail(f"task {task_id} never reached a terminal status")


def _wait_for_smart_summary(client, task_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    data = None
    while time.monotonic() < deadline:
        data = client.get(f"/api/tasks/{task_id}/smart-summary").json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    pytest.fail(f"smart summary never finished: {data}")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data)
    monkeypatch.setattr(settings, "downloads_dir", data / "downloads")
    (data / "downloads").mkdir(parents=True)


def test_smart_summary_end_to_end(client, mock_email, monkeypatch, isolated_dirs):
    import app.services.summarizer as summarizer

    async def fake_summarize_audio(audio_path, progress_cb):
        return LessonResult(summary="סיכום קצר של שיעור", chapters=[], quiz=[], language="he")

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", fake_summarize_audio)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    provider = FakeProvider()

    async def fake_factory():
        return provider

    monkeypatch.setattr(smart_summary, "get_summary_provider", fake_factory)

    _login(client, mock_email)
    task_id = _upload(client)
    _wait_for_terminal(client, task_id)

    # Trigger generation
    resp = client.post(f"/api/tasks/{task_id}/smart-summary")
    assert resp.status_code == 202
    assert resp.json()["status"] in ("pending", "running", "completed")

    # Poll to completion
    data = _wait_for_smart_summary(client, task_id)
    assert data["status"] == "completed", data
    assert data["markdown"].startswith("---")
    assert "## 📝 סיכום" in data["markdown"]
    assert data["markdown"].count("**שאלה:**") == 3

    # Export as .md
    exp = client.get(f"/api/tasks/{task_id}/smart-summary/export.md")
    assert exp.status_code == 200
    assert exp.headers["content-type"].startswith("text/markdown")
    assert "attachment" in exp.headers["content-disposition"]


def test_smart_summary_unknown_task_404(client, mock_email, isolated_dirs):
    _login(client, mock_email)
    assert client.post("/api/tasks/no-such/smart-summary").status_code == 404
    assert client.get("/api/tasks/no-such/smart-summary").status_code == 404


def test_smart_summary_idle_for_completed_task_before_trigger(client, mock_email, monkeypatch, isolated_dirs):
    import app.services.summarizer as summarizer

    async def fake_summarize_audio(audio_path, progress_cb):
        return LessonResult(summary="סיכום", chapters=[], quiz=[], language="he")

    async def fake_flashcards(summary, transcript=None):
        return []

    monkeypatch.setattr(summarizer, "summarize_audio", fake_summarize_audio)
    monkeypatch.setattr(summarizer, "generate_flashcards", fake_flashcards)

    _login(client, mock_email)
    task_id = _upload(client)
    _wait_for_terminal(client, task_id)

    data = client.get(f"/api/tasks/{task_id}/smart-summary").json()
    assert data["status"] == "idle"
    assert data["markdown"] is None
