"""
Tests for GET /api/tasks/{task_id}/export/pdf (Task 5 — PDF export).

WeasyPrint requires system libs (Pango, Cairo) that are not present on all
dev/CI machines. Every test patches weasyprint into sys.modules so the suite
runs anywhere without those system dependencies.
"""
import asyncio

import pytest


# ── Constants ─────────────────────────────────────────────────────────────────

FAKE_PDF = b"%PDF-1.4 fake-pdf-content-for-testing"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_completed(task_id: str, user_id: str = "test-user") -> None:
    from app import state
    from app.models import LessonResult, Chapter

    async def _run():
        await state.create_task(task_id, "https://zoom.us/rec/test", user_id=user_id)
        result = LessonResult(
            summary="סיכום קצר לבדיקת ייצוא PDF",
            chapters=[Chapter(title="נושא ראשון", content="תוכן הנושא")],
        )
        await state.complete_task(task_id, result)

    asyncio.get_event_loop().run_until_complete(_run())


def _seed_pending(task_id: str, user_id: str = "test-user") -> None:
    from app import state

    async def _run():
        await state.create_task(task_id, "https://zoom.us/rec/pending", user_id=user_id)

    asyncio.get_event_loop().run_until_complete(_run())


# ── Module-level fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _auth_bypass(client):
    """Bypass auth for all tests in this module."""
    from app.api import deps
    from app.main import app

    async def _user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _user
    yield
    app.dependency_overrides.pop(deps.get_current_user, None)


@pytest.fixture(autouse=True)
def _mock_build_pdf(monkeypatch):
    """
    Patch build_pdf directly on the exporter module.

    This avoids any dependency on weasyprint or its system libs (Pango/Cairo)
    so tests run on any machine. The route resolves build_pdf via
    `from app.services.exporters.pdf import build_pdf`, which reads the
    attribute from the already-imported module object — monkeypatching the
    attribute intercepts that lookup correctly.
    """
    import app.services.exporters.pdf as pdf_mod  # ensure module is loaded

    monkeypatch.setattr(pdf_mod, "build_pdf", lambda _task: FAKE_PDF)


# ── Happy-path tests ──────────────────────────────────────────────────────────

def test_pdf_export_returns_200(client):
    """Completed task → HTTP 200."""
    _seed_completed("pdf-ok-1")
    assert client.get("/api/tasks/pdf-ok-1/export/pdf").status_code == 200


def test_pdf_export_content_type_is_application_pdf(client):
    """Content-Type header must be application/pdf."""
    _seed_completed("pdf-ok-2")
    resp = client.get("/api/tasks/pdf-ok-2/export/pdf")
    assert resp.status_code == 200
    assert "application/pdf" in resp.headers["content-type"]


def test_pdf_export_has_attachment_content_disposition(client):
    """Content-Disposition must signal an attachment download with a .pdf name."""
    _seed_completed("pdf-ok-3")
    resp = client.get("/api/tasks/pdf-ok-3/export/pdf")
    assert resp.status_code == 200
    cd = resp.headers.get("content-disposition", "")
    assert "attachment" in cd
    assert ".pdf" in cd


def test_pdf_export_body_equals_weasyprint_output(client):
    """Response body must be exactly what weasyprint.HTML(...).write_pdf() returned."""
    _seed_completed("pdf-ok-4")
    resp = client.get("/api/tasks/pdf-ok-4/export/pdf")
    assert resp.status_code == 200
    assert resp.content == FAKE_PDF


# ── Error-path tests ──────────────────────────────────────────────────────────

def test_pdf_export_404_on_missing_task(client):
    """Task not in DB → 404."""
    resp = client.get("/api/tasks/no-such-task-pdf/export/pdf")
    assert resp.status_code == 404


def test_pdf_export_400_on_pending_task(client):
    """Task exists but has no result yet (still processing) → 400."""
    _seed_pending("pdf-pending-1")
    resp = client.get("/api/tasks/pdf-pending-1/export/pdf")
    assert resp.status_code == 400
