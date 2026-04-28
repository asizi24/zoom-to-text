"""
Tests for the Obsidian-flavored Markdown exporter (UPGRADE_PROMPT.md Task 1.4).

The exporter is a pure function: TaskResponse → str. No I/O, no Gemini, no
filesystem. The endpoint test lives below in this file as well.
"""
from datetime import datetime, timezone

import pytest

from app.models import (
    ActionItem,
    Chapter,
    Decision,
    LessonResult,
    Objection,
    OpenQuestion,
    PerSpeakerSentiment,
    QuizQuestion,
    SentimentAnalysis,
    TaskResponse,
    TaskStatus,
    ToneShift,
)
from app.services.exporters.markdown import build_obsidian_markdown


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_task(result: LessonResult, *, url: str | None = "https://zoom.us/rec/abc",
               created_at: str = "2026-04-28T10:00:00+00:00",
               task_id: str = "11111111-2222-3333-4444-555555555555") -> TaskResponse:
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        progress=100,
        message="ok",
        created_at=created_at,
        url=url,
        result=result,
        has_audio=False,
    )


# ── Frontmatter ──────────────────────────────────────────────────────────────

def test_frontmatter_has_yaml_block_at_top():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="hi")))
    assert md.startswith("---\n")
    # Frontmatter ends with closing --- followed by a blank line then body
    head, _, _ = md.partition("\n---\n")
    assert "date: 2026-04-28" in head
    assert "source: " in head
    assert "tags:" in head


def test_frontmatter_includes_content_type_when_present():
    r = LessonResult(summary="hi", content_type="meeting")
    md = build_obsidian_markdown(_make_task(r))
    assert "content_type: meeting" in md


def test_frontmatter_omits_content_type_when_none():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="hi")))
    head = md.split("\n---\n", 1)[0]
    assert "content_type:" not in head


def test_frontmatter_participants_collected_from_speaker_map_and_owners():
    r = LessonResult(
        summary="x",
        speaker_map={"Speaker A": "אסף", "Speaker B": "דן"},
        action_items=[ActionItem(owner="רוני", task="t")],
    )
    md = build_obsidian_markdown(_make_task(r))
    # Order is alphabetical-ish but we only care that all three are present
    head = md.split("\n---\n", 1)[0]
    assert "participants:" in head
    assert "אסף" in head
    assert "דן" in head
    assert "רוני" in head


def test_frontmatter_handles_upload_source():
    r = LessonResult(summary="x")
    md = build_obsidian_markdown(_make_task(r, url="upload:meeting.mp3"))
    head = md.split("\n---\n", 1)[0]
    assert "source: upload" in head


def test_frontmatter_tags_include_zoom_to_text_baseline():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="x", content_type="lecture")))
    head = md.split("\n---\n", 1)[0]
    # Baseline tag plus content-type tag
    assert "zoom-to-text" in head
    assert "lecture" in head


# ── Title + backlinks ────────────────────────────────────────────────────────

def test_title_includes_date():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="x")))
    assert "# סיכום שיעור — 2026-04-28" in md


def test_backlink_uses_first_chapter_title_when_available():
    r = LessonResult(
        summary="x",
        chapters=[Chapter(title="תכנון רבעון Q3", content="...")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "[[Meeting: 2026-04-28 - תכנון רבעון Q3]]" in md


def test_backlink_omitted_when_no_chapters():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="x")))
    assert "[[Meeting:" not in md


# ── Action items as checkboxes ───────────────────────────────────────────────

def test_action_items_render_as_checkboxes():
    r = LessonResult(
        summary="x",
        action_items=[
            ActionItem(owner="Asaf", task="לשלוח את הדק", deadline="EOW", priority="high"),
            ActionItem(owner="דן", task="לבדוק את ה-API"),
        ],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## ✅ Action Items" in md
    assert "- [ ] לשלוח את הדק — Asaf" in md
    assert "- [ ] לבדוק את ה-API — דן" in md


def test_action_items_include_owner_tag():
    r = LessonResult(
        summary="x",
        action_items=[ActionItem(owner="אסף", task="ship it")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "#action/אסף" in md


def test_action_items_owner_tag_handles_whitespace():
    r = LessonResult(
        summary="x",
        action_items=[ActionItem(owner="John Doe", task="t")],
    )
    md = build_obsidian_markdown(_make_task(r))
    # Tags can't contain spaces in Obsidian — collapse to hyphen
    assert "#action/John-Doe" in md


def test_action_items_section_omitted_when_empty():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="x")))
    assert "## ✅ Action Items" not in md


def test_action_items_render_deadline_and_priority_when_present():
    r = LessonResult(
        summary="x",
        action_items=[ActionItem(owner="A", task="t", deadline="2026-05-01", priority="high")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "📅 2026-05-01" in md
    assert "🔥 high" in md


# ── Decisions, open questions, sentiment, objections ─────────────────────────

def test_decisions_section_lists_each_with_stakeholders():
    r = LessonResult(
        summary="x",
        decisions=[Decision(decision="לעבור ל-Postgres", stakeholders=["אסף", "דן"], context="עומס גדל")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## 🎯 החלטות" in md
    assert "לעבור ל-Postgres" in md
    assert "אסף" in md and "דן" in md


def test_open_questions_section():
    r = LessonResult(
        summary="x",
        open_questions=[OpenQuestion(question="מה לוח הזמנים?", raised_by="דן")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## ❓ שאלות פתוחות" in md
    assert "מה לוח הזמנים?" in md


def test_sentiment_section_includes_overall_and_speakers():
    r = LessonResult(
        summary="x",
        sentiment_analysis=SentimentAnalysis(
            overall_tone="positive",
            per_speaker_sentiment=[PerSpeakerSentiment(speaker="אסף", sentiment="positive")],
            shifts_in_tone=[ToneShift(at="00:12", **{"from": "neutral", "to": "tense"}, trigger="budget")],
        ),
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## 💬 ניתוח רגשי" in md
    assert "positive" in md
    assert "00:12" in md


def test_objections_section():
    r = LessonResult(
        summary="x",
        objections_tracked=[Objection(objection="המחיר גבוה", raised_by="דן", resolved=False)],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## 🛡️ התנגדויות" in md
    assert "המחיר גבוה" in md


# ── Chapters ─────────────────────────────────────────────────────────────────

def test_chapters_render_as_h3_with_key_points():
    r = LessonResult(
        summary="x",
        chapters=[Chapter(title="פרק ראשון", content="תוכן", key_points=["נקודה א", "נקודה ב"])],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## 📚 פרקים ונושאים" in md
    assert "### 1. פרק ראשון" in md
    assert "- נקודה א" in md


# ── Exam in collapsible <details> ────────────────────────────────────────────

def test_quiz_renders_inside_details_block():
    r = LessonResult(
        summary="x",
        quiz=[QuizQuestion(question="Q?", options=["א. 1", "ב. 2"], correct_answer="א. 1", explanation="כי")],
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "<details>" in md
    assert "<summary>🧠 מבחן" in md
    assert "</details>" in md
    # Question content must be inside the details block
    details_block = md.split("<details>", 1)[1].split("</details>", 1)[0]
    assert "Q?" in details_block
    assert "א. 1" in details_block


def test_quiz_section_omitted_when_empty():
    md = build_obsidian_markdown(_make_task(LessonResult(summary="x")))
    assert "<details>" not in md


# ── Transcript: prefer diarized when present ─────────────────────────────────

def test_diarized_transcript_used_when_present():
    r = LessonResult(
        summary="x",
        transcript="raw text",
        diarized_transcript="Speaker A: שלום\nSpeaker B: היי",
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "## 🗣️ תמלול עם דוברים" in md
    assert "Speaker A: שלום" in md
    assert "raw text" not in md  # plain transcript NOT included when diarized exists


def test_plain_transcript_used_when_no_diarization():
    r = LessonResult(summary="x", transcript="raw text only")
    md = build_obsidian_markdown(_make_task(r))
    assert "## 📄 תמלול" in md
    assert "raw text only" in md


# ── Backward compat ──────────────────────────────────────────────────────────

def test_minimal_old_result_renders_without_errors():
    """A LessonResult from before Task 1.1 (no action_items, etc.) must still export."""
    r = LessonResult(
        summary="ישן וטוב",
        chapters=[Chapter(title="פרק", content="תוכן")],
        quiz=[QuizQuestion(question="?", options=["א. 1"], correct_answer="א. 1")],
        transcript="טקסט",
    )
    md = build_obsidian_markdown(_make_task(r))
    assert "ישן וטוב" in md
    assert "## ✅ Action Items" not in md
    assert "## 🎯 החלטות" not in md
    # Frontmatter still valid
    assert md.startswith("---\n")


# ── Endpoint integration ─────────────────────────────────────────────────────

def test_endpoint_returns_markdown_attachment(client, monkeypatch):
    """GET /api/tasks/{id}/export/obsidian returns markdown with download headers."""
    import asyncio
    from app import state
    from app.api import deps

    # Bypass auth
    async def _user():
        return "test-user"

    from app.main import app
    app.dependency_overrides[deps.get_current_user] = _user

    try:
        async def _seed():
            await state.create_task("task-abc", "https://zoom.us/rec/x", user_id="test-user")
            r = LessonResult(
                summary="סיכום קצר",
                action_items=[ActionItem(owner="Asaf", task="ship")],
            )
            await state.complete_task("task-abc", r)

        asyncio.get_event_loop().run_until_complete(_seed())

        resp = client.get("/api/tasks/task-abc/export/obsidian")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert "obsidian" in resp.headers.get("content-disposition", "").lower()
        body = resp.content.decode("utf-8")
        assert body.startswith("---\n")
        assert "סיכום קצר" in body
        assert "- [ ] ship — Asaf" in body
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)


def test_endpoint_404_on_missing_task(client):
    from app.api import deps
    from app.main import app

    async def _user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _user
    try:
        resp = client.get("/api/tasks/does-not-exist/export/obsidian")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)


def test_endpoint_400_when_no_result(client):
    """A task that hasn't completed yet has no result — must 400, not 500."""
    import asyncio
    from app import state
    from app.api import deps
    from app.main import app

    async def _user():
        return "test-user"

    app.dependency_overrides[deps.get_current_user] = _user
    try:
        async def _seed():
            await state.create_task("pending-task", "https://zoom.us/rec/y", user_id="test-user")

        asyncio.get_event_loop().run_until_complete(_seed())
        resp = client.get("/api/tasks/pending-task/export/obsidian")
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)
