"""
Obsidian-flavored Markdown exporter (UPGRADE_PROMPT.md Task 1.4).

Renders a completed TaskResponse to a self-contained Markdown document
optimized for an Obsidian vault: YAML frontmatter, action items as
checkboxes with owner-tagged backlinks, decisions / open questions /
sentiment / objections sections, chapters as H3, and the exam in a
collapsible <details> block.

This is a pure function — no I/O, no Gemini calls — so it can render
historical tasks that pre-date Task 1.1 fields just as cleanly.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from app.models import LessonResult, TaskResponse


# ── Public API ──────────────────────────────────────────────────────────────

def build_obsidian_markdown(task: TaskResponse) -> str:
    """Render the full Obsidian-flavored Markdown for a completed task."""
    if task.result is None:
        # Defensive — the API layer is expected to 400 before reaching here
        return _empty_skeleton(task)

    result = task.result
    date = _date_from_created_at(task.created_at)
    source = _source_label(task.url)
    participants = _collect_participants(result)

    parts: list[str] = []
    parts.append(_render_frontmatter(date=date, source=source, content_type=result.content_type,
                                     language=result.language, participants=participants))
    parts.append("")
    parts.append(f"# סיכום שיעור — {date}")
    parts.append(f"> מקור: {task.url or 'unknown'}")

    # Backlink — uses the first chapter title as the meeting topic when available
    if result.chapters:
        topic = result.chapters[0].title.strip()
        parts.append(f"[[Meeting: {date} - {topic}]]")
    parts.append("")

    if result.summary:
        parts.append("## 📝 סיכום")
        parts.append(result.summary.strip())
        parts.append("")

    if result.action_items:
        parts.append("## ✅ Action Items")
        for ai in result.action_items:
            parts.append(_render_action_item(ai))
        parts.append("")

    if result.decisions:
        parts.append("## 🎯 החלטות")
        for d in result.decisions:
            parts.append(_render_decision(d))
        parts.append("")

    if result.open_questions:
        parts.append("## ❓ שאלות פתוחות")
        for q in result.open_questions:
            line = f"- **{q.question.strip()}**"
            if q.raised_by:
                line += f" — הועלה על ידי {q.raised_by}"
            parts.append(line)
            if q.context:
                parts.append(f"  - הקשר: {q.context}")
        parts.append("")

    if result.sentiment_analysis:
        parts.append("## 💬 ניתוח רגשי")
        parts.append(_render_sentiment(result.sentiment_analysis))
        parts.append("")

    if result.objections_tracked:
        parts.append("## 🛡️ התנגדויות")
        for o in result.objections_tracked:
            parts.append(_render_objection(o))
        parts.append("")

    if result.chapters:
        parts.append("## 📚 פרקים ונושאים")
        for i, ch in enumerate(result.chapters, start=1):
            parts.append(f"### {i}. {ch.title}")
            if ch.content:
                parts.append(ch.content.strip())
            for kp in ch.key_points or []:
                parts.append(f"- {kp}")
            parts.append("")

    if result.quiz:
        parts.append("<details>")
        parts.append(f"<summary>🧠 מבחן — {len(result.quiz)} שאלות</summary>")
        parts.append("")
        for i, q in enumerate(result.quiz, start=1):
            parts.append(f"**שאלה {i}:** {q.question}")
            for opt in q.options or []:
                parts.append(f"- {opt}")
            parts.append(f"✅ תשובה נכונה: {q.correct_answer}")
            if q.explanation:
                parts.append(f"💡 {q.explanation}")
            parts.append("")
        parts.append("</details>")
        parts.append("")

    transcript_text = result.diarized_transcript or result.transcript
    if transcript_text:
        if result.diarized_transcript:
            parts.append("## 🗣️ תמלול עם דוברים")
        else:
            parts.append("## 📄 תמלול")
        parts.append(transcript_text.strip())
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ── Frontmatter ─────────────────────────────────────────────────────────────

def _render_frontmatter(*, date: str, source: str, content_type: str | None,
                        language: str, participants: list[str]) -> str:
    lines: list[str] = ["---", f"date: {date}", f"source: {source}"]
    if content_type:
        lines.append(f"content_type: {content_type}")
    lines.append(f"language: {language}")
    if participants:
        lines.append("participants:")
        for p in participants:
            lines.append(f"  - {p}")
    tags = _build_tags(content_type)
    lines.append("tags:")
    for t in tags:
        lines.append(f"  - {t}")
    lines.append("---")
    return "\n".join(lines)


def _build_tags(content_type: str | None) -> list[str]:
    tags = ["zoom-to-text"]
    if content_type:
        tags.append(content_type)
    return tags


def _date_from_created_at(created_at: str) -> str:
    """Extract YYYY-MM-DD; fall back to today on parse error."""
    try:
        # SQLite stores ISO 8601; strip a trailing Z if present
        s = created_at.replace("Z", "+00:00") if created_at.endswith("Z") else created_at
        return datetime.fromisoformat(s).date().isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()


def _source_label(url: str | None) -> str:
    """Compress the source URL to a short, frontmatter-friendly label."""
    if not url:
        return "unknown"
    if url.startswith("upload:"):
        return "upload"
    if "zoom.us" in url:
        return "zoom"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "url"


def _collect_participants(result: LessonResult) -> list[str]:
    """Union of speaker_map values, action_item owners, and decision stakeholders."""
    seen: list[str] = []
    def _add(name: str | None):
        if name:
            n = name.strip()
            if n and n not in seen:
                seen.append(n)

    if result.speaker_map:
        for v in result.speaker_map.values():
            _add(v)
    for ai in result.action_items:
        _add(ai.owner)
    for d in result.decisions:
        for st in d.stakeholders:
            _add(st)
    return seen


# ── Section renderers ───────────────────────────────────────────────────────

def _owner_tag(owner: str) -> str:
    """Whitespace → hyphens; Obsidian tags can't contain spaces."""
    cleaned = re.sub(r"\s+", "-", owner.strip())
    return f"#action/{cleaned}"


def _render_action_item(ai) -> str:
    parts = [f"- [ ] {ai.task.strip()} — {ai.owner.strip()}"]
    if ai.deadline:
        parts.append(f"📅 {ai.deadline}")
    if ai.priority:
        emoji = {"high": "🔥", "medium": "⚡", "low": "🌱"}.get(ai.priority.lower(), "•")
        parts.append(f"{emoji} {ai.priority}")
    parts.append(_owner_tag(ai.owner))
    line = " ".join(parts)
    if ai.source_quote:
        line += f"\n  - > {ai.source_quote.strip()}"
    return line


def _render_decision(d) -> str:
    line = f"- **{d.decision.strip()}**"
    if d.stakeholders:
        line += f" — {', '.join(d.stakeholders)}"
    extras = []
    if d.context:
        extras.append(f"  - הקשר: {d.context}")
    if d.source_quote:
        extras.append(f"  - > {d.source_quote.strip()}")
    if extras:
        line += "\n" + "\n".join(extras)
    return line


def _render_sentiment(s) -> str:
    lines = [f"- **טון כללי:** {s.overall_tone}"]
    if s.per_speaker_sentiment:
        lines.append("- **לפי דובר:**")
        for ps in s.per_speaker_sentiment:
            entry = f"  - {ps.speaker}: {ps.sentiment}"
            if ps.rationale:
                entry += f" — {ps.rationale}"
            lines.append(entry)
    if s.shifts_in_tone:
        lines.append("- **שינויי טון:**")
        for sh in s.shifts_in_tone:
            entry = f"  - {sh.at}: {sh.from_tone} → {sh.to_tone}"
            if sh.trigger:
                entry += f" (טריגר: {sh.trigger})"
            lines.append(entry)
    return "\n".join(lines)


def _render_objection(o) -> str:
    line = f"- **{o.objection.strip()}**"
    if o.raised_by:
        line += f" — {o.raised_by}"
    extras = []
    if o.response_given:
        extras.append(f"  - מענה: {o.response_given}")
    if o.resolved is not None:
        extras.append(f"  - נפתר: {'כן' if o.resolved else 'לא'}")
    if extras:
        line += "\n" + "\n".join(extras)
    return line


# ── Fallbacks ───────────────────────────────────────────────────────────────

def _empty_skeleton(task: TaskResponse) -> str:
    date = _date_from_created_at(task.created_at)
    return (
        f"---\ndate: {date}\nsource: {_source_label(task.url)}\n"
        f"language: he\ntags:\n  - zoom-to-text\n---\n\n"
        f"# סיכום שיעור — {date}\n\n_(אין תוצאה זמינה)_\n"
    )
