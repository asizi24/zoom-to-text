"""ICS (iCalendar) exporter — B5.

Renders a completed TaskResponse as a single VEVENT inside a VCALENDAR
document. Anyone can `Add to Calendar` from the generated `.ics` without
needing an OAuth flow against Google/Outlook — the user just clicks the
file and their default calendar app handles it.

Format reference: RFC 5545 §3.4 (iCalendar object).

The event covers the time slot from the task's `created_at` to one hour
later (we don't carry the original recording duration here — most use
cases are post-hoc "add this lecture to my study calendar," and a 1-hour
default reads well in calendar UIs).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from app.models import TaskResponse

# ASCII control chars that break Outlook/Google Calendar ICS parsers.
# Keep \t (\x09) and \n (\x0a) — those are legal/handled by _escape.
_ICS_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f]")

# iCalendar requires CRLF line endings (§3.1).
_CRLF = "\r\n"

# Folding boundary per RFC 5545 §3.1 — long lines must be wrapped at 75
# bytes with a leading space on continuation lines. We're conservative
# and wrap at 73 characters to leave room for the leading space + CRLF.
_FOLD_AT = 73


def build_ics(task: TaskResponse, *, duration_minutes: int = 60) -> str:
    """Render the task as an `.ics` document.

    `duration_minutes` is configurable so a future caller can pass the
    actual recording length when available.
    """
    summary = _summary_line(task)
    description = _description(task)
    url = _task_url(task)
    dtstart, dtend = _start_end(task, duration_minutes)
    dtstamp = _to_ics(_now())
    uid = f"task-{task.task_id}@zoom-to-text"

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//zoom-to-text//ICS Export//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{_escape(summary)}",
    ]
    if description:
        lines.append(f"DESCRIPTION:{_escape(description)}")
    if url:
        lines.append(f"URL:{_escape(url)}")
    lines.extend(["END:VEVENT", "END:VCALENDAR"])

    folded = [_fold(line) for line in lines]
    return _CRLF.join(folded) + _CRLF


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _start_end(task: TaskResponse, duration_minutes: int) -> tuple[str, str]:
    try:
        start = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        start = _now()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = start + timedelta(minutes=max(duration_minutes, 1))
    return _to_ics(start), _to_ics(end)


def _to_ics(dt: datetime) -> str:
    """Format UTC datetime as `YYYYMMDDTHHMMSSZ` (RFC 5545 §3.3.5 form 2)."""
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _summary_line(task: TaskResponse) -> str:
    if task.result and task.result.chapters:
        return task.result.chapters[0].title[:200]
    if task.url:
        return f"הרצאה — {task.url[:120]}"
    return f"הרצאה {task.task_id[:8]}"


def _description(task: TaskResponse) -> str:
    if task.result is None or not task.result.summary:
        return ""
    text = task.result.summary.strip()
    if len(text) > 2000:
        text = text[:2000].rstrip() + "…"
    return text


def _task_url(task: TaskResponse) -> str:
    """Best-effort hyperlink. Falls back to the source URL when no local URL is set."""
    return task.url or ""


def _escape(text: str) -> str:
    """Escape text fields per RFC 5545 §3.3.11 (TEXT).

    Strips ASCII control chars first — Outlook/Google Calendar refuse
    to parse VEVENTs whose SUMMARY/DESCRIPTION contain NUL/BEL/etc.
    """
    text = _ICS_CONTROL_RE.sub("", text)
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _fold(line: str) -> str:
    """RFC 5545 §3.1 line folding."""
    if len(line) <= _FOLD_AT:
        return line
    chunks = [line[:_FOLD_AT]]
    rest = line[_FOLD_AT:]
    while rest:
        chunks.append(" " + rest[:_FOLD_AT - 1])
        rest = rest[_FOLD_AT - 1:]
    return _CRLF.join(chunks)
