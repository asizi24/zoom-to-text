"""Weekly email digest of the user's recent recordings (Batch B2).

A digest is built and sent once per week to users who opted in via
PUT /api/auth/me/preferences. The content is purely derived from data
already in SQLite — no extra LLM calls. We send via the existing
Resend.com integration used by magic-link emails.

Scheduling lives in `app.main._digest_scheduler` which sleeps one hour at
a time and decides per-user whether their 7-day window has elapsed.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone

import httpx

from app import state
from app.config import settings
from app.models import TaskResponse

logger = logging.getLogger(__name__)

DIGEST_PERIOD = timedelta(days=7)


def _human_date(iso: str | None) -> str:
    """Return YYYY-MM-DD from an ISO timestamp, defensively."""
    if not iso:
        return ""
    return iso[:10]


def build_digest_html(email: str, tasks: list[TaskResponse], base_url: str) -> str:
    """Render the digest HTML body. Pure function — no I/O."""
    safe_email = html.escape(email)
    items_html = []
    for t in tasks:
        title = ""
        if t.result and t.result.summary:
            # First line of summary, capped — gives the user a tactile preview
            title = t.result.summary.strip().split("\n", 1)[0][:140]
        if not title:
            title = t.url or t.task_id
        date = _human_date(t.created_at)
        link = f"{base_url}/?task={t.task_id}"
        items_html.append(
            "<li style='margin:8px 0;line-height:1.55'>"
            f"<a href='{html.escape(link)}' style='color:#6366f1;text-decoration:none;font-weight:600'>"
            f"{html.escape(title)}"
            "</a>"
            f"<div style='font-size:.85em;color:#64748b'>{html.escape(date)}</div>"
            "</li>"
        )

    body = (
        "<div dir='rtl' style='font-family:sans-serif;max-width:560px;margin:auto;color:#0f172a'>"
        "<h2 style='color:#4f46e5'>📚 הסיכום השבועי שלך</h2>"
        f"<p>שלום {safe_email},</p>"
        f"<p>השבוע סיכמת <strong>{len(tasks)}</strong> הקלטות. הנה הרשימה:</p>"
        "<ul style='padding-inline-start:18px;list-style:none'>"
        + "".join(items_html)
        + "</ul>"
        "<p style='margin-top:24px;font-size:.85em;color:#64748b'>"
        f"להסרה מרשימת התפוצה היכנס ל-<a href='{html.escape(base_url)}' style='color:#6366f1'>"
        "Zoom to Text</a> → היסטוריה → הגדרות.</p>"
        "</div>"
    )
    return body


async def send_digest_email(email: str, html_body: str) -> None:
    """Send the digest via Resend. Raises httpx.HTTPStatusError on failure."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": "Zoom to Text <onboarding@resend.dev>",
                "to": [email],
                "subject": "📚 Zoom to Text — הסיכום השבועי שלך",
                "html": html_body,
            },
            timeout=15.0,
        )
        resp.raise_for_status()


async def _user_due_for_digest(last_iso: str | None, *, now: datetime | None = None) -> bool:
    """True iff this user has not been emailed in the last DIGEST_PERIOD."""
    now = now or datetime.now(timezone.utc)
    if not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last) >= DIGEST_PERIOD


async def run_digest_cycle() -> int:
    """Iterate opt-in users; send a digest if their period has elapsed and they
    have new recordings. Returns the number of digests actually dispatched.

    Designed to be called from a periodic scheduler; safe to call repeatedly —
    `last_digest_at` gates duplicate sends.
    """
    if not settings.resend_api_key:
        logger.debug("Digest cycle skipped — RESEND_API_KEY not configured")
        return 0

    subs = await state.list_digest_subscribers()
    if not subs:
        return 0

    now = datetime.now(timezone.utc)
    sent = 0
    for sub in subs:
        if not await _user_due_for_digest(sub.get("last_digest_at"), now=now):
            continue
        since_iso = (now - DIGEST_PERIOD).isoformat()
        tasks = await state.list_tasks_completed_since(sub["id"], since_iso)
        if not tasks:
            continue  # nothing new this week — skip without marking sent
        try:
            body = build_digest_html(sub["email"], tasks, settings.base_url)
            await send_digest_email(sub["email"], body)
            await state.mark_digest_sent(sub["id"])
            sent += 1
            logger.info(
                "Weekly digest sent to %s (%d recordings)", sub["email"], len(tasks)
            )
        except Exception as exc:
            logger.warning("Weekly digest to %s failed: %s", sub["email"], exc)
    return sent
