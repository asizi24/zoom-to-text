"""Outgoing webhook notifier (B5).

Fires Slack- or Discord-compatible JSON POSTs when a task completes. The
processor calls `notify_task_completed(user_id, task)` as a fire-and-forget
asyncio task — failures are logged but never propagated, so a flaky
webhook URL can't fail the lecture itself.

Design choices:
  * HTTP via httpx with a 5 s total timeout. The pipeline must not block
    on webhook latency.
  * URL must be https — we never send lesson content over plain http.
  * Payloads are minimal: a short preview of the summary + a link back
    to the share page if the task has one (otherwise the local UI URL).
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional
from urllib.parse import urlparse

import httpx

from app import state
from app.config import settings
from app.models import TaskResponse

logger = logging.getLogger(__name__)

# Total budget per webhook. Three of these still cap at ~15 s — safe to
# `await asyncio.gather` without holding the pipeline.
_HTTP_TIMEOUT = 5.0
# Cap on summary preview included in the webhook body so we don't dump a
# 50 KB lesson into Slack.
_PREVIEW_CHARS = 600

# Well-known managed webhook providers — trusted regardless of resolved IP.
# DNS spoofing against these public hostnames is not a realistic threat
# model for this app; allowlisting avoids a getaddrinfo round-trip and
# prevents false positives if a CDN frontend resolves to an unusual range.
_WEBHOOK_ALLOWLIST_HOSTS = frozenset({
    "hooks.slack.com",
    "discord.com",
    "discordapp.com",
})

# Block common internal service ports outright even if the destination
# resolves to a public IP. These are never legitimate webhook targets and
# blocking them defends against DNS-rebinding-style attacks that map a
# public name onto an internal port.
_BLOCKED_PORTS = frozenset({22, 25, 3306, 5432, 6379})


def _is_safe_webhook_url(url: str) -> bool:
    """Reject URLs that could be used for SSRF against internal services.

    Returns True only when the URL:
      * parses cleanly with an http/https scheme,
      * has a hostname that is either on the trusted allowlist OR resolves
        exclusively to public IPs,
      * does not target a blocked internal-service port.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False

    # Reject well-known internal-service ports regardless of host.
    port = parsed.port
    if port is not None and port in _BLOCKED_PORTS:
        return False

    # Allowlist trusted managed providers — skip DNS resolution.
    if host.lower() in _WEBHOOK_ALLOWLIST_HOSTS:
        return True

    # Resolve and confirm every returned address is publicly routable.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _summary_preview(task: TaskResponse) -> str:
    if task.result is None or not task.result.summary:
        return "(אין סיכום עדיין)"
    text = task.result.summary.strip()
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS].rstrip() + "…"


def _task_title(task: TaskResponse) -> str:
    if task.result and task.result.chapters:
        return task.result.chapters[0].title[:120]
    if task.url:
        return f"הרצאה מ-{task.url[:80]}"
    return f"הרצאה {task.task_id[:8]}"


def _task_link(task: TaskResponse) -> str:
    """Return a public share link if available, otherwise the local task URL."""
    base = settings.base_url.rstrip("/")
    # Note: share_token isn't on TaskResponse by design (it's a private cookie-
    # auth value). Use the local UI URL as a fallback. Users on the same
    # network/auth can click through.
    return f"{base}/#task-{task.task_id}"


def _slack_payload(task: TaskResponse) -> dict:
    """Slack incoming-webhook payload (also accepted by Discord with ?wait=true).

    Slack's "text" + "attachments" schema is the most portable — Discord's
    incoming webhook ignores attachments[].color but renders the rest.
    """
    title = _task_title(task)
    return {
        "text": f"📖 הרצאה חדשה מוכנה: *{title}*",
        "attachments": [
            {
                "color": "#6366f1",
                "title": title,
                "title_link": _task_link(task),
                "text": _summary_preview(task),
                "footer": "zoom-to-text",
            }
        ],
    }


def _discord_payload(task: TaskResponse) -> dict:
    """Discord-native payload (richer than Slack-compatible fallback)."""
    title = _task_title(task)
    return {
        "content": f"📖 הרצאה חדשה מוכנה: **{title}**",
        "embeds": [
            {
                "title": title,
                "url": _task_link(task),
                "description": _summary_preview(task),
                "color": 6510079,  # indigo, matches the Slack color
                "footer": {"text": "zoom-to-text"},
            }
        ],
    }


def _build_payload(kind: str, task: TaskResponse) -> dict:
    if kind == "discord":
        return _discord_payload(task)
    return _slack_payload(task)


async def _send_one(
    client: httpx.AsyncClient, hook: dict, task: TaskResponse
) -> Optional[int]:
    """Send one webhook. Returns status code on completion, None on failure.

    Never raises — every error is swallowed and logged so a bad webhook
    URL can't propagate into the pipeline.
    """
    url = hook.get("url") or ""
    kind = hook.get("kind") or "slack"
    if not url.startswith("https://"):
        logger.warning("webhook %s rejected: non-https URL", hook.get("id"))
        return None
    if not _is_safe_webhook_url(url):
        # Truncate the URL in the log to avoid persisting secrets that may
        # be embedded in the path (e.g. Slack hooks include a per-channel token).
        logger.warning(
            "webhook %s rejected: SSRF-unsafe destination (%.40s…)",
            hook.get("id"), url,
        )
        return None
    try:
        payload = _build_payload(kind, task)
        resp = await client.post(url, json=payload, timeout=_HTTP_TIMEOUT)
        if resp.status_code >= 400:
            logger.warning(
                "webhook %s returned HTTP %s", hook.get("id"), resp.status_code
            )
        return resp.status_code
    except Exception as exc:  # noqa: BLE001 — third-party endpoint, anything goes
        logger.warning("webhook %s send failed: %s", hook.get("id"), exc)
        return None


async def notify_task_completed(user_id: str, task: TaskResponse) -> int:
    """Fire all enabled webhooks for a user. Returns the count of successful sends.

    Called fire-and-forget from `app.services.processor` after
    `state.complete_task`. Safe to call from a `BackgroundTasks` handler
    or `asyncio.create_task`. Failures never raise.
    """
    hooks = await state.list_enabled_webhooks(user_id)
    if not hooks:
        return 0
    sent_ok = 0
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        for hook in hooks:
            code = await _send_one(client, hook, task)
            if code is not None and code < 400:
                sent_ok += 1
    return sent_ok
