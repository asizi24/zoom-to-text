"""
Authentication endpoints.

POST /api/auth/request   — request a magic link email
GET  /api/auth/verify    — verify token, create session, redirect home
POST /api/auth/logout    — delete session, clear cookie
"""
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Cookie, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app import state
from app.config import settings
from app.ratelimit import rate_limit

logger = logging.getLogger(__name__)
router = APIRouter()


class MagicLinkRequest(BaseModel):
    email: str


@router.post(
    "/auth/request",
    dependencies=[Depends(rate_limit("auth", "rate_limit_auth_per_minute"))],
)
async def request_magic_link(body: MagicLinkRequest):
    """
    Send a magic link to the given email if it is on the whitelist.
    Always returns the same response to prevent email enumeration.
    """
    email = body.email.strip().lower()
    allowed = {e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()}

    if email in allowed:
        user_id = await state.get_or_create_user(email)
        token = await state.create_magic_token(user_id)
        if not settings.resend_configured:
            if settings.is_production:
                # Startup validation (app/main.py) makes this unreachable, but
                # if it's ever hit, the login token must NEVER go to the logs.
                logger.error(
                    "Magic link NOT sent: Resend is not configured and the "
                    "dev bypass is disabled in production."
                )
            else:
                # Local dev bypass: without a real Resend key we couldn't email
                # anyway (e.g. developing offline), so print the link to the
                # terminal for copy-paste. Gated on environment != production —
                # a leaked log must never contain a usable login token there.
                magic_url = f"{settings.base_url}/api/auth/verify?token={token}"
                logger.info(
                    "\n" + "─" * 62 + "\n"
                    f"🔑 [DEV LOGIN] Resend not configured — magic link for {email}:\n"
                    f"   {magic_url}\n"
                    + "─" * 62
                )
        else:
            try:
                await _send_magic_link_email(email, token)
                logger.info(f"Magic link sent to {email}")
            except Exception as exc:
                logger.error(f"Failed to send magic link to {email}: {exc}")
    else:
        logger.warning(f"Magic link requested for non-whitelisted email: {email}")

    return {"message": "אם המייל רשום במערכת, תקבל קישור כניסה תוך דקה."}


@router.get("/auth/verify")
async def verify_magic_link(token: str):
    """
    Validate the magic link token.
    On success: create a 30-day session, set HttpOnly cookie, redirect to /.
    On failure: return 400.
    """
    user_id = await state.consume_magic_token(token)
    if not user_id:
        raise HTTPException(status_code=400, detail="קישור לא תקין או פג תוקף. בקש קישור חדש.")

    session_id = await state.create_session(user_id)

    # COOKIE_SECURE overrides explicitly (e.g. plain-http LAN serving);
    # None keeps the historical localhost heuristic.
    secure = (
        settings.cookie_secure
        if settings.cookie_secure is not None
        else not settings.base_url.startswith("http://localhost")
    )
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=30 * 24 * 60 * 60,
    )
    return response


@router.post("/auth/logout")
async def logout(session_id: Optional[str] = Cookie(default=None)):
    """Delete the current session and clear the cookie."""
    if session_id:
        await state.delete_session(session_id)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_id")
    return response


async def _send_magic_link_email(email: str, token: str) -> None:
    """Call the Resend API to send the magic link email."""
    magic_url = f"{settings.base_url}/api/auth/verify?token={token}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": "Zoom to Text <onboarding@resend.dev>",
                "to": [email],
                "subject": "כניסה ל-Zoom to Text",
                "html": (
                    "<div dir='rtl' style='font-family:sans-serif;max-width:400px;margin:auto'>"
                    "<h2>כניסה ל-Zoom to Text</h2>"
                    "<p>לחץ על הכפתור להתחברות:</p>"
                    f"<a href='{magic_url}' style='display:inline-block;padding:12px 24px;"
                    "background:#6c63ff;color:#fff;border-radius:8px;text-decoration:none;"
                    "font-size:16px'>כניסה למערכת</a>"
                    "<p><small>הקישור תקף ל-15 דקות בלבד.</small></p>"
                    "</div>"
                ),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
