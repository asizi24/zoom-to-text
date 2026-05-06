"""
FastAPI dependencies shared across routers.
"""
from typing import Optional

from fastapi import Cookie, Depends, HTTPException

from app import state
import app.api.auth as _auth_module


async def get_current_user(
    session_id: Optional[str] = Cookie(default=None),
) -> str:
    """
    Read session_id cookie and return the authenticated user_id.
    Raises HTTP 401 if the session is missing or expired.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = await state.get_session_user(session_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Session expired — please log in again")
    return user_id


async def enforce_rate_limit(user_id: str = Depends(get_current_user)) -> str:
    """
    Enforce the per-user 24-hour task-submission limit.

    Use as a dependency on task-creation endpoints in place of get_current_user.

    - Returns user_id when the request is within quota.
    - Raises 429 on the 3rd request within 24 h; applies a 24-hour block and
      dispatches a warning email.
    - Raises 403 when a blocked user makes any request (permanent ban applied),
      or when the account is already permanently banned.
    """
    action = await state.check_and_record_request(user_id)
    if action == "allow":
        return user_id
    if action == "block_now":
        email = await state.get_user_email(user_id)
        if email:
            await _auth_module._send_rate_limit_warning_email(email)
        raise HTTPException(
            status_code=429,
            detail="חרגת ממכסת הבקשות היומית. חשבונך חסום ל-24 שעות.",
        )
    # "ban_now" or "reject_banned"
    raise HTTPException(
        status_code=403,
        detail="חשבונך נחסם לצמיתות בשל שימוש לרעה.",
    )
