"""
FastAPI dependencies shared across routers.
"""
from typing import Optional

from fastapi import Cookie, HTTPException

from app import state


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
