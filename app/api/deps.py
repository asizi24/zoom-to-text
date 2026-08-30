"""
FastAPI dependencies shared across routers.
"""
from fastapi import Request, HTTPException, status
from app import state


async def get_current_user(request: Request) -> str:
    """Retrieve user_id from session cookie."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    user_id = await state.get_session_user(session_id)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
        )
    return user_id
