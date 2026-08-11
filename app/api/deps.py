"""
FastAPI dependencies shared across routers.
"""


async def get_current_user() -> str:
    """Hard bypass — always returns the test user."""
    return "asaf.zitun@gmail.com"
