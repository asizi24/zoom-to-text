r"""
Local dev login — mint a magic-link token WITHOUT email / Resend.

The normal flow emails a magic link via Resend. For local development you
usually have neither a Resend key nor inbound email, so this script creates
(or reuses) the user and prints a ready-to-open verify URL.

Usage (from the project root, with the venv python):
    .\venv\Scripts\python.exe scripts\dev_login.py [email]

If no email is given it falls back to ADMIN_EMAILS / ALLOWED_EMAILS from .env.
Start the app first (so it shares the same data/ DB), then open the printed
URL in your browser. The link is valid ~15 minutes.
"""
import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/dev_login.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import state          # noqa: E402
from app.config import settings  # noqa: E402


async def main(email: str) -> None:
    await state.init_db()
    try:
        user_id = await state.get_or_create_user(email)
        token = await state.create_magic_token(user_id)
    finally:
        await state.close_db()

    url = f"{settings.base_url}/api/auth/verify?token={token}"
    print()
    print(f"User : {email}")
    print("Open this URL in your browser to log in (valid ~15 min):")
    print(f"  {url}")
    print()


def _default_email() -> str:
    for src in (settings.admin_emails, settings.allowed_emails):
        first = next((e.strip() for e in src.split(",") if e.strip()), "")
        if first:
            return first
    return "dev@local"


if __name__ == "__main__":
    email = sys.argv[1] if len(sys.argv) > 1 else _default_email()
    asyncio.run(main(email))
