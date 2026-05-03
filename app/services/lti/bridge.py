"""
Bridge a validated LTI launch into a Zoom-to-Text session.

Allowlist enforcement uses the per-platform email list from
lti_platforms.json (Q1 = B). Email matching is case-insensitive and
trimmed. After this returns, the caller sets a session_id cookie and
the rest of the app sees a regular authenticated user.
"""
import logging

from app import state
from app.services.lti.config import LtiPlatform
from app.services.lti.oidc import LaunchClaims

logger = logging.getLogger(__name__)


class LtiAuthorizationError(Exception):
    """Raised when a valid LTI launch is for an email not on the allowlist."""


async def authorize_and_create_session(
    claims: LaunchClaims,
    platform: LtiPlatform,
) -> str:
    """
    Validate that claims.email is on the platform allowlist, then return
    a fresh session_id (caller is responsible for setting the cookie).
    """
    if claims.email not in platform.allowed_emails:
        logger.warning(
            "LTI launch rejected — email %r not in platform %r allowlist",
            claims.email, platform.issuer,
        )
        raise LtiAuthorizationError(
            f"Email {claims.email} is not authorized for this platform"
        )

    user_id = await state.get_or_create_user(claims.email)
    session_id = await state.create_session(user_id)
    logger.info(
        "LTI launch authorized: email=%s issuer=%s deployment=%s",
        claims.email, claims.issuer, claims.deployment_id,
    )
    return session_id
