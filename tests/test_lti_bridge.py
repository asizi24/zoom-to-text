"""
Direct tests for app.services.lti.bridge.authorize_and_create_session.

The launch endpoint already covers the bridge end-to-end; this file pins
the bridge's local contract so future refactors of /launch can't quietly
weaken authorization.
"""
import pytest

from app import state
from app.services.lti.bridge import (
    LtiAuthorizationError,
    authorize_and_create_session,
)
from app.services.lti.config import LtiPlatform
from app.services.lti.oidc import LaunchClaims


def _platform(allowed: list[str]) -> LtiPlatform:
    return LtiPlatform(
        issuer="https://canvas.example.com",
        client_id="client-abc-123",
        deployment_ids=("1:dep-xyz",),
        auth_login_url="https://canvas.example.com/auth",
        auth_token_url="https://canvas.example.com/token",
        jwks_url="https://canvas.example.com/jwks",
        allowed_emails=frozenset(e.lower().strip() for e in allowed),
    )


def _claims(email: str) -> LaunchClaims:
    return LaunchClaims(
        sub="user-sub-123",
        email=email,
        deployment_id="1:dep-xyz",
        issuer="https://canvas.example.com",
        audience="client-abc-123",
    )


@pytest.mark.asyncio
async def test_authorize_returns_session_id_for_allowlisted_email(client):
    platform = _platform(["alice@tau.ac.il"])
    session_id = await authorize_and_create_session(_claims("alice@tau.ac.il"), platform)

    assert isinstance(session_id, str) and len(session_id) > 0
    # The session must be retrievable and map back to the user
    user_id = await state.get_session_user(session_id)
    assert user_id is not None
    # And the user record must hold this email
    db = await state._get_db()
    async with db.execute("SELECT email FROM users WHERE id=?", [user_id]) as cur:
        row = await cur.fetchone()
    assert row["email"] == "alice@tau.ac.il"


@pytest.mark.asyncio
async def test_authorize_rejects_email_not_in_allowlist(client):
    platform = _platform(["alice@tau.ac.il"])
    with pytest.raises(LtiAuthorizationError):
        await authorize_and_create_session(
            _claims("stranger@elsewhere.com"), platform
        )


@pytest.mark.asyncio
async def test_authorize_does_not_create_session_when_rejected(client):
    """A rejected launch must not pollute the sessions table."""
    platform = _platform(["alice@tau.ac.il"])

    db = await state._get_db()
    async with db.execute("SELECT COUNT(*) AS n FROM sessions") as cur:
        before = (await cur.fetchone())["n"]

    with pytest.raises(LtiAuthorizationError):
        await authorize_and_create_session(_claims("stranger@elsewhere.com"), platform)

    async with db.execute("SELECT COUNT(*) AS n FROM sessions") as cur:
        after = (await cur.fetchone())["n"]
    assert before == after


@pytest.mark.asyncio
async def test_authorize_with_empty_allowlist_rejects_everyone(client):
    """A platform whose allowed_emails is empty rejects every user."""
    platform = _platform([])
    with pytest.raises(LtiAuthorizationError):
        await authorize_and_create_session(_claims("alice@tau.ac.il"), platform)
