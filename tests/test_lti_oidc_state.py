"""
Direct tests for app.state.store_lti_oidc_state / consume_lti_oidc_state.

These guard the contract that /lti/login and /lti/launch rely on:
  * store creates a row keyed by `state`
  * consume returns nonce/issuer/client_id once and only once
  * expired rows are not honoured even within the consume window
"""
import pytest

from app import state


@pytest.mark.asyncio
async def test_store_then_consume_returns_matching_record(client):
    await state.store_lti_oidc_state(
        state="state-1",
        nonce="nonce-1",
        issuer="https://canvas.example.com",
        client_id="client-abc",
        ttl_seconds=300,
    )
    saved = await state.consume_lti_oidc_state("state-1")
    assert saved == {
        "nonce": "nonce-1",
        "issuer": "https://canvas.example.com",
        "client_id": "client-abc",
    }


@pytest.mark.asyncio
async def test_consume_is_one_time_use(client):
    await state.store_lti_oidc_state(
        state="state-2", nonce="n", issuer="i", client_id="c", ttl_seconds=300
    )
    first = await state.consume_lti_oidc_state("state-2")
    second = await state.consume_lti_oidc_state("state-2")
    assert first is not None
    assert second is None


@pytest.mark.asyncio
async def test_consume_unknown_state_returns_none(client):
    assert await state.consume_lti_oidc_state("never-stored") is None


@pytest.mark.asyncio
async def test_expired_state_returns_none_and_is_deleted(client):
    """ttl_seconds=-1 → row is already past expiry the moment it lands."""
    await state.store_lti_oidc_state(
        state="state-3", nonce="n", issuer="i", client_id="c", ttl_seconds=-1
    )
    assert await state.consume_lti_oidc_state("state-3") is None
    # The row must have been deleted on the failed-consume path so a leaked
    # state can't be retried by an attacker.
    db = await state._get_db()
    async with db.execute(
        "SELECT 1 FROM lti_oidc_state WHERE state=?", ["state-3"]
    ) as cur:
        assert await cur.fetchone() is None
