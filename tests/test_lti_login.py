"""
Tests for the OIDC initiation endpoint POST /api/lti/login (and GET).

The platform calls /lti/login first with iss, login_hint, target_link_uri,
client_id, lti_message_hint. We must:
  * 400 if a required field is missing
  * 404 if (iss, client_id) is not in lti_platforms.json
  * 303 to the platform's auth_login_url with the canonical OIDC params
  * persist the state/nonce row so /launch can consume it
"""
from urllib.parse import parse_qs, urlparse

import pytest

from app import state as state_db


PLATFORM = {
    "issuer": "https://canvas.example.com",
    "client_id": "client-abc-123",
    "deployment_ids": ["1:dep-xyz"],
    "auth_login_url": "https://canvas.example.com/api/lti/authorize_redirect",
    "auth_token_url": "https://canvas.example.com/login/oauth2/token",
    "jwks_url": "https://canvas.example.com/api/lti/security/jwks",
    "allowed_emails": ["alice@tau.ac.il"],
}

VALID_LOGIN_FORM = {
    "iss": PLATFORM["issuer"],
    "client_id": PLATFORM["client_id"],
    "login_hint": "user-456",
    "target_link_uri": "https://zoom-to-text.fly.dev/",
    "lti_message_hint": "msg-hint-789",
}


# ── Sad paths ────────────────────────────────────────────────────────────────


def test_login_missing_iss_returns_400(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    form = {k: v for k, v in VALID_LOGIN_FORM.items() if k != "iss"}
    resp = client.post("/api/lti/login", data=form, follow_redirects=False)
    assert resp.status_code == 400


def test_login_missing_login_hint_returns_400(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    form = {k: v for k, v in VALID_LOGIN_FORM.items() if k != "login_hint"}
    resp = client.post("/api/lti/login", data=form, follow_redirects=False)
    assert resp.status_code == 400


def test_login_missing_target_link_uri_returns_400(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    form = {k: v for k, v in VALID_LOGIN_FORM.items() if k != "target_link_uri"}
    resp = client.post("/api/lti/login", data=form, follow_redirects=False)
    assert resp.status_code == 400


def test_login_unknown_platform_returns_404(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    form = dict(VALID_LOGIN_FORM, iss="https://unregistered.example.com")
    resp = client.post("/api/lti/login", data=form, follow_redirects=False)
    assert resp.status_code == 404


def test_login_no_platforms_file_returns_404(client, lti_env):
    """Empty platforms file → no platform matches → 404."""
    lti_env.write_platforms([])
    resp = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)
    assert resp.status_code == 404


# ── Happy path ───────────────────────────────────────────────────────────────


def test_login_redirects_303_to_auth_login_url(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    resp = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)

    assert resp.status_code == 303
    location = resp.headers["location"]
    parsed = urlparse(location)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == PLATFORM["auth_login_url"]


def test_login_redirect_carries_canonical_oidc_params(client, lti_env):
    lti_env.write_platforms([PLATFORM])
    resp = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)

    qs = parse_qs(urlparse(resp.headers["location"]).query)
    assert qs["scope"] == ["openid"]
    assert qs["response_type"] == ["id_token"]
    assert qs["response_mode"] == ["form_post"]
    assert qs["prompt"] == ["none"]
    assert qs["client_id"] == [PLATFORM["client_id"]]
    assert qs["redirect_uri"] == ["http://testserver/api/lti/launch"]
    assert qs["login_hint"] == [VALID_LOGIN_FORM["login_hint"]]
    assert qs["lti_message_hint"] == [VALID_LOGIN_FORM["lti_message_hint"]]
    # state and nonce are present, non-empty, and >= 32 bytes URL-safe
    assert len(qs["state"][0]) >= 32
    assert len(qs["nonce"][0]) >= 32
    # state and nonce are independent random values
    assert qs["state"][0] != qs["nonce"][0]


def test_login_each_call_issues_fresh_state_and_nonce(client, lti_env):
    """Two successive logins must produce different state + nonce — anti-replay."""
    lti_env.write_platforms([PLATFORM])
    a = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)
    b = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)
    qa = parse_qs(urlparse(a.headers["location"]).query)
    qb = parse_qs(urlparse(b.headers["location"]).query)
    assert qa["state"][0] != qb["state"][0]
    assert qa["nonce"][0] != qb["nonce"][0]


@pytest.mark.asyncio
async def test_login_persists_state_row_with_matching_nonce(client, lti_env):
    """The state token in the redirect must be consumable, with the same nonce."""
    lti_env.write_platforms([PLATFORM])
    resp = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    state_token = qs["state"][0]
    expected_nonce = qs["nonce"][0]

    saved = await state_db.consume_lti_oidc_state(state_token)
    assert saved is not None
    assert saved["nonce"] == expected_nonce
    assert saved["issuer"] == PLATFORM["issuer"]
    assert saved["client_id"] == PLATFORM["client_id"]


def test_login_omits_lti_message_hint_when_not_provided(client, lti_env):
    """If the platform doesn't send lti_message_hint, we don't echo it."""
    lti_env.write_platforms([PLATFORM])
    form = {k: v for k, v in VALID_LOGIN_FORM.items() if k != "lti_message_hint"}
    resp = client.post("/api/lti/login", data=form, follow_redirects=False)
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    assert "lti_message_hint" not in qs


def test_login_get_method_also_accepted(client, lti_env):
    """Spec allows GET initiation; we read params from query string."""
    lti_env.write_platforms([PLATFORM])
    resp = client.get("/api/lti/login", params=VALID_LOGIN_FORM, follow_redirects=False)
    assert resp.status_code == 303
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    assert qs["client_id"] == [PLATFORM["client_id"]]
