"""
Tests for POST /api/lti/launch — the id_token verification + session bridge.

Strategy:
  * Use POST /api/lti/login to seed the OIDC state row (real flow exercise),
    then mint a signed id_token locally and POST /launch with it.
  * Replace oidc._jwks_client with a stub so PyJWT verifies signatures
    against a key we control. All other JWT validation (claims, exp, nonce,
    audience, issuer) runs unmodified.
"""
from urllib.parse import parse_qs, urlparse

from app.config import settings

from tests._lti_helpers import install_jwks, mint_id_token, make_platform_keypair


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


# ── helpers ──────────────────────────────────────────────────────────────────


def _seed_login(client, lti_env, monkeypatch):
    """Run /lti/login, return (state_token, nonce, private_pem, public_key)."""
    private_pem, public_key = make_platform_keypair()
    install_jwks(monkeypatch, public_key)

    resp = client.post("/api/lti/login", data=VALID_LOGIN_FORM, follow_redirects=False)
    assert resp.status_code == 303, resp.text
    qs = parse_qs(urlparse(resp.headers["location"]).query)
    return qs["state"][0], qs["nonce"][0], private_pem, public_key


def _happy_token(private_pem: str, nonce: str, **overrides) -> str:
    kwargs = dict(
        private_pem=private_pem,
        iss=PLATFORM["issuer"],
        aud=PLATFORM["client_id"],
        email=PLATFORM["allowed_emails"][0],
        deployment_id=PLATFORM["deployment_ids"][0],
        nonce=nonce,
    )
    kwargs.update(overrides)
    return mint_id_token(**kwargs)


# ── Happy path ───────────────────────────────────────────────────────────────


def test_launch_happy_path_returns_200_and_top_redirect_html(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )

    assert resp.status_code == 200, resp.text
    assert "text/html" in resp.headers["content-type"]
    assert "window.top.location.replace('/')" in resp.text


def test_launch_happy_path_sets_session_cookie_with_required_attributes(
    client, lti_env, monkeypatch
):
    """Cookie must be HttpOnly + Secure + SameSite=None for the cross-site LMS POST."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    set_cookie = resp.headers.get("set-cookie", "")
    assert "session_id=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "samesite=none" in set_cookie.lower()
    # 30 days = 2_592_000 seconds
    assert "Max-Age=2592000" in set_cookie


def test_launch_happy_path_authenticates_subsequent_requests(
    client, lti_env, monkeypatch
):
    """The session_id minted by /launch must authenticate subsequent requests."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 200

    # The Set-Cookie is Secure — httpx won't auto-send it back over http://testserver,
    # so extract the session_id and pass it explicitly. The DB and the auth dep
    # don't care which transport delivered it.
    session_id = resp.cookies.get("session_id")
    assert session_id, f"no session_id cookie — set-cookie was {resp.headers.get('set-cookie')!r}"

    tasks = client.get("/api/tasks?limit=10", cookies={"session_id": session_id})
    assert tasks.status_code == 200
    assert tasks.json() == []


# ── State-row failures (return 400 from /launch directly) ────────────────────


def test_launch_unknown_state_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    private_pem, public_key = make_platform_keypair()
    install_jwks(monkeypatch, public_key)
    token = _happy_token(private_pem, nonce="anything")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": "this-state-was-never-issued"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_state_is_one_time_use(client, lti_env, monkeypatch):
    """A second /launch with the same state must 400 (replay protection)."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce)

    first = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert first.status_code == 200

    second = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert second.status_code == 400


def test_launch_expired_state_returns_400(client, lti_env, monkeypatch):
    """Set TTL negative so the state row is already expired when consumed."""
    lti_env.write_platforms([PLATFORM])
    monkeypatch.setattr(settings, "lti_state_ttl_seconds", -1, raising=False)

    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


# ── id_token validation failures (LtiValidationError → 400) ──────────────────


def test_launch_tampered_signature_returns_400(client, lti_env, monkeypatch):
    """Sign with a different key than the JWKS stub returns → bad signature."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, _, public_key = _seed_login(client, lti_env, monkeypatch)
    # Mint with an unrelated keypair the JWKS stub doesn't know about
    rogue_pem, _ = make_platform_keypair()
    token = _happy_token(rogue_pem, nonce)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_nonce_mismatch_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, _, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce="not-the-nonce-we-issued")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_expired_id_token_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    # exp 1 hour in the past — past PyJWT's 60s leeway
    token = _happy_token(private_pem, nonce, exp_offset=-3600)

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_wrong_issuer_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, iss="https://attacker.example.com")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_wrong_audience_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, aud="some-other-client-id")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_multi_aud_without_azp_returns_400(client, lti_env, monkeypatch):
    """When aud is a list, azp must equal our client_id."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(
        private_pem, nonce,
        aud=[PLATFORM["client_id"], "other-aud"],
        azp="other-aud",  # wrong azp
    )

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_wrong_lti_version_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, version="1.2.0")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_wrong_message_type_returns_400(client, lti_env, monkeypatch):
    """Only LtiResourceLinkRequest is accepted; DeepLinkingRequest etc. are rejected."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, message_type="LtiDeepLinkingRequest")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_unregistered_deployment_id_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, deployment_id="99:rogue-deployment")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_launch_missing_email_claim_returns_400(client, lti_env, monkeypatch):
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, drop_claims=("email",))

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 400


# ── Authorization failures (allowlist → 403) ─────────────────────────────────


def test_launch_email_not_in_allowlist_returns_403(client, lti_env, monkeypatch):
    """Token is fully valid but the user isn't in the platform's allow-list."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, email="stranger@elsewhere.com")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 403


def test_launch_allowlist_match_is_case_insensitive(client, lti_env, monkeypatch):
    """Token email in mixed case + whitespace must still match the allowlist."""
    lti_env.write_platforms([PLATFORM])
    state_token, nonce, private_pem, _ = _seed_login(client, lti_env, monkeypatch)
    token = _happy_token(private_pem, nonce, email="  Alice@TAU.AC.IL  ")

    resp = client.post(
        "/api/lti/launch",
        data={"id_token": token, "state": state_token},
        follow_redirects=False,
    )
    assert resp.status_code == 200
