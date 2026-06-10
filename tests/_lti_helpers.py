"""
LTI test helpers — generate keys, mint signed id_tokens, install fake JWKS.

Not a test file (leading underscore in filename) — imported by the
test_lti_*.py modules.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

NS = "https://purl.imsglobal.org/spec/lti/claim"
PLATFORM_KID = "test-platform-key"


def make_platform_keypair() -> tuple[str, RSAPublicKey]:
    """Generate a platform-side RSA-2048 keypair for signing id_tokens.

    Returns (private_pem_str, public_key_obj). The public key object is
    handed to install_jwks_for_platform() so PyJWT can verify signatures
    without a real JWKS HTTP fetch.
    """
    key: RSAPrivateKey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    return private_pem, key.public_key()


def mint_id_token(
    *,
    private_pem: str,
    iss: str,
    aud: str | list[str],
    sub: str = "user-sub-123",
    email: str = "alice@tau.ac.il",
    deployment_id: str = "1:abcdef1234567890",
    nonce: str = "test-nonce",
    azp: Optional[str] = None,
    version: str = "1.3.0",
    message_type: str = "LtiResourceLinkRequest",
    exp_offset: int = 600,
    iat_offset: int = 0,
    kid: str = PLATFORM_KID,
    drop_claims: tuple[str, ...] = (),
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Mint an LTI 1.3 id_token. Defaults yield a happy-path token; override
    fields or use drop_claims to construct invalid variants."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "email": email,
        "exp": now + exp_offset,
        "iat": now + iat_offset,
        "nonce": nonce,
        f"{NS}/version": version,
        f"{NS}/message_type": message_type,
        f"{NS}/deployment_id": deployment_id,
    }
    if azp is not None:
        payload["azp"] = azp
    elif isinstance(aud, list):
        payload["azp"] = aud[0]
    if extra_claims:
        payload.update(extra_claims)
    for key in drop_claims:
        payload.pop(key, None)
    return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})


class _StubKey:
    def __init__(self, key: RSAPublicKey):
        self.key = key


class _StubJWKClient:
    def __init__(self, key: RSAPublicKey):
        self._key = key

    def get_signing_key_from_jwt(self, token: str) -> _StubKey:  # noqa: ARG002 - token unused
        return _StubKey(self._key)


def install_jwks(monkeypatch, public_key: RSAPublicKey) -> None:
    """Replace oidc._jwks_client so signature verification uses public_key.

    Bypasses the real httpx fetch in tests — we still exercise PyJWT's
    full claim-validation pipeline (signature, iss, aud, exp, nonce, etc.)."""
    from app.services.lti import oidc

    stub = _StubJWKClient(public_key)
    monkeypatch.setattr(oidc, "_jwks_client", lambda url: stub)
