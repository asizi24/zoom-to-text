"""
Tests for GET /api/lti/jwks and the underlying RSA keypair lifecycle.

The JWKS endpoint is what every registered platform fetches to verify
tokens we sign. Its shape is part of the public contract:
  - kty=RSA, alg=RS256, use=sig
  - kid is stable across calls (so platforms can pin)
  - n / e are the unsigned big-endian base64url-encoded modulus + exponent
"""
import base64

from app.services.lti import keys as lti_keys


def _b64u_decode(s: str) -> bytes:
    """Base64url decode, restoring missing padding."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def test_jwks_endpoint_returns_keys_array(client, lti_env):
    resp = client.get("/api/lti/jwks")
    assert resp.status_code == 200
    body = resp.json()
    assert "keys" in body
    assert isinstance(body["keys"], list)
    assert len(body["keys"]) == 1


def test_jwks_jwk_has_required_fields(client, lti_env):
    resp = client.get("/api/lti/jwks")
    jwk = resp.json()["keys"][0]
    assert jwk["kty"] == "RSA"
    assert jwk["alg"] == "RS256"
    assert jwk["use"] == "sig"
    assert jwk["kid"] == "lti-rsa-2048-v1"
    assert "n" in jwk and isinstance(jwk["n"], str) and jwk["n"]
    assert "e" in jwk and isinstance(jwk["e"], str) and jwk["e"]


def test_jwks_modulus_is_rsa_2048(client, lti_env):
    """n must decode to a 2048-bit unsigned big-endian integer."""
    jwk = client.get("/api/lti/jwks").json()["keys"][0]
    n_bytes = _b64u_decode(jwk["n"])
    # 2048 bits = 256 bytes; the high bit is set so leading-zero stripping
    # never shrinks it below 256.
    assert len(n_bytes) == 256
    n = int.from_bytes(n_bytes, "big")
    assert n.bit_length() == 2048


def test_jwks_first_call_generates_pem_files_on_disk(client, lti_env):
    """First /jwks fetch should auto-generate the PEM files at KEYS_DIR."""
    assert not lti_keys.PRIVATE_PATH.exists()
    assert not lti_keys.PUBLIC_PATH.exists()

    client.get("/api/lti/jwks")

    assert lti_keys.PRIVATE_PATH.exists()
    assert lti_keys.PUBLIC_PATH.exists()
    # PEM markers — we don't care about the exact bytes, just that they look right.
    assert lti_keys.PRIVATE_PATH.read_bytes().startswith(b"-----BEGIN PRIVATE KEY-----")
    assert lti_keys.PUBLIC_PATH.read_bytes().startswith(b"-----BEGIN PUBLIC KEY-----")


def test_jwks_second_call_returns_same_key(client, lti_env):
    """The persistent keypair must be reused across calls — platforms pin on n."""
    first = client.get("/api/lti/jwks").json()["keys"][0]
    second = client.get("/api/lti/jwks").json()["keys"][0]
    assert first["n"] == second["n"]
    assert first["e"] == second["e"]
    assert first["kid"] == second["kid"]
