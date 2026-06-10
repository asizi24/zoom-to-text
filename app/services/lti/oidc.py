"""
OIDC + JWT primitives for the LTI 1.3 launch flow.

  generate_state_and_nonce()  → fresh URL-safe random tokens
  verify_id_token(...)        → fetch JWKS, verify RS256, validate LTI claims

State persistence lives in app.state (SQLite); see
store_lti_oidc_state / consume_lti_oidc_state there.

We deliberately do not depend on pylti1.3. Its only added value would
be a Flask/Django request adapter — none ships for FastAPI, and the
JWT validation surface that LTI 1.3 actually requires is ~80 lines on
top of PyJWT. Importing pylti1.3 just to write a 150-line shim
violates the "don't add deps casually" rule in CLAUDE.md.
"""
import logging
import secrets
from dataclasses import dataclass

import httpx
import jwt
from jwt import PyJWKClient

from app.services.lti.config import LtiPlatform

logger = logging.getLogger(__name__)

LTI_VERSION = "1.3.0"
LTI_MESSAGE_TYPE = "LtiResourceLinkRequest"
NS = "https://purl.imsglobal.org/spec/lti/claim"

# Per-URL JWKS clients; PyJWKClient caches the fetched keys internally.
_jwks_cache: dict[str, PyJWKClient] = {}


@dataclass(frozen=True)
class LaunchClaims:
    sub: str
    email: str
    deployment_id: str
    issuer: str
    audience: str


class LtiValidationError(Exception):
    """Raised when an id_token fails any LTI 1.3 validation check."""


def generate_state_and_nonce() -> tuple[str, str]:
    """Return (state, nonce). Both 32 bytes of OS randomness, URL-safe."""
    return secrets.token_urlsafe(32), secrets.token_urlsafe(32)


def _jwks_client(url: str) -> PyJWKClient:
    client = _jwks_cache.get(url)
    if client is None:
        client = PyJWKClient(url, cache_keys=True, lifespan=300)
        _jwks_cache[url] = client
    return client


def verify_id_token(
    id_token: str,
    platform: LtiPlatform,
    expected_nonce: str,
) -> LaunchClaims:
    """
    Validate an LTI 1.3 id_token and return the safe subset of claims.

    Checks:
      * RS256 signature against the platform JWKS (kid lookup)
      * iss == platform.issuer ; aud contains platform.client_id
      * exp / iat presence and freshness (60s leeway)
      * azp == client_id when 'aud' is a list
      * nonce == expected_nonce (anti-replay)
      * LTI version == 1.3.0 ; message_type == LtiResourceLinkRequest
      * deployment_id ∈ platform.deployment_ids
      * email claim present and non-empty
    """
    try:
        signing_key = _jwks_client(platform.jwks_url).get_signing_key_from_jwt(id_token)
    except (jwt.exceptions.PyJWKClientError, httpx.HTTPError) as exc:
        raise LtiValidationError(f"Failed to fetch JWKS: {exc}") from exc

    try:
        decoded = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=platform.client_id,
            issuer=platform.issuer,
            options={"require": ["exp", "iat", "nonce", "sub", "aud"]},
            leeway=60,
        )
    except jwt.InvalidTokenError as exc:
        raise LtiValidationError(f"id_token failed JWT validation: {exc}") from exc

    if decoded.get("nonce") != expected_nonce:
        raise LtiValidationError("nonce mismatch — possible replay")

    if isinstance(decoded.get("aud"), list) and decoded.get("azp") != platform.client_id:
        raise LtiValidationError("azp claim missing or mismatched for multi-aud token")

    if decoded.get(f"{NS}/version") != LTI_VERSION:
        raise LtiValidationError(
            f"Unsupported LTI version: {decoded.get(f'{NS}/version')!r}"
        )
    if decoded.get(f"{NS}/message_type") != LTI_MESSAGE_TYPE:
        raise LtiValidationError(
            f"Unsupported message_type: {decoded.get(f'{NS}/message_type')!r}"
        )

    deployment_id = decoded.get(f"{NS}/deployment_id")
    if deployment_id not in platform.deployment_ids:
        raise LtiValidationError(
            f"deployment_id {deployment_id!r} is not registered for this platform"
        )

    email = (decoded.get("email") or "").strip().lower()
    if not email:
        raise LtiValidationError("id_token does not include an email claim")

    return LaunchClaims(
        sub=str(decoded["sub"]),
        email=email,
        deployment_id=str(deployment_id),
        issuer=platform.issuer,
        audience=platform.client_id,
    )
