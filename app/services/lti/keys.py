"""
RSA key pair for our LTI tool.

Keys live at <data_dir>/lti_keys/{private,public}.pem. Generated once
on first call, reused across deploys (Fly volume is persistent).

Rotation: delete both files, restart, then ask each registered platform
to re-fetch /api/lti/jwks.
"""
import base64
import logging
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from app.config import settings

logger = logging.getLogger(__name__)

KEYS_DIR = settings.data_dir / "lti_keys"
PRIVATE_PATH = KEYS_DIR / "private.pem"
PUBLIC_PATH = KEYS_DIR / "public.pem"
KID = "lti-rsa-2048-v1"  # bumped on rotation; included in JWKS + signed JWTs

_private_key: Optional[RSAPrivateKey] = None
_public_key: Optional[RSAPublicKey] = None


def _generate_and_persist() -> RSAPrivateKey:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    PRIVATE_PATH.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    PUBLIC_PATH.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    try:
        PRIVATE_PATH.chmod(0o600)
    except (OSError, NotImplementedError):
        pass  # chmod is a no-op on Windows; private key still has filesystem ACLs.
    logger.info("Generated new LTI RSA-2048 key pair at %s", KEYS_DIR)
    return key


def _load_or_generate() -> RSAPrivateKey:
    if PRIVATE_PATH.exists():
        loaded = serialization.load_pem_private_key(
            PRIVATE_PATH.read_bytes(), password=None
        )
        if not isinstance(loaded, RSAPrivateKey):
            raise RuntimeError(f"Expected RSAPrivateKey at {PRIVATE_PATH}, got {type(loaded)}")
        return loaded
    return _generate_and_persist()


def get_private_key() -> RSAPrivateKey:
    global _private_key
    if _private_key is None:
        _private_key = _load_or_generate()
    return _private_key


def get_public_key() -> RSAPublicKey:
    global _public_key
    if _public_key is None:
        _public_key = get_private_key().public_key()
    return _public_key


def _b64url_uint(value: int) -> str:
    """Encode an int as the unsigned big-endian base64url string used in JWKs."""
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big") or b"\x00"
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def get_public_jwk() -> dict:
    """Return our public key in JWK format (RFC 7517) for the /jwks endpoint."""
    pub = get_public_key()
    numbers = pub.public_numbers()
    return {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": KID,
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
    }
