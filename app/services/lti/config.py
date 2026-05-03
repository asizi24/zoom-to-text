"""
Per-platform LTI 1.3 configuration loader.

Config lives at <data_dir>/lti_platforms.json — a JSON array of records,
one per registered LMS install. The file is read on first lookup and
cached in-process; restart the app after editing.

Record shape:

    {
      "issuer":         "https://canvas.instructure.com",
      "client_id":      "10000000000001",
      "deployment_ids": ["1:abcdef..."],
      "auth_login_url": "https://canvas.instructure.com/api/lti/authorize_redirect",
      "auth_token_url": "https://canvas.instructure.com/login/oauth2/token",
      "jwks_url":       "https://canvas.instructure.com/api/lti/security/jwks",
      "allowed_emails": ["alice@tau.ac.il", "bob@tau.ac.il"]
    }

Lookups are keyed by (issuer, client_id). Different deployments under
one issuer/client_id share the same record; deployment_id is checked
separately at launch time against the deployment_ids list.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

PLATFORMS_FILE = settings.data_dir / "lti_platforms.json"


@dataclass(frozen=True)
class LtiPlatform:
    issuer: str
    client_id: str
    deployment_ids: tuple[str, ...]
    auth_login_url: str
    auth_token_url: str
    jwks_url: str
    allowed_emails: frozenset[str]


_platforms: Optional[dict[tuple[str, str], LtiPlatform]] = None


def _load() -> dict[tuple[str, str], LtiPlatform]:
    if not PLATFORMS_FILE.exists():
        logger.warning(
            "LTI platforms config not found at %s — LTI launches will 404.",
            PLATFORMS_FILE,
        )
        return {}
    try:
        raw = json.loads(PLATFORMS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("Failed to read %s: %s", PLATFORMS_FILE, exc)
        return {}
    if not isinstance(raw, list):
        logger.error("LTI platforms config must be a JSON array, got %s", type(raw).__name__)
        return {}

    out: dict[tuple[str, str], LtiPlatform] = {}
    for i, entry in enumerate(raw):
        try:
            p = LtiPlatform(
                issuer=str(entry["issuer"]),
                client_id=str(entry["client_id"]),
                deployment_ids=tuple(str(d) for d in entry["deployment_ids"]),
                auth_login_url=str(entry["auth_login_url"]),
                auth_token_url=str(entry["auth_token_url"]),
                jwks_url=str(entry["jwks_url"]),
                allowed_emails=frozenset(
                    e.strip().lower()
                    for e in entry.get("allowed_emails", [])
                    if isinstance(e, str) and e.strip()
                ),
            )
        except (KeyError, TypeError) as exc:
            logger.error("LTI platforms[%d] is malformed: %s — skipping", i, exc)
            continue
        out[(p.issuer, p.client_id)] = p
    logger.info("Loaded %d LTI platform(s) from %s", len(out), PLATFORMS_FILE)
    return out


def get_platform(issuer: str, client_id: str) -> Optional[LtiPlatform]:
    """Return the platform record for (issuer, client_id), or None if unknown."""
    global _platforms
    if _platforms is None:
        _platforms = _load()
    return _platforms.get((issuer, client_id))


def reset_cache() -> None:
    """Force re-read of PLATFORMS_FILE on next lookup. Test-only."""
    global _platforms
    _platforms = None
