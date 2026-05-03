"""
LTI 1.3 endpoints — institutional SSO via Canvas / Moodle / etc.

  GET  /api/lti/jwks    — our public RSA key (platform fetches at registration)
  POST /api/lti/login   — OIDC initiation (platform calls us first)
  POST /api/lti/launch  — id_token verification + session bridge

Only /lti/launch issues a session_id cookie. The cookie attributes are
intentionally different from the magic-link cookie:

  magic-link:  SameSite=Lax     (same-origin redirect)
  LTI launch:  SameSite=None    (cross-origin POST from the LMS)

Both write the same opaque session_id value, so deps.get_current_user
treats LTI- and magic-link-authenticated users identically.

After a successful launch we render a tiny HTML page that uses
window.top.location to break out of the Canvas iframe (Q5b).
"""
import logging
from urllib.parse import urlencode

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app import state as state_db
from app.config import settings
from app.services.lti import bridge, config, keys, oidc

logger = logging.getLogger(__name__)
router = APIRouter()


# ── /jwks ────────────────────────────────────────────────────────────────────

@router.get("/lti/jwks")
async def lti_jwks() -> JSONResponse:
    """Public key set in JWK format. Platforms fetch this during tool registration."""
    return JSONResponse({"keys": [keys.get_public_jwk()]})


# ── /login (OIDC initiation) ─────────────────────────────────────────────────

@router.post("/lti/login")
@router.get("/lti/login")
async def lti_login(request: Request) -> RedirectResponse:
    """
    Step 1 of the LTI 1.3 launch flow.

    The platform POSTs (or, less commonly, GETs) here with iss, login_hint,
    target_link_uri, lti_message_hint, and client_id. We look up the platform,
    persist a fresh (state, nonce) pair to the SQLite OIDC table, and 303 the
    user to the platform's auth_login_url to complete the OIDC handshake.
    """
    if request.method == "POST":
        form = await request.form()
        params_in = {k: v for k, v in form.items() if isinstance(v, str)}
    else:
        params_in = dict(request.query_params)

    iss = (params_in.get("iss") or "").strip()
    login_hint = params_in.get("login_hint")
    target_link_uri = params_in.get("target_link_uri")
    lti_message_hint = params_in.get("lti_message_hint")
    client_id = (params_in.get("client_id") or "").strip()

    if not iss or not login_hint or not target_link_uri:
        raise HTTPException(status_code=400, detail="Missing required OIDC parameters")

    platform = config.get_platform(iss, client_id)
    if platform is None:
        logger.warning("LTI /login: unknown platform iss=%r client_id=%r", iss, client_id)
        raise HTTPException(status_code=404, detail="Unknown LTI platform")

    state_token, nonce = oidc.generate_state_and_nonce()
    await state_db.store_lti_oidc_state(
        state=state_token,
        nonce=nonce,
        issuer=platform.issuer,
        client_id=platform.client_id,
        ttl_seconds=settings.lti_state_ttl_seconds,
    )

    redirect_uri = f"{settings.base_url.rstrip('/')}/api/lti/launch"
    params_out = {
        "scope": "openid",
        "response_type": "id_token",
        "response_mode": "form_post",
        "prompt": "none",
        "client_id": platform.client_id,
        "redirect_uri": redirect_uri,
        "state": state_token,
        "nonce": nonce,
        "login_hint": login_hint,
    }
    if lti_message_hint:
        params_out["lti_message_hint"] = lti_message_hint

    auth_url = f"{platform.auth_login_url}?{urlencode(params_out)}"
    return RedirectResponse(url=auth_url, status_code=303)


# ── /launch (id_token validation + session bridge) ───────────────────────────

@router.post("/lti/launch")
async def lti_launch(
    id_token: str = Form(...),
    state: str = Form(...),
) -> HTMLResponse:
    """
    Step 2 of the LTI 1.3 launch flow.

    The platform POSTs the signed id_token + the state token we issued.
    We:
      1. Look up & consume the state row (one-time use).
      2. Verify the id_token signature + claims against the platform JWKS.
      3. Enforce the per-platform email allowlist.
      4. Mint a session_id and set the cross-site cookie.
      5. Return a tiny HTML page that redirects window.top to '/'.
    """
    saved = await state_db.consume_lti_oidc_state(state)
    if saved is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired state — relaunch from your LMS",
        )

    platform = config.get_platform(saved["issuer"], saved["client_id"])
    if platform is None:
        raise HTTPException(
            status_code=400, detail="Platform configuration was removed mid-flow"
        )

    try:
        claims = oidc.verify_id_token(id_token, platform, expected_nonce=saved["nonce"])
    except oidc.LtiValidationError as exc:
        logger.warning("LTI launch — id_token validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid id_token: {exc}")

    try:
        session_id = await bridge.authorize_and_create_session(claims, platform)
    except bridge.LtiAuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    response = HTMLResponse(content=_TOP_REDIRECT_HTML)
    # SameSite=None + Secure: required for the cross-site POST flow from the LMS (Q5a).
    # Magic-link cookie path stays SameSite=Lax — see app/api/auth.py.
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=30 * 24 * 60 * 60,
    )
    return response


# Hardcoded landing page — no user content is interpolated, so no XSS surface.
_TOP_REDIRECT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Loading...</title></head>
<body><script>
  try { window.top.location.replace('/'); }
  catch (e) { window.location.replace('/'); }
</script><noscript><a href="/">Continue</a></noscript></body></html>
"""
