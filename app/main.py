"""
FastAPI application entry point.

Lifespan handles:
  1. Database initialization (creates tables, marks crashed tasks as failed)
  2. GCP credentials setup
  3. Background idle-watcher (unloads Whisper from RAM when not in use)
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from app import state
from app.api.routes import router
from app.api.auth import router as auth_router
from app.api.lti import router as lti_router
from app.api.streaming import router as streaming_router
from app.config import settings
from app.rate_limit import limiter
from app.services import email_digest, transcriber
from app.services.clip_extractor import ClipExtractionError, extract_clip_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Background tasks ──────────────────────────────────────────────────────────────

async def _idle_watcher():
    """
    Runs every 60 seconds. Unloads the Whisper model from RAM if it hasn't been
    used for AUTO_SHUTDOWN_IDLE_MINUTES. This prevents OOM on low-RAM machines
    between processing jobs.
    """
    while True:
        await asyncio.sleep(60)
        try:
            await transcriber.unload_model_if_idle()
        except Exception as e:
            logger.warning(f"Idle watcher error (non-fatal): {e}")


# Failed-task TTL (hours). Failed tasks not retried within this window are
# auto-deleted by _failed_task_cleanup() to free disk on the 10 GB volume.
_FAILED_TTL_HOURS = 24
_CLEANUP_INTERVAL_SECONDS = 60 * 60  # once an hour


# Weekly digest cadence — the loop wakes once an hour and per-user gating
# (last_digest_at >= 7d) decides whether to actually send.
_DIGEST_INTERVAL_SECONDS = 60 * 60  # hourly tick


async def _weekly_digest_scheduler():
    """Hourly background task that dispatches weekly digest emails.

    Per-user gating lives in `email_digest.run_digest_cycle` — this loop only
    has to wake regularly. It never raises; logs and continues on errors.
    """
    # Initial delay so a freshly-deployed server doesn't immediately blast emails.
    await asyncio.sleep(120)
    while True:
        try:
            sent = await email_digest.run_digest_cycle()
            if sent:
                logger.info(f"Weekly digest cycle: sent {sent} email(s)")
        except Exception as exc:
            logger.warning(f"Weekly digest scheduler error (non-fatal): {exc}")
        await asyncio.sleep(_DIGEST_INTERVAL_SECONDS)


async def _failed_task_cleanup():
    """
    Hourly background task that deletes failed tasks older than _FAILED_TTL_HOURS,
    along with their persisted audio file on disk. Logs the count and any
    per-file removal errors but never raises — the loop must keep running.
    """
    # Run once shortly after startup so a long-down server cleans up legacy
    # stragglers immediately instead of waiting an hour.
    await asyncio.sleep(5)
    while True:
        try:
            removed = await state.cleanup_stale_failed_tasks(_FAILED_TTL_HOURS)
            if removed:
                logger.info(
                    f"Cleanup: removed {len(removed)} stale failed task(s) "
                    f"older than {_FAILED_TTL_HOURS}h"
                )
                for entry in removed:
                    audio = entry.get("audio_path")
                    if not audio:
                        continue
                    try:
                        p = Path(audio)
                        if p.exists():
                            p.unlink()
                    except Exception as exc:
                        logger.warning(
                            f"Cleanup: could not remove audio for {entry['id']}: {exc}"
                        )
        except Exception as exc:
            logger.warning(f"Failed-task cleanup error (non-fatal): {exc}")
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)


# ── Application lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("=" * 60)
    logger.info(f"  {settings.app_title} — starting up")
    logger.info("=" * 60)

    # Initialize SQLite (creates tables + marks interrupted tasks as failed)
    await state.init_db()

    # Reset block_until / is_banned for any admin email so admins are never
    # locked out by stale rate-limit state from before the admin bypass landed.
    cleared = await state.reset_admin_flags()
    if cleared:
        logger.info(f"Admin flag reset: cleared rate-limit state for {cleared} admin user(s)")

    # Configure GCP credentials for Vertex AI / Gemini
    creds_path = settings.google_application_credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and Path(creds_path).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        logger.info(f"GCP credentials loaded from: {creds_path}")
    elif settings.google_api_key:
        logger.info("Using Gemini API key (AI Studio)")
    else:
        logger.warning(
            "No Google credentials found! "
            "Set GOOGLE_API_KEY in .env or ensure key.json is present."
        )

    # Start background idle watcher
    watcher = asyncio.create_task(_idle_watcher())
    logger.info(
        f"Idle watcher started (unloads Whisper after "
        f"{settings.auto_shutdown_idle_minutes} idle minutes)"
    )

    # Start failed-task cleanup task (auto-deletes failed tasks older than 24h)
    cleanup = asyncio.create_task(_failed_task_cleanup())
    logger.info(
        f"Failed-task cleanup started (TTL: {_FAILED_TTL_HOURS}h, "
        f"interval: {_CLEANUP_INTERVAL_SECONDS // 60}min)"
    )

    # Start weekly email digest scheduler (per-user 7d cadence enforced inside)
    digest = asyncio.create_task(_weekly_digest_scheduler())
    logger.info(
        f"Weekly digest scheduler started (tick: {_DIGEST_INTERVAL_SECONDS // 60}min)"
    )

    logger.info("✅ Server ready — listening on port 8000")
    if settings.enable_docs:
        logger.info(f"   API docs: {settings.base_url}/docs")
    else:
        logger.info("   API docs: disabled (ENABLE_DOCS=false)")

    yield  # ← application runs here

    # ── Shutdown ──
    watcher.cancel()
    cleanup.cancel()
    digest.cancel()
    await state.close_db()
    logger.info("Server shutting down — goodbye")


# ── App factory ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version="2.0.0",
    description="Transcribe and summarize Zoom class recordings with AI",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api", tags=["tasks"])
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(lti_router, prefix="/api", tags=["lti"])
app.include_router(streaming_router, tags=["streaming"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so unhandled errors return clean JSON instead of raw HTML."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", tags=["system"])
async def health():
    """Docker healthcheck endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    """Serve the login page."""
    login_path = Path("static/login.html")
    if login_path.exists():
        return FileResponse(login_path)
    return HTMLResponse("<h1>Login</h1><p>static/login.html not found</p>")


@app.get("/share/{token}", response_class=HTMLResponse, include_in_schema=False)
async def share_page(token: str):
    """Serve the SPA for a public share link — no auth required."""
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Zoom Transcriber</h1><p>static/index.html not found</p>")


# ── B4: Public audio-clip endpoint ────────────────────────────────────────────
# Streams a slice of a lecture's audio to anyone who has the clip URL.
# No auth (that's the point — it's a share link). The slice itself is gated by
# clip_id, which is a 32-char hex UUID, so guessing is infeasible.

@app.get("/clips/{clip_id}.mp3", include_in_schema=False)
async def public_audio_clip(clip_id: str):
    clip = await state.get_audio_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    audio_path = await state.get_audio_path(clip["task_id"])
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=410, detail="Source audio is no longer available")
    try:
        data = await extract_clip_bytes(audio_path, clip["start_sec"], clip["end_sec"])
    except ClipExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(
        content=data,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="clip-{clip_id[:8]}.mp3"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/clips/{clip_id}/meta", include_in_schema=False)
async def public_clip_meta(clip_id: str):
    """Public metadata for a shared clip — used by the share page UI."""
    clip = await state.get_audio_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    return {
        "id": clip["id"],
        "start_sec": clip["start_sec"],
        "end_sec": clip["end_sec"],
        "label": clip["label"],
        "created_at": clip["created_at"],
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the frontend. Redirect to /login if not authenticated."""
    session_id = request.cookies.get("session_id")
    user_id = await state.get_session_user(session_id) if session_id else None
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Zoom Transcriber</h1><p>static/index.html not found</p>")
