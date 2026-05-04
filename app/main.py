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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import state
from app.api.routes import router
from app.api.auth import router as auth_router
from app.api.lti import router as lti_router
from app.api.streaming import router as streaming_router
from app.config import settings
from app.services import transcriber

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


# ── Application lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("=" * 60)
    logger.info(f"  {settings.app_title} — starting up")
    logger.info("=" * 60)

    # Initialize SQLite (creates tables + marks interrupted tasks as failed)
    await state.init_db()

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

    logger.info("✅ Server ready — listening on port 8000")
    logger.info(f"   API docs: {settings.base_url}/docs")

    yield  # ← application runs here

    # ── Shutdown ──
    watcher.cancel()
    await state.close_db()
    logger.info("Server shutting down — goodbye")


# ── App factory ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version="2.0.0",
    description="Transcribe and summarize Zoom class recordings with AI",
    lifespan=lifespan,
)

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
