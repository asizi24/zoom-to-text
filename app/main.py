"""
FastAPI application entry point.

Lifespan handles:
  1. Database initialization (creates tables + migrations)
  2. Pipeline worker pool startup (re-enqueues tasks interrupted by restart)
  3. GCP credentials setup
  4. Background idle-watcher (unloads Whisper from RAM when not in use)
"""
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import __version__, state
from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.api.auth import router as auth_router
from app.config import settings
from app.logging_config import RequestContextMiddleware, setup_logging
from app.services import transcriber, worker

setup_logging(settings.log_format)
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


# ── Storage retention ─────────────────────────────────────────────────────────────

# Every terminal task's media (persisted playback audio, plus the upload source
# retained so failed/cancelled tasks stay retryable) is deleted once the task is
# older than settings.media_retention_days. Transcript/summary stay in the DB.
_RETENTION_SWEEP_INTERVAL_S = 12 * 3600  # twice a day is plenty for a day-granular policy
_DATA_ROOT = settings.data_dir.resolve()


def _safe_to_delete(path_str: str | None) -> Optional[Path]:
    """Return the Path only if it exists and resolves strictly inside data_dir —
    a guard so a malformed/injected path in the DB can never delete elsewhere."""
    if not path_str:
        return None
    try:
        p = Path(path_str).resolve()
        if p.is_file() and p.is_relative_to(_DATA_ROOT):
            return p
    except Exception:
        return None
    return None


async def _retention_sweep() -> None:
    """One pass: purge dead auth rows, then reclaim media for terminal tasks
    past the retention window."""
    # Auth hygiene is unconditional — expired sessions and burned magic tokens
    # are dead rows regardless of the media policy, and would grow forever.
    try:
        sessions, tokens = await state.purge_expired_auth()
        if sessions or tokens:
            logger.info(
                f"Retention sweep: purged {sessions} expired session(s), "
                f"{tokens} dead magic token(s)"
            )
    except Exception as exc:
        logger.warning(f"Retention: auth purge failed (non-fatal): {exc}")

    days = settings.media_retention_days
    if days <= 0:
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    tasks = await state.list_reclaimable_media(cutoff)
    freed_bytes = 0
    freed_files = 0
    for t in tasks:
        for path_str in (t["audio_path"], t["payload_file_path"]):
            p = _safe_to_delete(path_str)
            if p is None:
                continue
            try:
                freed_bytes += p.stat().st_size
                p.unlink()
                freed_files += 1
            except Exception as exc:
                logger.warning(f"Retention: could not delete {p} for task {t['id']}: {exc}")
        # Null the audio_path so has_audio is honest and we don't re-scan it.
        if t["audio_path"]:
            await state.clear_audio_path(t["id"])
        # Once the retained upload source is gone (reclaimed just now, or
        # already missing), the payload can never be replayed — drop it so
        # /retry 409s cleanly and the row stops surfacing in future sweeps.
        fp = t["payload_file_path"]
        if fp and not Path(fp).exists():
            await state.clear_job_payload(t["id"])
    if freed_files:
        logger.info(
            f"Retention sweep: reclaimed {freed_files} file(s), "
            f"{freed_bytes / 1_000_000:.1f} MB (tasks older than {days}d)"
        )

    # Orphan pass: downloads_dir accumulates strays no task row references —
    # e.g. an upload whose task creation failed, or a crash between download
    # and persist. Same age policy; payload-referenced files are never touched.
    referenced: set[str] = set()
    for fp in await state.list_payload_file_paths():
        try:
            referenced.add(str(Path(fp).resolve()))
        except Exception:
            pass
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
    orphans, orphan_bytes = _sweep_orphan_downloads(cutoff_dt, referenced)
    if orphans:
        logger.info(
            f"Retention sweep: removed {orphans} orphan download(s), "
            f"{orphan_bytes / 1_000_000:.1f} MB"
        )


def _sweep_orphan_downloads(cutoff_dt: datetime, referenced: set[str]) -> tuple[int, int]:
    """Delete files in downloads_dir older than the cutoff that no stored job
    payload references. Returns (files_deleted, bytes_freed)."""
    root = settings.downloads_dir.resolve()
    # Refuse to sweep a downloads dir outside the data root — the invariant
    # always holds at runtime (downloads_dir is derived from data_dir); a
    # mismatch means a partially-mocked test environment, not real strays.
    if not root.is_dir() or not root.is_relative_to(_DATA_ROOT):
        return 0, 0
    deleted = 0
    freed = 0
    for p in root.iterdir():
        try:
            if not p.is_file():
                continue
            mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
            if mtime >= cutoff_dt or str(p.resolve()) in referenced:
                continue
            size = p.stat().st_size
            p.unlink()
            deleted += 1
            freed += size
        except Exception as exc:
            logger.warning(f"Retention: could not remove orphan {p}: {exc}")
    return deleted, freed


async def _retention_watcher():
    """Run the retention sweep once at startup, then every 12 hours."""
    while True:
        try:
            await _retention_sweep()
        except Exception as e:
            logger.warning(f"Retention watcher error (non-fatal): {e}")
        await asyncio.sleep(_RETENTION_SWEEP_INTERVAL_S)


# ── Application lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("=" * 60)
    logger.info(f"  {settings.app_title} — starting up ({settings.environment})")
    logger.info("=" * 60)

    # Fail fast, not unsafe: in production the dev magic-link bypass (which
    # logs login tokens) must never be the fallback for a missing email key.
    if settings.is_production and not settings.resend_configured:
        raise RuntimeError(
            "ENVIRONMENT=production requires a real RESEND_API_KEY — "
            "the dev magic-link bypass logs login tokens and is disabled in "
            "production, so login would be impossible. Set the key or run "
            "with ENVIRONMENT=development."
        )

    # Working directories (config no longer mkdirs at import time)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.downloads_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite (creates tables + runs migrations)
    await state.init_db()

    # Adopt legacy ownerless tasks (rows created before per-user ownership) so
    # the strict owner check in get_task_for_user doesn't strand them.
    first_email = next(
        (e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()),
        None,
    )
    if first_email:
        owner_id = await state.get_or_create_user(first_email)
        adopted = await state.backfill_task_owners(owner_id)
        if adopted:
            logger.info(f"Adopted {adopted} legacy ownerless task(s) → {first_email}")

    # Start the pipeline worker pool — re-enqueues tasks interrupted by restart
    await worker.start()

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

    # Start background storage-retention watcher (reclaims old media on disk)
    retention_watcher = asyncio.create_task(_retention_watcher())
    if settings.media_retention_days > 0:
        logger.info(
            f"Retention watcher started (reclaims media older than "
            f"{settings.media_retention_days} days)"
        )

    logger.info("✅ Server ready — listening on port 8000")
    logger.info(f"   API docs: {settings.base_url}/docs")

    yield  # ← application runs here

    # ── Shutdown ──
    # Ordering matters — everything that can still write must be quiet before
    # the DB closes:
    #   1. watchers: a mid-sweep coroutine must not race close_db()
    #   2. worker.stop(): flags in-flight tasks for cooperative cancel and
    #      joins the worker coroutines
    #   3. transcriber.drain(): the Whisper THREAD outlives its coroutine; wait
    #      for it to pass a cancel checkpoint so no run_coroutine_threadsafe
    #      write lands on a closed connection
    #   4. close_db()
    watcher.cancel()
    retention_watcher.cancel()
    await asyncio.gather(watcher, retention_watcher, return_exceptions=True)
    await worker.stop()
    await transcriber.drain()
    await state.close_db()
    logger.info("Server shutting down — goodbye")


# ── App factory ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=__version__,
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
# Added last = outermost: every request (CORS rejections included) gets a
# request_id bound for its logs and an X-Request-ID response header.
app.add_middleware(RequestContextMiddleware)

# Uniform error envelope: {"detail", "code", "request_id"} — see app/api/errors.py
register_exception_handlers(app)

# ── Routes ────────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api", tags=["tasks"])
app.include_router(auth_router, prefix="/api", tags=["auth"])

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", tags=["system"])
async def health():
    """Liveness probe (Docker healthcheck): the process is up and serving."""
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["system"])
async def ready():
    """Readiness probe: dependencies are actually usable, not just the process.

    Checks the SQLite connection, the pipeline worker pool, and ffmpeg on PATH
    (downloads and preprocessing shell out to it). Returns 503 with the failing
    checks so a wedged dependency is visible instead of silently degrading.
    """
    checks = {
        "database": await state.ping(),
        "worker": worker.is_running(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "degraded", "checks": checks},
    )


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
