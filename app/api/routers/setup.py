"""
First-boot Setup Wizard endpoints.

These are intentionally UNAUTHENTICATED — on a truly fresh install there is no
session yet — but they SELF-DISABLE the moment setup is complete: every
mutating/probing route depends on `require_setup_incomplete`, which returns 403
once the wizard has run. GET /setup/status stays open so the frontend can
always ask "am I set up?".

Flow:
  GET  /api/setup/status                 → {setup_complete, summary_backend, …}
  GET  /api/setup/hardware               → probe result + Ollama availability
  GET  /api/setup/ollama/pull/stream     → SSE: model-download progress
  POST /api/setup/complete               → persist choice, flip setup_complete
"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.services import hardware, runtime_config
from app.services.llm.base import LLMError
from app.services.llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)
router = APIRouter()


async def require_setup_incomplete() -> bool:
    """Gate that 403s once the wizard has already run — so setup endpoints can
    never be re-driven (or abused) after first boot."""
    if await runtime_config.is_setup_complete():
        raise HTTPException(status_code=403, detail="ההגדרה כבר הושלמה")
    return True


# ── Status (always available) ─────────────────────────────────────────────────

@router.get("/setup/status")
async def setup_status():
    return {
        "setup_complete": await runtime_config.is_setup_complete(),
        "summary_backend": await runtime_config.get(runtime_config.KEY_SUMMARY_BACKEND),
        "ollama_model": await runtime_config.get(runtime_config.KEY_OLLAMA_MODEL),
    }


# ── Hardware probe ────────────────────────────────────────────────────────────

@router.get("/setup/hardware", dependencies=[Depends(require_setup_incomplete)])
async def setup_hardware():
    """Detect the GPU (off-thread so the probe never blocks the loop) and report
    whether the local Ollama server is reachable + which models it already has."""
    info = await asyncio.to_thread(hardware.detect_hardware, settings.ollama_model)

    probe = OllamaProvider(settings.ollama_host, settings.ollama_model, timeout=10.0)
    ollama_available = await probe.is_available()
    installed_models: list[str] = []
    if ollama_available:
        try:
            installed_models = await probe.list_models()
        except LLMError:
            installed_models = []

    return {
        **info.as_dict(),
        "ollama_available": ollama_available,
        "installed_models": installed_models,
        "default_model": settings.ollama_model,
    }


# ── Model pull (SSE progress) ─────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/setup/ollama/pull/stream", dependencies=[Depends(require_setup_incomplete)])
async def pull_model_stream(model: Optional[str] = Query(None)):
    """Stream Ollama's model-download progress to the wizard's progress bar.

    Server-Sent Events (so a plain EventSource drives it). Each `progress` event
    carries the current layer's percent + Ollama's status text; a terminal
    `done` (or `error`) closes the stream.
    """
    model_name = (model or settings.ollama_model).strip()
    provider = OllamaProvider(settings.ollama_host, model_name, timeout=settings.ollama_timeout)

    async def generate():
        try:
            async for event in provider.pull(model_name):
                status = event.get("status", "")
                total = event.get("total")
                completed = event.get("completed")
                percent = None
                if isinstance(total, (int, float)) and total > 0 and completed is not None:
                    percent = max(0, min(100, round(100 * completed / total)))
                yield _sse({
                    "type": "progress",
                    "status": status,
                    "percent": percent,
                    "total": total,
                    "completed": completed,
                })
                if status == "success":
                    yield _sse({"type": "done", "model": model_name})
                    return
            # Stream ended without an explicit success line — treat as done.
            yield _sse({"type": "done", "model": model_name})
        except LLMError as exc:
            logger.warning("Ollama pull failed for %s: %s", model_name, exc.detail)
            yield _sse({"type": "error", "message": exc.user_message, "detail": exc.detail})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Finish setup ──────────────────────────────────────────────────────────────

class SetupComplete(BaseModel):
    backend: str                          # "ollama" | "gemini"
    ollama_model: Optional[str] = None
    gemini_api_key: Optional[str] = None


@router.post("/setup/complete", dependencies=[Depends(require_setup_incomplete)])
async def complete_setup(body: SetupComplete):
    """Persist the chosen backend and flip setup_complete. After this the wizard
    disappears (GET / serves the app) and these endpoints 403."""
    backend = body.backend.strip().lower()
    if backend not in ("ollama", "gemini"):
        raise HTTPException(status_code=400, detail="backend חייב להיות 'ollama' או 'gemini'")

    if backend == "ollama":
        model = (body.ollama_model or settings.ollama_model).strip()
        if not model:
            raise HTTPException(status_code=400, detail="חסר שם מודל ל-Ollama")
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "ollama")
        await runtime_config.set(runtime_config.KEY_OLLAMA_MODEL, model)
        await runtime_config.set(runtime_config.KEY_HARDWARE_PROFILE, "gpu")
    else:
        api_key = (body.gemini_api_key or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="נדרש מפתח Gemini API עבור מסלול הענן")
        await runtime_config.set(runtime_config.KEY_SUMMARY_BACKEND, "gemini")
        await runtime_config.set(runtime_config.KEY_GEMINI_API_KEY, api_key)
        await runtime_config.set(runtime_config.KEY_HARDWARE_PROFILE, "cpu")

    await runtime_config.mark_setup_complete()
    logger.info("Setup wizard complete — Smart Summary backend: %s", backend)
    return {"status": "ok", "backend": backend}
