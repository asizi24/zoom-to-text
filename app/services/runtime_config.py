"""
Runtime-mutable configuration service.

A thin, typed layer over the app_config kv repository. This is the seam the rest
of the app reads first-boot Setup Wizard choices through:

    backend = await runtime_config.get(runtime_config.KEY_SUMMARY_BACKEND)

Why no cache: reads happen on GET / (setup gate) and per Smart Summary request
— both infrequent — and the app runs a single uvicorn worker over one SQLite
connection, so a direct SELECT is cheap and, crucially, always fresh. A stale
process-local cache would also leak across the test suite's per-test databases.

Access goes through the state facade (state.get_config / set_config / …) rather
than importing app.repositories.config directly: state fully initializes before
its re-exports are usable, which sidesteps the repository↔state import cycle
regardless of which module is imported first.
"""
from typing import Optional

from app import state

# ── Canonical keys (single source of truth for wizard + provider factory) ──────
KEY_SETUP_COMPLETE = "setup_complete"        # "true" once the wizard finishes
KEY_SUMMARY_BACKEND = "summary_backend"      # "ollama" | "gemini"
KEY_OLLAMA_MODEL = "ollama_model"            # e.g. "gemma2:9b"
KEY_GEMINI_API_KEY = "gemini_api_key"        # user-supplied key (CPU-only path)
KEY_GEMINI_MODEL = "gemini_model"            # optional override
KEY_HARDWARE_PROFILE = "hardware_profile"    # "gpu" | "cpu" (what the wizard saw)

_TRUE = {"1", "true", "yes", "on"}


async def get(key: str, default: Optional[str] = None) -> Optional[str]:
    value = await state.get_config(key)
    return default if value is None else value


async def get_bool(key: str, default: bool = False) -> bool:
    value = await state.get_config(key)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


async def set(key: str, value) -> None:
    await state.set_config(key, str(value))


async def delete(key: str) -> None:
    await state.delete_config(key)


async def all() -> dict[str, str]:
    return await state.get_all_config()


# ── Convenience helpers ────────────────────────────────────────────────────────

async def is_setup_complete() -> bool:
    return await get_bool(KEY_SETUP_COMPLETE, default=False)


async def mark_setup_complete() -> None:
    await set(KEY_SETUP_COMPLETE, "true")
