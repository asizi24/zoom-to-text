"""
API router aggregator.

The endpoints live in per-concern modules under app/api/routers/ (tasks,
events, chat, audio, flashcards); this module combines them into the single
`router` that app/main.py mounts under /api, preserving the historical
import point.

The helper re-exports at the bottom keep the pre-split import paths working
(the test suite imports `_parse_range` from here); new code should import
from the owning router module instead.
"""
from fastapi import APIRouter

from app.api.routers.audio import router as audio_router
from app.api.routers.chat import router as chat_router
from app.api.routers.events import router as events_router
from app.api.routers.flashcards import router as flashcards_router
from app.api.routers.tasks import router as tasks_router

router = APIRouter()
router.include_router(tasks_router)
router.include_router(events_router)
router.include_router(chat_router)
router.include_router(audio_router)
router.include_router(flashcards_router)

# ── Backward-compat re-exports ────────────────────────────────────────────────────
from app.api.routers.audio import (   # noqa: E402,F401
    _AUDIO_ROOT,
    _AUDIO_TYPES,
    _parse_range,
    _path_under_audio_root,
)
from app.api.routers.chat import AskRequest  # noqa: E402,F401
from app.api.routers.tasks import _ALLOWED_EXTENSIONS  # noqa: E402,F401
