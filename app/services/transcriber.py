"""
Faster-Whisper transcription service.

Key design decisions:
  1. Lazy loading  — the model is NOT loaded at startup. It loads on first use.
     This prevents a 30-60 second startup delay and avoids consuming RAM when
     the server is idle.

  2. Idle unloading — after AUTO_SHUTDOWN_IDLE_MINUTES of inactivity, the model
     is deleted from memory and garbage-collected. On the next request it
     silently reloads. A 2-hour class on CPU takes ~15 min to transcribe;
     this ensures RAM is freed between jobs.

  3. Thread safety — an asyncio.Lock prevents two coroutines from loading the
     model simultaneously (which would double-allocate RAM).

  4. Executor offload — model loading and transcription are blocking operations.
     They run in a thread-pool executor to avoid blocking the event loop.
"""
import asyncio
import gc
import logging
import time
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level model state ──────────────────────────────────────────────────────
_model = None
_last_used: float = 0.0
_model_lock = asyncio.Lock()
_IDLE_THRESHOLD = settings.auto_shutdown_idle_minutes * 60  # seconds


# ── Model lifecycle ───────────────────────────────────────────────────────────────

def _load_model_sync():
    """Blocking: load the Faster-Whisper model. Runs in a thread executor."""
    from faster_whisper import WhisperModel

    cache_dir = Path.home() / ".cache" / "faster_whisper"
    logger.info(
        f"Loading Whisper model '{settings.whisper_model}' "
        f"on {settings.whisper_device} ({settings.whisper_compute_type})..."
    )
    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=str(cache_dir),
    )
    logger.info("✅ Whisper model loaded")
    return model


async def _get_model():
    """Return the loaded model, loading it if necessary (async, thread-safe)."""
    global _model, _last_used
    async with _model_lock:
        if _model is None:
            loop = asyncio.get_running_loop()
            _model = await loop.run_in_executor(None, _load_model_sync)
        _last_used = time.time()
    return _model


async def unload_model_if_idle():
    """
    Called every 60s by the idle watcher in main.py.
    Frees RAM by deleting the model if it hasn't been used recently.
    """
    global _model, _last_used
    if _model is None:
        return  # Already unloaded, nothing to do

    idle_seconds = time.time() - _last_used
    if idle_seconds > _IDLE_THRESHOLD:
        async with _model_lock:
            if _model is not None:  # Re-check inside lock
                logger.info(
                    f"Whisper idle for {idle_seconds / 60:.1f} min "
                    f"(threshold: {settings.auto_shutdown_idle_minutes} min) — unloading"
                )
                del _model
                _model = None
                gc.collect()


# ── Transcription ─────────────────────────────────────────────────────────────────

def _transcribe_sync(model, audio_path: str, language: str) -> tuple[str, str]:
    """
    Blocking transcription call.
    Returns (full_transcript_text, detected_language_code).
    """
    lang_hint = language if language != "auto" else None

    segments, info = model.transcribe(
        audio_path,
        language=lang_hint,
        beam_size=5,
        vad_filter=True,                        # Skip silent segments
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=False,                  # Saves memory
    )

    # Consume the generator and join segments
    full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return full_text, info.language


async def transcribe(audio_path: str, language: str = "he") -> tuple[str, str]:
    """
    Transcribe an audio file.
    Returns (transcript_text, detected_language).
    """
    model = await _get_model()
    loop = asyncio.get_running_loop()

    logger.info(f"Transcribing: {audio_path} (language hint: {language})")
    transcript, detected_lang = await loop.run_in_executor(
        None, _transcribe_sync, model, audio_path, language
    )

    # Update last_used timestamp after transcription completes
    global _last_used
    _last_used = time.time()

    char_count = len(transcript)
    logger.info(f"Transcription complete: {char_count:,} chars, detected language: {detected_lang}")
    return transcript, detected_lang
