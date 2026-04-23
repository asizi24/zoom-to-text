"""
Whisper transcription service — two backends:

  WHISPER_LOCAL (default)
    Runs Faster-Whisper on the local machine. No API key needed.
    The model is lazy-loaded, idle-unloaded, and thread-safe.
    Handles files of any length natively.

  WHISPER_API (OpenAI)
    Sends audio to OpenAI's Whisper API (whisper-1).
    Requires OPENAI_API_KEY in the environment / .env.
    Because the API has a 25 MB file size limit (~22 min at 96 kbps),
    the audio is preprocessed first:
      1. Silence removal  — strips dead air with ffmpeg silenceremove
      2. Chunking         — splits into ≤13-min pieces (safely under the limit)
    Each chunk is sent independently; transcripts are joined in order.
"""
import asyncio
import gc
import logging
import time
from pathlib import Path

import httpx

from app.config import settings
from app.services import audio_preprocessor

logger = logging.getLogger(__name__)


# ── Local model state ─────────────────────────────────────────────────────────
# Two independent caches — one for vanilla faster-whisper, one for ivrit-ai's
# Hebrew-tuned model. They are loaded on demand and unloaded separately after
# _IDLE_THRESHOLD seconds of inactivity. Keeping them separate (rather than
# one "active model" slot) lets users switch modes per task without paying
# a reload penalty each time.

_model = None
_last_used: float = 0.0
_model_lock = asyncio.Lock()

_ivrit_model = None
_ivrit_last_used: float = 0.0
_ivrit_lock = asyncio.Lock()

_IDLE_THRESHOLD = settings.auto_shutdown_idle_minutes * 60


# ── Model lifecycle (local only) ──────────────────────────────────────────────

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
    Frees RAM by deleting models that haven't been used recently.
    Handles both the vanilla Whisper cache and the ivrit-ai cache independently.
    """
    global _model, _last_used, _ivrit_model, _ivrit_last_used

    if _model is not None:
        idle = time.time() - _last_used
        if idle > _IDLE_THRESHOLD:
            async with _model_lock:
                if _model is not None:
                    logger.info(
                        f"Whisper idle for {idle / 60:.1f} min "
                        f"(threshold: {settings.auto_shutdown_idle_minutes} min) — unloading"
                    )
                    del _model
                    _model = None
                    gc.collect()

    if _ivrit_model is not None:
        idle = time.time() - _ivrit_last_used
        if idle > _IDLE_THRESHOLD:
            async with _ivrit_lock:
                if _ivrit_model is not None:
                    logger.info(
                        f"ivrit-ai idle for {idle / 60:.1f} min — unloading"
                    )
                    del _ivrit_model
                    _ivrit_model = None
                    gc.collect()


# ── LOCAL transcription ───────────────────────────────────────────────────────

def _transcribe_sync(
    model,
    audio_path: str,
    language: str,
    segment_cb=None,
) -> tuple[str, str]:
    """
    Blocking transcription — runs in a thread executor.
    Returns (full_transcript_text, detected_language_code).

    segment_cb: optional sync callable(text: str) called every ~10 segments or
    ~5 seconds so callers can stream live text to the DB for the preview panel.
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

    full_texts: list[str] = []
    buffer: list[str] = []
    last_flush = time.time()

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        full_texts.append(text)

        if segment_cb is not None:
            buffer.append(text)
            now = time.time()
            # Flush every 10 segments or every 5 seconds to avoid DB overload
            if len(buffer) >= 10 or now - last_flush >= 5.0:
                segment_cb(" ".join(buffer) + " ")
                buffer = []
                last_flush = now

    # Final flush — make sure nothing is left in the buffer
    if segment_cb is not None and buffer:
        segment_cb(" ".join(buffer) + " ")

    return " ".join(full_texts), info.language


async def transcribe(
    audio_path: str,
    language: str = "he",
    task_id: str | None = None,
) -> tuple[str, str]:
    """
    Transcribe an audio file locally with Faster-Whisper.
    Returns (transcript_text, detected_language).

    Pass task_id to enable live transcript preview: each segment batch is
    appended to the task's partial_transcript column so the UI can poll it.
    """
    from app import state as _state  # local import avoids circular at module level

    model = await _get_model()
    loop  = asyncio.get_running_loop()

    # Build a thread-safe segment callback that fires-and-forgets into the event loop.
    # Errors from append_partial_transcript are logged via a done-callback so they
    # don't silently disappear inside the Future.
    if task_id is not None:
        def segment_cb(text: str, _tid=task_id, _loop=loop) -> None:
            future = asyncio.run_coroutine_threadsafe(
                _state.append_partial_transcript(_tid, text),
                _loop,
            )
            future.add_done_callback(
                lambda f: f.exception() and logger.warning(
                    "partial transcript write failed for %s: %s", _tid, f.exception()
                )
            )
    else:
        segment_cb = None

    logger.info(f"[Local Whisper] Transcribing: {audio_path} (language: {language})")
    transcript, detected_lang = await loop.run_in_executor(
        None, _transcribe_sync, model, audio_path, language, segment_cb
    )

    global _last_used
    _last_used = time.time()

    logger.info(
        f"[Local Whisper] Done: {len(transcript):,} chars, "
        f"detected language: {detected_lang}"
    )
    return transcript, detected_lang


# ── OPENAI API transcription ──────────────────────────────────────────────────

async def transcribe_via_api(
    audio_path: str,
    language: str = "he",
    task_id: str | None = None,
) -> tuple[str, str]:
    """
    Transcribe an audio file via OpenAI's Whisper API (whisper-1).

    Steps:
      1. Silence removal  (ffmpeg) — strips dead air to reduce file size
      2. Chunking         (ffmpeg) — splits into ≤13-min pieces under the 25 MB API limit
      3. API calls        (httpx)  — each chunk sent independently
      4. Join                      — transcripts concatenated in order

    Pass task_id to enable live transcript preview: each chunk's transcript is
    appended to partial_transcript as soon as the API responds.

    Requires settings.openai_api_key to be set.
    Returns (transcript_text, language).
    """
    from app import state as _state  # local import avoids circular at module level

    if not settings.openai_api_key:
        raise RuntimeError(
            "OpenAI API key not configured. "
            "Set OPENAI_API_KEY in your .env file to use this mode."
        )

    loop = asyncio.get_running_loop()

    logger.info(f"[OpenAI Whisper] Preprocessing audio: {audio_path}")
    chunks = await loop.run_in_executor(None, audio_preprocessor.preprocess, audio_path)
    logger.info(f"[OpenAI Whisper] Sending {len(chunks)} chunk(s) to API...")

    try:
        transcripts: list[str] = []
        async with httpx.AsyncClient(timeout=300) as client:
            for i, chunk_path in enumerate(chunks, start=1):
                logger.info(f"[OpenAI Whisper] Chunk {i}/{len(chunks)}: {chunk_path}")
                text = await _call_whisper_api(client, chunk_path, language)
                if text.strip():
                    transcripts.append(text.strip())
                    # Stream each completed chunk to the live preview
                    if task_id is not None:
                        await _state.append_partial_transcript(task_id, text.strip() + " ")

        transcript = " ".join(transcripts)
        logger.info(f"[OpenAI Whisper] Done: {len(transcript):,} chars")
        return transcript, language

    finally:
        audio_preprocessor.cleanup_chunks(chunks)


async def _call_whisper_api(client: httpx.AsyncClient, chunk_path: str, language: str) -> str:
    """Send a single audio chunk to the OpenAI Whisper API and return the transcript text."""
    lang_param = language if language != "auto" else None

    with open(chunk_path, "rb") as f:
        data = {"model": "whisper-1", "response_format": "text"}
        if lang_param:
            data["language"] = lang_param

        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            files={"file": (Path(chunk_path).name, f, "audio/mpeg")},
            data=data,
        )

    response.raise_for_status()
    return response.text


# ── ivrit-ai transcription ────────────────────────────────────────────────────
# Uses ivrit-ai's Hebrew-tuned Whisper checkpoint via faster-whisper (CT2 backend).
# Model path is settings.ivrit_ai_model (default: ivrit-ai/whisper-large-v3-turbo-ct2).
# Shares the same cache dir as WHISPER_LOCAL — mounted as a Docker volume so the
# ~GB download survives container restarts.

def _load_ivrit_model_sync():
    """Blocking: load the ivrit-ai CT2 model. Runs in a thread executor."""
    from faster_whisper import WhisperModel

    cache_dir = Path.home() / ".cache" / "faster_whisper"
    model_repo = settings.ivrit_ai_model
    logger.info(
        f"Loading ivrit-ai model '{model_repo}' "
        f"on {settings.whisper_device} ({settings.whisper_compute_type})..."
    )
    model = WhisperModel(
        model_repo,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=str(cache_dir),
    )
    logger.info("✅ ivrit-ai model loaded")
    return model


async def _get_ivrit_model():
    """Return the loaded ivrit-ai model, loading it if necessary (async, thread-safe)."""
    global _ivrit_model, _ivrit_last_used
    async with _ivrit_lock:
        if _ivrit_model is None:
            loop = asyncio.get_running_loop()
            _ivrit_model = await loop.run_in_executor(None, _load_ivrit_model_sync)
        _ivrit_last_used = time.time()
    return _ivrit_model


async def transcribe_ivrit_ai(
    audio_path: str,
    language: str = "he",
    task_id: str | None = None,
) -> tuple[str, str]:
    """
    Transcribe with ivrit-ai's Hebrew-tuned Whisper model.

    Output format is identical to transcribe() (plain-text concatenation of
    segments, same segment_cb streaming contract) — this is the contract the
    live-preview panel and Feature 7's timestamp-click flow rely on.
    """
    from app import state as _state  # local import avoids circular at module level

    model = await _get_ivrit_model()
    loop  = asyncio.get_running_loop()

    if task_id is not None:
        def segment_cb(text: str, _tid=task_id, _loop=loop) -> None:
            future = asyncio.run_coroutine_threadsafe(
                _state.append_partial_transcript(_tid, text),
                _loop,
            )
            future.add_done_callback(
                lambda f: f.exception() and logger.warning(
                    "partial transcript write failed for %s: %s", _tid, f.exception()
                )
            )
    else:
        segment_cb = None

    logger.info(f"[ivrit-ai] Transcribing: {audio_path} (language: {language})")
    # _transcribe_sync is reused — ivrit-ai speaks the same faster-whisper API,
    # so timestamps, VAD behavior, and segment batching are byte-identical.
    transcript, detected_lang = await loop.run_in_executor(
        None, _transcribe_sync, model, audio_path, language, segment_cb
    )

    global _ivrit_last_used
    _ivrit_last_used = time.time()

    logger.info(
        f"[ivrit-ai] Done: {len(transcript):,} chars, "
        f"detected language: {detected_lang}"
    )
    return transcript, detected_lang
