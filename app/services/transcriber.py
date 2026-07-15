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

Model lifecycle:
  A single _ModelSlot holds at most ONE Whisper variant (vanilla or ivrit-ai)
  in RAM at a time. The slot refcounts active transcriptions so the idle
  watcher can never evict a model that is mid-transcription, and switching
  variants evicts the old model before loading the new one — worst-case
  resident memory is exactly one model.
"""
import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from app import cancellation
from app import state as _state
from app.config import settings
from app.models import TaskStatus
from app.services import audio_preprocessor
from app.services.errors import PipelineError, TaskCancelled

logger = logging.getLogger(__name__)

_IDLE_THRESHOLD = settings.auto_shutdown_idle_minutes * 60

# Dedicated 1-thread pool for model loading + transcription. Serializes the
# CPU-bound Whisper work and keeps it from starving the default executor
# used by downloads, DB writes, and Gemini uploads.
_whisper_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


# ── Model lifecycle ───────────────────────────────────────────────────────────

class _ModelSlot:
    """At most one Whisper variant in RAM; never evicted while in use."""

    def __init__(self):
        self._model = None
        self._key: str | None = None
        self._active = 0
        self._last_used = 0.0
        self._cond = asyncio.Condition()

    async def acquire(self, key: str, loader):
        """Return the model for `key`, loading it (and evicting any other
        variant) if needed. Callers MUST pair with release()."""
        async with self._cond:
            # A different variant is mid-transcription — wait for it to finish
            # rather than pulling its model out from under it.
            while self._model is not None and self._key != key and self._active > 0:
                await self._cond.wait()
            if self._key != key:
                self._evict()
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(_whisper_pool, loader)
                self._key = key
            self._active += 1
            self._last_used = time.time()
            return self._model

    async def release(self):
        async with self._cond:
            self._active = max(0, self._active - 1)
            self._last_used = time.time()
            self._cond.notify_all()

    async def unload_if_idle(self, threshold_s: float):
        async with self._cond:
            if self._model is None or self._active > 0:
                return
            idle = time.time() - self._last_used
            if idle > threshold_s:
                logger.info(
                    f"Model '{self._key}' idle for {idle / 60:.1f} min "
                    f"(threshold: {threshold_s / 60:.0f} min) — unloading"
                )
                self._evict()

    def _evict(self):
        if self._model is not None:
            logger.info(f"Unloading model '{self._key}' from RAM")
        self._model = None
        self._key = None
        gc.collect()


_slot = _ModelSlot()


def _resolve_device() -> tuple[str, str]:
    """
    Resolve (device, compute_type) from settings, honoring "auto".

    CUDA detection goes through ctranslate2 (faster-whisper's runtime) rather
    than torch — torch is deliberately not installed (see Dockerfile). When
    the container has no GPU (or the CUDA libs are missing) this silently
    falls back to CPU/int8, so the same image runs anywhere.
    """
    device = settings.whisper_device.lower()
    compute = settings.whisper_compute_type.lower()

    if device == "auto":
        cuda_available = False
        try:
            import ctranslate2
            cuda_available = ctranslate2.get_cuda_device_count() > 0
        except Exception as exc:
            logger.info(f"CUDA probe failed ({exc}) — using CPU")
        device = "cuda" if cuda_available else "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute


def _load_whisper(model_id: str, label: str):
    """Blocking: load a faster-whisper model. Runs in the whisper pool."""
    from faster_whisper import WhisperModel

    device, compute = _resolve_device()
    logger.info(f"Loading {label} model '{model_id}' on {device} ({compute})...")
    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute,
        download_root=str(settings.whisper_cache_dir),
    )
    logger.info(f"✅ {label} model loaded on {device}")
    return model


def _load_model_sync():
    return _load_whisper(settings.whisper_model, "Whisper")


def _load_ivrit_model_sync():
    return _load_whisper(settings.ivrit_ai_model, "ivrit-ai")


async def unload_model_if_idle():
    """Called every 60s by the idle watcher in main.py. Frees RAM by evicting
    the loaded model once it has been idle past the threshold. A model with an
    active transcription is never evicted."""
    await _slot.unload_if_idle(_IDLE_THRESHOLD)


async def drain(timeout: float = 30.0) -> bool:
    """Wait until the whisper thread pool has no in-flight work.

    Called during shutdown AFTER worker.stop() flagged in-flight tasks for
    cancellation and BEFORE state.close_db(): the pool has one FIFO worker, so
    a no-op submitted now completes only once the running transcription has
    unwound past its next per-segment cancel checkpoint — guaranteeing no
    thread-side run_coroutine_threadsafe callback can land on a closed DB.

    Deliberately drains rather than shutting the pool down: the executor is
    module-level and tests start/stop the app repeatedly in one process.
    Returns False if the pool didn't quiesce within `timeout` (wedged decode);
    shutdown proceeds anyway and Docker's stop grace period is the backstop.
    """
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(_whisper_pool, lambda: None), timeout
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(
            f"Whisper pool did not drain within {timeout:.0f}s — continuing shutdown"
        )
        return False
    except RuntimeError:
        # Pool already shut down (interpreter teardown) — nothing to wait for.
        return True


# ── LOCAL transcription ───────────────────────────────────────────────────────

def _transcribe_sync(
    model,
    audio_path: str,
    language: str,
    segment_cb=None,
    progress_cb=None,
    cancel_cb=None,
) -> tuple[str, str]:
    """
    Blocking transcription — runs in the whisper pool.
    Returns (full_transcript_text, detected_language_code).

    segment_cb: optional sync callable(text: str) called every ~10 segments or
    ~5 seconds so callers can stream live text to the DB for the preview panel.

    progress_cb: optional sync callable(fraction: float) called every ~5 seconds
    with how far into the recording the transcription has reached (0.0–1.0).
    Transcription is by far the longest pipeline step — without this the task
    progress bar sits frozen for the entire run.

    cancel_cb: optional sync callable() -> bool, polled once per segment. When
    it returns True we raise TaskCancelled immediately — the generator stops
    pulling audio, the model slot is released by the caller's finally, and the
    GPU is freed. A lecture is thousands of ~5s segments, so the check is both
    cheap (a set lookup) and responsive (abort within one segment).
    """
    lang_hint = language if language != "auto" else None

    segments, info = model.transcribe(
        audio_path,
        language=lang_hint,
        # ── Accuracy ──
        beam_size=settings.whisper_beam_size,
        best_of=settings.whisper_best_of,
        # Temperature ladder: greedy/beam first; only when a window fails the
        # quality thresholds below does decoding retry at higher temperatures.
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # ── Anti-hallucination ──
        # A window is rejected (and retried hotter) when its output is too
        # repetitive (gzip ratio) or too improbable — the two signatures of
        # Whisper inventing text over noise/music.
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        # Decoding each window independently prevents one bad window from
        # poisoning the rest of the lecture with repetition loops.
        condition_on_previous_text=settings.whisper_condition_on_previous_text,
        # Domain vocabulary hint (names, technical terms) — biases spelling.
        initial_prompt=settings.whisper_initial_prompt or None,
        # ── VAD: skip silence entirely instead of transcribing it ──
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": settings.whisper_vad_min_silence_ms,
            "speech_pad_ms": settings.whisper_vad_speech_pad_ms,
        },
        word_timestamps=False,                  # Saves memory
    )
    total_s = float(getattr(info, "duration", 0) or 0)

    full_texts: list[str] = []
    buffer: list[str] = []
    last_flush = time.time()
    last_progress = time.time()

    for seg in segments:
        # Cooperative cancellation checkpoint. Checked before doing any work on
        # the segment so an abort takes effect at the very next window. Flush
        # whatever partial text we already have so the live preview isn't lost.
        if cancel_cb is not None and cancel_cb():
            if segment_cb is not None and buffer:
                segment_cb(" ".join(buffer) + " ")
            raise TaskCancelled()

        text = seg.text.strip()
        if not text:
            continue
        # Feature 7: prepend [MM:SS] from seg.start so downstream prompts can
        # reference timestamps and the UI can linkify them into seek anchors.
        start = int(getattr(seg, "start", 0) or 0)
        mm, ss = divmod(start, 60)
        tagged = f"[{mm:02d}:{ss:02d}] {text}"
        full_texts.append(tagged)

        if segment_cb is not None:
            buffer.append(tagged)
            now = time.time()
            # Flush every 10 segments or every 5 seconds to avoid DB overload
            if len(buffer) >= 10 or now - last_flush >= 5.0:
                segment_cb(" ".join(buffer) + " ")
                buffer = []
                last_flush = now

        if progress_cb is not None and total_s > 0:
            now = time.time()
            if now - last_progress >= 5.0:
                progress_cb(min(float(getattr(seg, "end", 0) or 0) / total_s, 1.0))
                last_progress = now

    # Final flush — make sure nothing is left in the buffer
    if segment_cb is not None and buffer:
        segment_cb(" ".join(buffer) + " ")

    return " ".join(full_texts), info.language


def _make_cancel_cb(task_id: str | None):
    """
    Return a sync predicate the transcription thread polls to learn whether the
    task has been cancelled, or None when no task_id is given. Pure in-memory
    (app.cancellation) — no event-loop bridging, so it is safe and cheap to call
    from the worker thread on every segment.
    """
    if task_id is None:
        return None
    return lambda _tid=task_id: cancellation.is_cancelled(_tid)


def _make_segment_cb(task_id: str | None, loop: asyncio.AbstractEventLoop):
    """
    Build a thread-safe segment callback that fires-and-forgets partial
    transcript writes into the event loop, or None when no task_id is given.
    Errors from append_partial_transcript are logged via a done-callback so
    they don't silently disappear inside the Future.
    """
    if task_id is None:
        return None

    def segment_cb(text: str, _tid=task_id, _loop=loop) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(
                _state.append_partial_transcript(_tid, text),
                _loop,
            )
        except RuntimeError:
            # Loop closed mid-shutdown — nowhere to persist to; the restart
            # recovery re-runs this task anyway.
            return
        future.add_done_callback(
            lambda f: f.cancelled() or (f.exception() and logger.warning(
                "partial transcript write failed for %s: %s", _tid, f.exception()
            ))
        )

    return segment_cb


# Transcription owns the 50→78 slice of the progress bar; the summarizer
# picks up at 80 (see processor.py milestones), so 78 is the safe ceiling.
_PROGRESS_FLOOR = 50
_PROGRESS_CEIL  = 78


def _make_progress_cb(task_id: str | None, loop: asyncio.AbstractEventLoop):
    """
    Build a thread-safe progress callback mapping audio position (0.0–1.0)
    into the task's progress column, same fire-and-forget pattern as
    _make_segment_cb. Skips writes that wouldn't change the integer percent.
    """
    if task_id is None:
        return None

    last_pct = _PROGRESS_FLOOR

    def progress_cb(fraction: float, _tid=task_id, _loop=loop) -> None:
        nonlocal last_pct
        pct = _PROGRESS_FLOOR + int(fraction * (_PROGRESS_CEIL - _PROGRESS_FLOOR))
        if pct <= last_pct:
            return
        last_pct = pct
        message = f"🎙️ מתמלל... {int(fraction * 100)}% מההקלטה"
        try:
            future = asyncio.run_coroutine_threadsafe(
                _state.update_task(_tid, TaskStatus.TRANSCRIBING, pct, message),
                _loop,
            )
        except RuntimeError:
            return  # loop closed mid-shutdown — see segment_cb
        future.add_done_callback(
            lambda f: f.cancelled() or (f.exception() and logger.warning(
                "progress update failed for %s: %s", _tid, f.exception()
            ))
        )

    return progress_cb


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
    model = await _slot.acquire(f"whisper:{settings.whisper_model}", _load_model_sync)
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"[Local Whisper] Transcribing: {audio_path} (language: {language})")
        transcript, detected_lang = await loop.run_in_executor(
            _whisper_pool, _transcribe_sync, model, audio_path, language,
            _make_segment_cb(task_id, loop), _make_progress_cb(task_id, loop),
            _make_cancel_cb(task_id),
        )
    finally:
        # Runs on cancellation too: the slot is released, its refcount drops,
        # and the idle watcher can evict the model to reclaim VRAM.
        await _slot.release()

    logger.info(
        f"[Local Whisper] Done: {len(transcript):,} chars, "
        f"detected language: {detected_lang}"
    )
    return transcript, detected_lang


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
    model = await _slot.acquire(f"ivrit:{settings.ivrit_ai_model}", _load_ivrit_model_sync)
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"[ivrit-ai] Transcribing: {audio_path} (language: {language})")
        # _transcribe_sync is reused — ivrit-ai speaks the same faster-whisper API,
        # so timestamps, VAD behavior, and segment batching are byte-identical.
        transcript, detected_lang = await loop.run_in_executor(
            _whisper_pool, _transcribe_sync, model, audio_path, language,
            _make_segment_cb(task_id, loop), _make_progress_cb(task_id, loop),
            _make_cancel_cb(task_id),
        )
    finally:
        await _slot.release()

    logger.info(
        f"[ivrit-ai] Done: {len(transcript):,} chars, "
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
    if not settings.openai_api_key:
        raise PipelineError(
            "מפתח OpenAI לא מוגדר בשרת — בחר מצב עיבוד אחר או הוסף OPENAI_API_KEY",
            detail="OPENAI_API_KEY not configured; WHISPER_API mode unavailable",
        )

    loop = asyncio.get_running_loop()

    logger.info(f"[OpenAI Whisper] Preprocessing audio: {audio_path}")
    chunks = await loop.run_in_executor(None, audio_preprocessor.preprocess, audio_path)
    logger.info(f"[OpenAI Whisper] Sending {len(chunks)} chunk(s) to API...")

    try:
        transcripts: list[str] = []
        async with httpx.AsyncClient(timeout=300) as client:
            for i, chunk_path in enumerate(chunks, start=1):
                # Cooperative cancellation between chunks — the finally below
                # still cleans up the temp chunk files on the way out.
                if task_id is not None and cancellation.is_cancelled(task_id):
                    raise TaskCancelled(task_id)
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
