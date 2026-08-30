"""
Pipeline orchestrator.

Two entry points (invoked by the worker queue — app/services/worker.py):
  run_pipeline()           — URL → download → process → save result
  run_pipeline_from_file() — uploaded file → process → save result

Each step updates the task's progress in SQLite so the frontend can display
a live progress bar while polling GET /api/tasks/{id}.

Error handling: everything user-facing raises PipelineError (or a subclass
like ZoomDownloadError) with a ready Hebrew message; the technical cause is
stored separately in tasks.error_detail so failures stay debuggable.

Progress milestones (GEMINI_DIRECT):
  5%  → Downloading
  40% → Download complete
  50% → Sending to Gemini
  55% → Audio uploaded
  65% → Gemini processed file
  72% → Generating content
  100%→ Complete

Progress milestones (WHISPER paths):
  5%  → Downloading
  40% → Download complete
  50% → Transcribing
  80% → Summarizing (initial Gemini generation)
  82–88% → Per-chunk progress (long transcripts only)
  88% → Merging chunks / Critique running
  95% → Revising low-quality questions (if needed)
  100%→ Complete

TRANSCRIPTION_ONLY jumps 50% → 100% (no Gemini step at all).
"""
import logging
import re
import shutil
from pathlib import Path

from app import cancellation, state
from app.config import settings
from app.models import LessonResult, ProcessingMode, TaskStatus
from app.services import runtime_config
from app.services import audio_preprocessor, summarizer, transcriber, zoom_downloader
from app.services.errors import PipelineError, TaskCancelled

logger = logging.getLogger(__name__)


def _make_progress_cb(task_id: str, status: TaskStatus):
    """Async progress callback passed into the summarizer — a direct await,
    no thread bridging (the summarizer is fully async)."""
    async def cb(progress: int, message: str) -> None:
        await state.update_task(task_id, status, progress, message)
    return cb


async def _fail(task_id: str, exc: Exception):
    """Store a failure: user message in error, technical cause in error_detail."""
    if isinstance(exc, PipelineError):
        await state.fail_task(task_id, exc.user_message, detail=exc.detail or repr(exc))
    else:
        await state.fail_task(
            task_id, "שגיאה בלתי צפויה בעיבוד — נסה שוב", detail=repr(exc)
        )


def _raise_if_cancelled(task_id: str) -> None:
    """Cooperative cancellation checkpoint between major pipeline steps.

    Mid-transcription aborts are handled inside the transcriber (per segment);
    this covers the gaps — cancelled-while-queued, and cancelled during the
    long download/summarize awaits — so the work stops at the next boundary
    instead of running to completion after the user hit cancel.
    """
    if cancellation.is_cancelled(task_id):
        raise TaskCancelled(task_id)


async def _get_resume_offset(task_id: str) -> float:
    """Return a start-time offset (seconds) derived from the task's partial transcript.

    Scans for the last ``[MM:SS]`` or ``[HH:MM:SS]`` timestamp; converts it to
    total seconds and returns it, or ``0.0`` when no timestamps exist.
    """
    text, _ = await state.get_partial_transcript(task_id)
    if not text:
        return 0.0
    # Match the last [MM:SS] or [HH:MM:SS] in the transcript
    matches = re.findall(r"\[(\d{1,2}:\d{2}(:\d{2})?)\]", text)
    if not matches:
        return 0.0
    _, last_ts = matches[-1]
    parts = last_ts.split(":")
    if len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:                  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])


# ── Entry points ──────────────────────────────────────────────────────────────────

async def run_pipeline(
    task_id: str,
    url: str,
    mode: ProcessingMode,
    cookies: str | None,
    language: str,
):
    """Full pipeline starting from a Zoom URL."""
    audio_path: str | None = None
    try:
        _raise_if_cancelled(task_id)
        await state.update_task(
            task_id, TaskStatus.DOWNLOADING, 5, "⬇️ מוריד את ההקלטה מ-Zoom..."
        )
        audio_path = await zoom_downloader.download_audio(
            url=url,
            task_id=task_id,
            cookies_netscape=cookies,
            # GEMINI_DIRECT sends the raw file to Gemini Files API which accepts
            # M4A/MP4 natively — skipping ffmpeg re-encode saves 15-20 min on
            # a shared Fly.io CPU.
            extract_to_mp3=(mode != ProcessingMode.GEMINI_DIRECT),
        )
        await state.update_task(
            task_id, TaskStatus.DOWNLOADING, 40, "✅ ההורדה הושלמה. מעבד אודיו..."
        )
        if mode != ProcessingMode.GEMINI_DIRECT:
            await state.update_task(
                task_id, TaskStatus.TRANSCRIBING, 45, "🎞️ מכין את האודיו לעיבוד..."
            )

        _raise_if_cancelled(task_id)
        result = await _process_audio(task_id, audio_path, mode, language)
        result = await _generate_flashcards_step(task_id, result)
        # Move the audio into a persistent per-task location so the UI player
        # can stream it back. Replaces the old "cleanup in finally" pattern.
        audio_path = await _persist_audio_for_task(task_id, audio_path)
        if await state.complete_task(task_id, result):
            logger.info(f"Task {task_id} completed ✅")
        else:
            # A cancel landed in the pipeline's final moments — the terminal
            # CANCELLED row is sticky, so the result is discarded.
            logger.info(f"Task {task_id} was cancelled at the finish line — result discarded")

    except TaskCancelled:
        # Not a failure — status is already CANCELLED (set by the cancel
        # endpoint). Drop the temp download: a URL retry re-downloads from
        # scratch, so keeping it would only orphan bytes on disk.
        logger.info(f"Task {task_id} cancelled — cleaning up download")
        cancellation.clear(task_id)
        if audio_path:
            await zoom_downloader.cleanup_audio(audio_path)
    except Exception as exc:
        if isinstance(exc, PipelineError):
            logger.error(f"Task {task_id} failed: {exc.user_message} ({exc.detail})")
        else:
            logger.exception(f"Task {task_id} — unexpected error")
        await _fail(task_id, exc)
        if audio_path:
            await zoom_downloader.cleanup_audio(audio_path)


async def run_pipeline_from_file(
    task_id: str,
    file_path: str,
    mode: ProcessingMode,
    language: str,
):
    """Pipeline starting from an already-saved uploaded file."""
    try:
        _raise_if_cancelled(task_id)
        await state.update_task(
            task_id, TaskStatus.TRANSCRIBING, 10, "📁 קובץ התקבל. מתחיל עיבוד..."
        )
        # Mirror the URL path's extract_to_mp3 rule: GEMINI_DIRECT sends the
        # native container (M4A/MP4) straight to the Gemini Files API, so
        # extraction is wasted CPU there. Every other mode transcribes the
        # audio locally/via Whisper — swap the heavy upload (e.g. a 1 GB
        # lecture .mp4) for a lean MP3 and free the disk immediately.
        if mode != ProcessingMode.GEMINI_DIRECT:
            await state.update_task(
                task_id, TaskStatus.TRANSCRIBING, 12, "🎞️ מחלץ אודיו מהקובץ שהועלה..."
            )
            extracted = await audio_preprocessor.extract_audio_track(file_path)
            if extracted != file_path:
                file_path = extracted
                # Extraction deleted the original upload, but the durable job
                # payload still points at it — a restart mid-transcription would
                # see a missing file and wrongly fail the resume. Repoint it.
                payload = await state.get_job_payload(task_id)
                if payload:
                    payload["file_path"] = file_path
                    await state.set_job_payload(task_id, payload)
        _raise_if_cancelled(task_id)
        result = await _process_audio(task_id, file_path, mode, language)
        result = await _generate_flashcards_step(task_id, result)
        file_path = await _persist_audio_for_task(task_id, file_path)
        if await state.complete_task(task_id, result):
            logger.info(f"Task {task_id} (upload) completed ✅")
        else:
            logger.info(f"Task {task_id} (upload) was cancelled at the finish line — result discarded")

    except TaskCancelled:
        # Keep the uploaded source file on disk so POST /tasks/{id}/retry can
        # replay it without a re-upload. The retention sweep reclaims it after
        # settings.media_retention_days if the user never retries.
        logger.info(f"Task {task_id} (upload) cancelled — keeping source for retry")
        cancellation.clear(task_id)
    except Exception as exc:
        if isinstance(exc, PipelineError):
            logger.error(f"Task {task_id} (upload) failed: {exc.user_message} ({exc.detail})")
        else:
            logger.exception(f"Task {task_id} (upload) — unexpected error")
        await _fail(task_id, exc)
        # NOTE: the uploaded file is deliberately NOT deleted here — a failure
        # is often transient (Gemini 5xx, network) and retry reuses this exact
        # file. Retention (app/main.py) reclaims it later if never retried.


# ── Core processing logic ─────────────────────────────────────────────────────────

async def _process_audio(
    task_id: str,
    audio_path: str,
    mode: ProcessingMode,
    language: str,
) -> LessonResult:
    """
    Transcribe and/or summarize the audio depending on the selected mode.
    Returns a populated LessonResult.

    Local-only pipeline (default):
      WHISPER_LOCAL  → Faster-Whisper transcript + Ollama summary/exam
      WHISPER_API     → OpenAI whisper transcript + Ollama summary/exam
      IVRIT_AI        → ivrit-ai transcript + Ollama summary/exam

    Cloud paths (DISABLED — safely bypassed for local-first operation):
      GEMINI_DIRECT   → raw audio uploaded to Gemini Files API  [bypassed]
      TRANSCRIPTION_ONLY → transcript only, no LLM call         [still works]
    """
    if mode == ProcessingMode.GEMINI_DIRECT:
        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            50,
            "🎧 שולח את האודיו ישירות ל-Gemini Files API...",
        )
        result = await summarizer.summarize_audio(
            audio_path,
            _make_progress_cb(task_id, TaskStatus.SUMMARIZING),
            language=language,
        )
        return result

    payload = await state.get_job_payload(task_id)
    if payload is None:
        payload = {}

    # ── Resume from cached full transcript ────────────────────────────────
    final_transcript = payload.get("final_transcript")
    detected_lang = payload.get("cached_language", language)

    if final_transcript:
        logger.info(f"Task {task_id}: Resuming from cached full transcript ({len(final_transcript)} chars)")
        transcript = final_transcript
        # Skip transcription entirely; proceed to summarization below.
    elif detected_lang is None or detected_lang == "":
        detected_lang = language

    if not final_transcript:
        # ── Compute resume offset from partial transcript ───────────────
        resume_offset = await _get_resume_offset(task_id)

        if mode == ProcessingMode.TRANSCRIPTION_ONLY:
            await state.update_task(
                task_id,
                TaskStatus.TRANSCRIBING,
                50,
                "📝 מתמלל בלבד — ללא סיכום ומבחן...",
            )
            transcript, detected_lang = await transcriber.transcribe_ivrit_ai(
                audio_path, language, task_id=task_id
            )
            if resume_offset > 0:
                partial_text, _ = await state.get_partial_transcript(task_id)
                if partial_text:
                    # Strip timestamps from the partial so we can concat cleanly
                    clean_partial = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", partial_text)
                    transcript = clean_partial + " " + transcript

            payload["cached_transcript"] = transcript
            payload["cached_language"] = detected_lang
            await state.set_job_payload(task_id, payload)
            return LessonResult(transcript=transcript, language=detected_lang or "he")

        # ── Whisper transcription ───────────────────────────────────────
        if mode == ProcessingMode.WHISPER_API:
            await state.update_task(
                task_id,
                TaskStatus.TRANSCRIBING,
                50,
                "☁️ מסיר שקט ושולח ל-OpenAI Whisper API...",
            )
            transcript, detected_lang = await transcriber.transcribe_via_api(
                audio_path, language, task_id=task_id, resume_offset=resume_offset
            )

        elif mode == ProcessingMode.IVRIT_AI:
            await state.update_task(
                task_id,
                TaskStatus.TRANSCRIBING,
                50,
                "🇮🇱 מתמלל עם ivrit-ai (מודל מותאם לעברית)...",
            )
            transcript, detected_lang = await transcriber.transcribe_ivrit_ai(
                audio_path, language, task_id=task_id, resume_offset=resume_offset
            )

        else:
            # Default / WHISPER_LOCAL — the primary local path
            await state.update_task(
                task_id,
                TaskStatus.TRANSCRIBING,
                50,
                "🎙️ מתמלל עם Whisper מקומי (עשוי לקחת מספר דקות)...",
            )
            transcript, detected_lang = await transcriber.transcribe(
                audio_path, language, task_id=task_id, resume_offset=resume_offset
            )

        # ── Save full transcript as checkpoint BEFORE summarization ───────
        if resume_offset > 0:
            partial_text, _ = await state.get_partial_transcript(task_id)
            if partial_text:
                clean_partial = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", partial_text)
                transcript = clean_partial + " " + transcript

        payload["cached_transcript"] = transcript
        payload["final_transcript"] = transcript
        payload["cached_language"] = detected_lang
        await state.set_job_payload(task_id, payload)

    # ── Ollama summarization (local LLM) ────────────────────────────────────
    if not transcript:
        raise PipelineError(
            "⚠️ התמלול נכשל — לא קיבלתי תוצאה מה-Mודל",
            detail="transcription returned None transcript for mode={mode}",
        )

    await state.update_task(
        task_id,
        TaskStatus.SUMMARIZING,
        80,
        "🤖 יוצר סיכום ומבחן עם Ollama מקומי...",
    )
    result = await summarizer.summarize_transcript_with_ollama(
        transcript,
        ollama_host=settings.ollama_host,
        model=(await runtime_config.get(runtime_config.KEY_OLLAMA_MODEL))
            or settings.ollama_model,
        timeout=settings.ollama_timeout,
    )
    result.transcript = transcript
    return result


# ── Audio persistence (Feature 7) ─────────────────────────────────────────────────

def _audio_root() -> Path:
    """
    Persistent root for per-task audio files. Mounted as a Docker volume so the
    files survive container restarts. Kept separate from downloads/ so we can
    wipe downloads/ without touching stored playback audio.
    """
    root = settings.data_dir / "audio"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


async def _persist_audio_for_task(task_id: str, src_path: str | None) -> str | None:
    """
    Move the temp audio to {data_dir}/audio/{task_id}{ext} and update the DB.
    Returns the new path (or the original if the move failed).
    """
    if not src_path:
        return None
    src = Path(src_path)
    if not src.exists():
        return None
    dest = _audio_root() / f"{task_id}{src.suffix or '.mp3'}"
    try:
        # Move is a rename when on same volume — cheap and atomic
        shutil.move(str(src), str(dest))
    except Exception as exc:
        # Cross-device or permission issue: fall back to copy + remove-source
        logger.warning(f"audio move failed ({exc}); falling back to copy")
        try:
            shutil.copy2(str(src), str(dest))
            src.unlink(missing_ok=True)
        except Exception as copy_exc:
            logger.error(f"audio persist failed for {task_id}: {copy_exc}")
            return str(src)
    await state.set_audio_path(task_id, str(dest))
    logger.info(f"Task {task_id}: audio persisted at {dest}")
    return str(dest)


# ── Flashcards step ───────────────────────────────────────────────────────────────

async def _generate_flashcards_step(task_id: str, result) -> "LessonResult":
    """
    Append a flashcards-generation pass to a completed result.
    One extra Gemini call — adds ~30% to per-task cost, so failures are
    non-fatal (we log and return the result without flashcards rather than
    failing the whole task).
    """
    if not result or not result.summary:
        return result
    await state.update_task(
        task_id,
        TaskStatus.SUMMARIZING,
        98,
        "🃏 מייצר כרטיסיות לחזרה...",
    )
    try:
        cards = await summarizer.generate_flashcards(result.summary, result.transcript, language=result.language)
        result.flashcards = cards
        logger.info(f"Task {task_id}: generated {len(cards)} flashcards")
    except Exception as exc:
        # Soft-fail: the lesson itself is already complete. Log and continue.
        logger.warning(f"Task {task_id}: flashcards generation failed: {exc}")
    return result
