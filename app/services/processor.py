"""
Pipeline orchestrator.

Two entry points:
  run_pipeline()           — URL → download → process → save result
  run_pipeline_from_file() — uploaded file → process → save result

Each step updates the task's progress in SQLite so the frontend can display
a live progress bar while polling GET /api/tasks/{id}.

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
"""
import asyncio
import logging
import shutil
from pathlib import Path

from app import state
from app.config import settings
from app.models import LessonResult, ProcessingMode, TaskStatus
from app.services import summarizer, transcriber, zoom_downloader

logger = logging.getLogger(__name__)


# ── Error helpers ─────────────────────────────────────────────────────────────────

def _user_friendly_error(exc: Exception) -> str:
    """Map exception types to Hebrew user-facing error strings."""
    msg = str(exc)
    low = msg.lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "10 דקות" in msg:
        return "⏱️ הפעולה לקחה יותר מדי זמן ופסקה — נסה שוב"
    if "quota" in low or "429" in low or "מכסת" in msg or "rate limit" in low:
        return "⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות"
    if "json" in low or "JSON" in msg or "malformed" in low:
        return "🔄 Gemini החזיר תוצאה לא תקינה — נסה שוב"
    if isinstance(exc, zoom_downloader.ZoomDownloadError):
        return str(exc)
    return f"שגיאה: {msg[:300]}"


def _make_progress_cb(
    task_id: str,
    status: TaskStatus,
    loop: asyncio.AbstractEventLoop,
):
    """
    Return a thread-safe sync callable that fires-and-forgets a progress update.
    Used to pass into the sync summarizer functions running in a threadpool executor.
    """
    def cb(progress: int, message: str) -> None:
        asyncio.run_coroutine_threadsafe(
            state.update_task(task_id, status, progress, message),
            loop,
        )
    return cb


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

        result = await _process_audio(task_id, audio_path, mode, language)
        result = await _generate_flashcards_step(task_id, result)
        # Move the audio into a persistent per-task location so the UI player
        # can stream it back. Replaces the old "cleanup in finally" pattern.
        audio_path = await _persist_audio_for_task(task_id, audio_path)
        await state.complete_task(task_id, result)
        logger.info(f"Task {task_id} completed ✅")

    except zoom_downloader.ZoomDownloadError as exc:
        logger.error(f"Task {task_id} — download error: {exc}")
        await state.fail_task(task_id, _user_friendly_error(exc))
        if audio_path:
            await zoom_downloader.cleanup_audio(audio_path)

    except Exception as exc:
        logger.exception(f"Task {task_id} — unexpected error")
        await state.fail_task(task_id, _user_friendly_error(exc))
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
        await state.update_task(
            task_id, TaskStatus.TRANSCRIBING, 10, "📁 קובץ התקבל. מתחיל עיבוד..."
        )
        result = await _process_audio(task_id, file_path, mode, language)
        result = await _generate_flashcards_step(task_id, result)
        file_path = await _persist_audio_for_task(task_id, file_path)
        await state.complete_task(task_id, result)
        logger.info(f"Task {task_id} (upload) completed ✅")

    except Exception as exc:
        logger.exception(f"Task {task_id} (upload) — unexpected error")
        await state.fail_task(task_id, _user_friendly_error(exc))
        await zoom_downloader.cleanup_audio(file_path)


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
    """
    loop = asyncio.get_running_loop()

    if mode == ProcessingMode.GEMINI_DIRECT:
        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            50,
            "🤖 שולח אודיו ל-Gemini AI — מייצר סיכום ומבחן...",
        )
        progress_cb = _make_progress_cb(task_id, TaskStatus.SUMMARIZING, loop)
        result = await summarizer.summarize_audio(audio_path, progress_cb)

    elif mode == ProcessingMode.WHISPER_API:
        await state.update_task(
            task_id,
            TaskStatus.TRANSCRIBING,
            50,
            "☁️ מסיר שקט ושולח ל-OpenAI Whisper API...",
        )
        transcript, _ = await transcriber.transcribe_via_api(audio_path, language, task_id=task_id)

        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            80,
            "🤖 יוצר סיכום ומבחן עם Gemini AI...",
        )
        progress_cb = _make_progress_cb(task_id, TaskStatus.SUMMARIZING, loop)
        result = await summarizer.summarize_transcript(transcript, progress_cb)
        result.transcript = transcript

    elif mode == ProcessingMode.IVRIT_AI:
        await state.update_task(
            task_id,
            TaskStatus.TRANSCRIBING,
            50,
            "🇮🇱 מתמלל עם ivrit-ai (מודל מותאם לעברית)...",
        )
        transcript, _ = await transcriber.transcribe_ivrit_ai(audio_path, language, task_id=task_id)

        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            80,
            "🤖 יוצר סיכום ומבחן עם Gemini AI...",
        )
        progress_cb = _make_progress_cb(task_id, TaskStatus.SUMMARIZING, loop)
        result = await summarizer.summarize_transcript(transcript, progress_cb)
        result.transcript = transcript

    else:
        await state.update_task(
            task_id,
            TaskStatus.TRANSCRIBING,
            50,
            "🎙️ מתמלל עם Whisper מקומי (עשוי לקחת מספר דקות)...",
        )
        transcript, detected_lang = await transcriber.transcribe(audio_path, language, task_id=task_id)

        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            80,
            "🤖 יוצר סיכום ומבחן עם Gemini AI...",
        )
        progress_cb = _make_progress_cb(task_id, TaskStatus.SUMMARIZING, loop)
        result = await summarizer.summarize_transcript(transcript, progress_cb)
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
        cards = await summarizer.generate_flashcards(result.summary, result.transcript)
        result.flashcards = cards
        logger.info(f"Task {task_id}: generated {len(cards)} flashcards")
    except Exception as exc:
        # Soft-fail: the lesson itself is already complete. Log and continue.
        logger.warning(f"Task {task_id}: flashcards generation failed: {exc}")
    return result
