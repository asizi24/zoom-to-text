"""
Pipeline orchestrator.

Two entry points:
  run_pipeline()           — URL → download → process → save result
  run_pipeline_from_file() — uploaded file → process → save result

Each step updates the task's progress in SQLite so the frontend can display
a live progress bar while polling GET /api/tasks/{id}.

Progress milestones:
  5%  → Downloading
  40% → Download complete
  50% → Transcribing (Whisper) OR Sending to Gemini
  80% → Summarizing (Whisper path only)
  100%→ Complete
"""
import logging

from app import state
from app.models import LessonResult, ProcessingMode, TaskStatus
from app.services import summarizer, transcriber, zoom_downloader

logger = logging.getLogger(__name__)


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
        # ── Step 1: Download ──────────────────────────────────────────────────
        await state.update_task(
            task_id, TaskStatus.DOWNLOADING, 5, "⬇️ מוריד את הקלטה מ-Zoom..."
        )
        audio_path = await zoom_downloader.download_audio(
            url=url,
            task_id=task_id,
            cookies_netscape=cookies,
        )
        await state.update_task(
            task_id, TaskStatus.DOWNLOADING, 40, "✅ ההורדה הושלמה. מעבד אודיו..."
        )

        # ── Steps 2 & 3: Process ─────────────────────────────────────────────
        result = await _process_audio(task_id, audio_path, mode, language)

        # ── Step 4: Persist result ───────────────────────────────────────────
        await state.complete_task(task_id, result)
        logger.info(f"Task {task_id} completed ✅")

    except zoom_downloader.ZoomDownloadError as exc:
        logger.error(f"Task {task_id} — download error: {exc}")
        await state.fail_task(task_id, str(exc))

    except Exception as exc:
        logger.exception(f"Task {task_id} — unexpected error")
        await state.fail_task(task_id, str(exc))

    finally:
        # Always clean up the temp audio file regardless of success/failure
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
        await state.complete_task(task_id, result)
        logger.info(f"Task {task_id} (upload) completed ✅")

    except Exception as exc:
        logger.exception(f"Task {task_id} (upload) — unexpected error")
        await state.fail_task(task_id, str(exc))

    finally:
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
    if mode == ProcessingMode.GEMINI_DIRECT:
        # ── Fast path: send audio directly to Gemini ──────────────────────────
        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            50,
            "🤖 שולח אודיו ל-Gemini AI — מייצר סיכום ומבחן...",
        )
        result = await summarizer.summarize_audio(audio_path)

    else:
        # ── Whisper path: local transcription → Gemini ────────────────────────
        await state.update_task(
            task_id,
            TaskStatus.TRANSCRIBING,
            50,
            "🎙️ מתמלל עם Whisper (עשוי לקחת מספר דקות)...",
        )
        transcript, detected_lang = await transcriber.transcribe(audio_path, language)

        await state.update_task(
            task_id,
            TaskStatus.SUMMARIZING,
            80,
            "🤖 יוצר סיכום ומבחן עם Gemini AI...",
        )
        result = await summarizer.summarize_transcript(transcript)
        # Attach the raw transcript to the result so the user can download it
        result.transcript = transcript

    return result
