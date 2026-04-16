"""
Summarization + Quiz generation using Google Gemini.

Two modes:
  ┌─ GEMINI_DIRECT ─────────────────────────────────────────────────────────────┐
  │  Audio file → uploaded to Gemini Files API → model processes natively       │
  │  • ~2-3 min for a 2-hour class                                              │
  │  • Best accuracy (hears tone, emphasis, speaker pauses)                     │
  │  • Supports up to 9.5 hours of audio per request                            │
  └─────────────────────────────────────────────────────────────────────────────┘
  ┌─ WHISPER_LOCAL ──────────────────────────────────────────────────────────────┐
  │  Transcript text → sent to Gemini as text                                   │
  │  • Used after local Whisper transcription                                   │
  │  • Handles very long transcripts via chunking                               │
  └─────────────────────────────────────────────────────────────────────────────┘

Output is always a structured LessonResult with:
  - summary    : 3-5 paragraph overview
  - chapters   : logical topic breakdown with key points
  - quiz       : 7-10 MCQ questions with 4 options each + explanations
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from google import genai
from google.genai import types

from app.config import settings
from app.models import Chapter, LessonResult, QuizQuestion

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
אתה מומחה לחינוך וניתוח שיעורים אקדמיים.
המשימה שלך: לנתח את הקלטת השיעור ולהפיק פלט מובנה ומקיף **בעברית**.

החזר אך ורק JSON תקין. אל תוסיף שום טקסט לפני או אחרי ה-JSON.
אל תשתמש בגושי קוד markdown. רק JSON גולמי.

הפורמט הנדרש:
{
  "summary": "סיכום מקיף של השיעור כולו (3-5 פסקאות, תסביר גם למה הנושאים חשובים)",
  "chapters": [
    {
      "title": "כותרת נושא",
      "content": "הסבר מפורט של הנושא (לפחות 3-4 משפטים)",
      "key_points": ["נקודה מרכזית 1", "נקודה מרכזית 2", "נקודה מרכזית 3"]
    }
  ],
  "quiz": [
    {
      "question": "שאלה שבודקת הבנה אמיתית ולא רק שינון",
      "options": ["א. תשובה ראשונה", "ב. תשובה שנייה", "ג. תשובה שלישית", "ד. תשובה רביעית"],
      "correct_answer": "א. תשובה ראשונה",
      "explanation": "הסבר מדוע זו התשובה הנכונה ולמה האחרות שגויות"
    }
  ],
  "language": "he"
}

הנחיות קריטיות:
1. סיכום: כסה את כל הנושאים המרכזיים, הסבר את ה'למה' לא רק את ה'מה'
2. פרקים: חלק לפי נושאים לוגיים כפי שהם הוצגו בשיעור
3. מבחן: צור בדיוק 8-10 שאלות אמריקאיות
   - כל שאלה חייבת 4 אפשרויות תשובה (א, ב, ג, ד)
   - השאלות יבדקו הבנה ויישום, לא רק זיכרון עובדתי
   - הפצ'ה הנכונה חייבת להתאים בדיוק לאחת מהאפשרויות
   - הסבר מפורט לכל תשובה
"""

# ── Client (singleton) ────────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return the cached Gemini client, initializing it once on first use."""
    global _client
    if _client is not None:
        return _client

    if settings.google_api_key:
        _client = genai.Client(api_key=settings.google_api_key)
    else:
        creds_path = settings.google_application_credentials
        if Path(creds_path).exists():
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", creds_path)
        _client = genai.Client()

    logger.info(f"Gemini client initialized (model: {settings.gemini_model})")
    return _client


# ── Generation config (shared) ────────────────────────────────────────────────────

_GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.3,        # Low temperature → consistent, factual output
    max_output_tokens=65536,  # Gemini 2.5 Flash max — needed for 3-hour lectures
)


# ── Retry helper ──────────────────────────────────────────────────────────────────

def _generate_with_retry(client: genai.Client, contents, max_retries: int = 3):
    """
    Call generate_content with exponential backoff on transient errors.
    Retries on rate-limit (429) and server errors (5xx).
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=_GEN_CONFIG,
            )
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s → 2s → 4s
            logger.warning(
                f"Gemini error (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait}s: {exc}"
            )
            time.sleep(wait)


# ── Parsing ───────────────────────────────────────────────────────────────────────

def _parse_response(text: str) -> LessonResult:
    """Parse the Gemini JSON response into a LessonResult."""
    # Strip markdown code fences if the model added them despite instructions
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:])
        stripped = stripped.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response (first 500 chars):\n{text[:500]}")
        raise RuntimeError(
            "AI returned malformed JSON. This is usually a transient error — "
            "please try again."
        )

    chapters = [
        Chapter(
            title=c.get("title", ""),
            content=c.get("content", ""),
            key_points=c.get("key_points", []),
        )
        for c in data.get("chapters", [])
    ]

    quiz = [
        QuizQuestion(
            question=q.get("question", ""),
            options=q.get("options", []),
            correct_answer=q.get("correct_answer", ""),
            explanation=q.get("explanation", ""),
        )
        for q in data.get("quiz", [])
    ]

    return LessonResult(
        summary=data.get("summary", ""),
        chapters=chapters,
        quiz=quiz,
        language=data.get("language", "he"),
    )


# ── Direct audio mode ─────────────────────────────────────────────────────────────

def _summarize_audio_sync(audio_path: str) -> LessonResult:
    """Upload audio to Gemini and get summary + quiz in one call."""
    client = _get_client()

    logger.info(f"Uploading audio to Gemini Files API: {audio_path}")
    audio_file = client.files.upload(file=audio_path)

    # Wait for Gemini to process the uploaded file
    max_wait = 300  # 5 minutes
    waited = 0
    while audio_file.state.name == "PROCESSING":
        if waited >= max_wait:
            raise RuntimeError("Gemini timed out processing the audio file")
        time.sleep(5)
        waited += 5
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the audio file")

    logger.info("Audio processed by Gemini. Generating summary + quiz...")
    response = _generate_with_retry(client, [_SYSTEM_PROMPT, audio_file])

    # Clean up the uploaded file from Gemini's storage
    try:
        client.files.delete(name=audio_file.name)
        logger.info("Cleaned up Gemini file upload")
    except Exception:
        pass  # Non-critical

    return _parse_response(response.text)


# ── Text (transcript) mode ────────────────────────────────────────────────────────

# Max characters per Gemini call (~100k tokens ≈ 400k chars; we stay under)
_MAX_CHUNK_CHARS = 350_000


def _summarize_text_sync(transcript: str) -> LessonResult:
    """Send transcript text to Gemini. Handles chunking for very long classes."""
    client = _get_client()

    if len(transcript) <= _MAX_CHUNK_CHARS:
        # Short enough to process in one call
        prompt = f"{_SYSTEM_PROMPT}\n\nתמלול השיעור:\n{transcript}"
        response = _generate_with_retry(client, prompt)
        return _parse_response(response.text)

    # Long transcript: chunk → partial summaries → final merge
    logger.info(
        f"Transcript is {len(transcript):,} chars — chunking into "
        f"{len(transcript) // _MAX_CHUNK_CHARS + 1} parts"
    )

    chunks = [
        transcript[i : i + _MAX_CHUNK_CHARS]
        for i in range(0, len(transcript), _MAX_CHUNK_CHARS)
    ]

    partial_prompt = """
    להלן חלק מתמלול שיעור. סכם את הנקודות המרכזיות בחלק זה בלבד.
    החזר JSON עם שדות: "summary" ו-"key_points" (רשימה).
    """

    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {i}/{len(chunks)}")
        resp = _generate_with_retry(client, f"{partial_prompt}\n\nחלק {i}:\n{chunk}")
        partial_summaries.append(resp.text)

    # Merge all partial summaries into the final structured output
    merge_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        "להלן סיכומי ביניים של חלקי השיעור. "
        "בנה מהם סיכום מלא, פרקים ומבחן אמריקאי כפי שנדרש:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )
    final_response = _generate_with_retry(client, merge_prompt)
    return _parse_response(final_response.text)


# ── Async wrappers ────────────────────────────────────────────────────────────────

async def summarize_audio(audio_path: str) -> LessonResult:
    """Async: upload audio directly to Gemini (GEMINI_DIRECT mode)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _summarize_audio_sync, audio_path)


async def summarize_transcript(transcript: str) -> LessonResult:
    """Async: summarize a text transcript (WHISPER_LOCAL mode)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _summarize_text_sync, transcript)
