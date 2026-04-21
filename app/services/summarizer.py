"""
Summarization + Quiz generation using Google Gemini.

Two modes:
  ┌─ GEMINI_DIRECT ─────────────────────────────────────────────────────────────┐
  │  Audio file → uploaded to Gemini Files API → model processes natively       │
  │  • ~2-3 min for a 2-hour class                                              │
  │  • Best accuracy (hears tone, emphasis, speaker pauses)                     │
  │  • Supports up to 9.5 hours of audio per request                            │
  └─────────────────────────────────────────────────────────────────────────────┘
  ┌─ WHISPER_LOCAL / WHISPER_API ───────────────────────────────────────────────┐
  │  Transcript text → sent to Gemini as text                                   │
  │  • Used after local or API Whisper transcription                            │
  │  • Handles very long transcripts via chunking                               │
  └─────────────────────────────────────────────────────────────────────────────┘

Output is always a structured LessonResult with:
  - summary    : 3-5 paragraph overview
  - chapters   : logical topic breakdown with key points
  - quiz       : 8-10 MCQ questions (Bloom's levels 1-6) with 4 options + explanations
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types

from app.config import settings
from app.models import Chapter, LessonResult, QuizQuestion

logger = logging.getLogger(__name__)

_ProgressCallback = Callable[[int, str], None]

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
      "explanation": "הסבר מדוע זו התשובה הנכונה ולמה כל אחת מהאחרות שגויה"
    }
  ],
  "language": "he"
}

הנחיות לסיכום ופרקים:
1. סיכום: כסה את כל הנושאים המרכזיים, הסבר את ה'למה' לא רק את ה'מה'
2. פרקים: חלק לפי נושאים לוגיים כפי שהוצגו בשיעור

══════════════════════════════════════════════════
הנחיות למבחן — קרא בעיון ופעל לפיהן במדויק:
══════════════════════════════════════════════════

צור בדיוק 8-10 שאלות אמריקאיות עם ההתפלגות הבאה לפי רמות בלום:
  • 2 שאלות הבנה/זיהוי (רמה 1-2): הגדרת מושג, זיהוי מאפיין, השלמת עובדה
  • 4 שאלות ניתוח/יישום (רמה 3-4): למה X גורם ל-Y, מה יקרה אם, השוואת גישות, יישום עיקרון במצב חדש
  • 2+ שאלות הערכה/סינתזה (רמה 5-6): ביקורת על גישה, הסקת מסקנה שלא נאמרה במפורש, פתרון בעיה חדשה

אסור בהחלט — שאלות מהסוגים הבאים פסולות:
  ❌ "מה אמר המרצה על X" — שאלת שינון ישיר מהטקסט
  ❌ "כיצד הגדיר המרצה את..." — ציטוט מחומר ההרצאה
  ❌ שאלה שתלמיד חרוץ יכול לענות עליה מבלי להבין את החומר, רק עם זיכרון טוב

חובה לכל שאלה ותשובותיה:
  ✅ כל 4 האפשרויות באותו אורך ובאותו מבנה דקדוקי — לא "כן/לא" מול משפטים ארוכים
  ✅ כל תשובה שגויה מייצגת טעות מחשבתית שכיחה או הבנה חלקית אמיתית — לא תשובה מגוחכת
  ✅ אין שימוש ב"תמיד", "אף פעם", "בלבד", "רק" בתשובות שגויות (טלטלת MCQ קלאסית)
  ✅ תשובה נכונה אחת ברורה — שלושת האחרות שגויות גם אם נשמעות הגיוניות
  ✅ ההסבר יציין במפורש מדוע כל אחת מ-3 האפשרויות השגויות אינה נכונה
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


# ── Generation config ─────────────────────────────────────────────────────────────

# Attempt to disable thinking on Gemini 2.5 Flash so the model doesn't emit
# reasoning text as preamble before the JSON.  ThinkingConfig was added in
# google-genai ≥1.5 and the exact constructor signature varies — wrap in
# try/except so a missing or mis-versioned ThinkingConfig never crashes the app.
try:
    _thinking_cfg = (
        types.ThinkingConfig(thinking_budget=0)
        if hasattr(types, "ThinkingConfig")
        else None
    )
except Exception:
    _thinking_cfg = None

_GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=65536,
    **({} if _thinking_cfg is None else {"thinking_config": _thinking_cfg}),
)

# Timeout for each async Gemini call (10 minutes)
_GEMINI_TIMEOUT = 600.0


# ── Retry helper ──────────────────────────────────────────────────────────────────

def _generate_with_retry(client: genai.Client, contents, max_retries: int = 3):
    """
    Exponential backoff with error classification.
    Rate-limit (429/quota) → longer backoff + user-friendly raise.
    Server errors (5xx) → standard backoff + retry.
    Other errors → fail immediately.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=_GEN_CONFIG,
            )
        except Exception as exc:
            exc_str = str(exc)
            exc_low = exc_str.lower()
            is_rate_limit = "429" in exc_str or "quota" in exc_low or "rate limit" in exc_low
            is_server_error = any(c in exc_str for c in ("500", "502", "503", "504"))
            is_retriable = is_rate_limit or is_server_error

            if attempt == max_retries - 1 or not is_retriable:
                if is_rate_limit:
                    raise RuntimeError(
                        "⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות"
                    ) from exc
                raise

            wait = (4 if is_rate_limit else 2) ** attempt
            logger.warning(
                f"Gemini {'rate-limit' if is_rate_limit else 'server'} error "
                f"(attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {exc}"
            )
            time.sleep(wait)


# ── Parsing ───────────────────────────────────────────────────────────────────────

def _response_text(response) -> str:
    """
    Extract only non-thinking text from a Gemini response.

    Gemini 2.5 Flash with thinking enabled emits "thought" parts before the
    actual answer.  When those are included in response.text the JSON parser
    sees a preamble full of {…} references and breaks.

    Strategy: iterate the candidate parts and skip any part where thought=True.
    Falls back to response.text if the parts API is unavailable.
    """
    try:
        parts = response.candidates[0].content.parts
        non_thought = [
            p.text
            for p in parts
            if not getattr(p, "thought", False) and getattr(p, "text", None)
        ]
        if non_thought:
            return "\n".join(non_thought)
    except Exception:
        pass
    return response.text or ""


def _sanitize_json_escapes(text: str) -> str:
    """
    Fix invalid JSON escape sequences that Gemini sometimes emits.

    Gemini occasionally produces strings like '...path \\מ...' or '...(AWS \\X)...'
    where a backslash precedes a character that is not a valid JSON escape sequence.
    json.loads rejects these with 'Invalid \\escape'.

    Strategy: double any stray backslash so the parser sees a literal backslash
    followed by the character (valid JSON representing the original text).

    Valid JSON single-char escapes: \" \\ \/ \b \f \n \r \t
    Valid JSON unicode escape:       \\uXXXX  (u + exactly 4 hex digits)
    Everything else → double the backslash.
    """
    return re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', text)


def _parse_response(text: str) -> LessonResult:
    """
    Parse Gemini JSON into a LessonResult.

    Handles several real-world response formats:
      1. Bare JSON (ideal)
      2. ```json ... ``` code fences (Gemini sometimes adds these despite instructions)
      3. Thinking-model output — preamble text before the JSON object
         (Gemini 2.5 Flash can emit reasoning text before the JSON)
      4. Any trailing text after the closing }
      5. Invalid JSON escape sequences (backslash before Hebrew / special chars)

    Strategy: strip fences → find JSON start via "summary" key → sanitize escapes → parse.
    """
    stripped = text.strip()

    # 1. Strip markdown code fences
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:])
        stripped = stripped.rsplit("```", 1)[0].strip()

    # 2. Find the JSON object using our known root key "summary".
    #    Gemini thinking preamble often contains bare { } characters when
    #    referencing JSON schemas, so find("{") picks up the wrong brace.
    #    Searching for '{"summary":' reliably skips all preamble.
    m = re.search(r'\{\s*"summary"\s*:', stripped)
    json_start = m.start() if m else stripped.find("{")
    json_end = stripped.rfind("}")
    if json_start != -1 and json_end > json_start:
        stripped = stripped[json_start : json_end + 1]

    # 3. Fix any invalid escape sequences before handing off to json.loads.
    #    Gemini sometimes emits \X where X is not a valid JSON escape char
    #    (e.g. backslash before a Hebrew letter or an open-paren).
    stripped = _sanitize_json_escapes(stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response (first 500 chars):\n{text[:500]}")
        raise RuntimeError(
            "🔄 Gemini החזיר JSON לא תקין — זו שגיאה חולפת, נסה שוב"
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

def _summarize_audio_sync(
    audio_path: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Upload audio to Gemini and get summary + quiz in one call."""
    client = _get_client()

    logger.info(f"Uploading audio to Gemini Files API: {audio_path}")
    audio_file = client.files.upload(file=audio_path)

    if progress_cb:
        progress_cb(55, "✅ האודיו הועלה ל-Gemini. ממתין לעיבוד הקובץ...")

    max_wait = 300
    waited = 0
    while audio_file.state.name == "PROCESSING":
        if waited >= max_wait:
            raise RuntimeError("⏱️ Gemini לא סיים לעבד את קובץ האודיו תוך 5 דקות")
        time.sleep(5)
        waited += 5
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise RuntimeError("❌ Gemini נכשל בעיבוד קובץ האודיו")

    if progress_cb:
        progress_cb(65, "🔄 Gemini עיבד את הקובץ. מייצר סיכום, פרקים ומבחן...")

    logger.info("Audio processed by Gemini. Generating summary + quiz...")

    if progress_cb:
        progress_cb(72, "✍️ Gemini כותב את הסיכום והמבחן — עוד רגע...")

    result = None
    last_error = None
    for attempt in range(2):
        response = _generate_with_retry(client, [_SYSTEM_PROMPT, audio_file])
        try:
            result = _parse_response(_response_text(response))
            break
        except RuntimeError as e:
            last_error = e
            if attempt == 0:
                logger.warning("JSON parse failed on first attempt, retrying...")

    if result is None:
        raise last_error

    try:
        client.files.delete(name=audio_file.name)
        logger.info("Cleaned up Gemini file upload")
    except Exception:
        pass

    return result


# ── Text (transcript) mode ────────────────────────────────────────────────────────

_MAX_CHUNK_CHARS = 350_000


def _summarize_text_sync(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Send transcript text to Gemini. Handles chunking for very long classes."""
    client = _get_client()

    if len(transcript) <= _MAX_CHUNK_CHARS:
        prompt = f"{_SYSTEM_PROMPT}\n\nתמלול השיעור:\n{transcript}"
        result = None
        last_error = None
        for attempt in range(2):
            response = _generate_with_retry(client, prompt)
            try:
                result = _parse_response(_response_text(response))
                break
            except RuntimeError as e:
                last_error = e
                if attempt == 0:
                    logger.warning("JSON parse failed on first attempt, retrying...")
        if result is None:
            raise last_error
        return result

    # Long transcript: chunk → partial summaries → final merge
    chunks = [
        transcript[i: i + _MAX_CHUNK_CHARS]
        for i in range(0, len(transcript), _MAX_CHUNK_CHARS)
    ]
    n = len(chunks)
    logger.info(f"Transcript is {len(transcript):,} chars — chunking into {n} parts")

    partial_prompt = """
    להלן חלק מתמלול שיעור. סכם את הנקודות המרכזיות בחלק זה בלבד.
    החזר JSON עם שדות: "summary" ו-"key_points" (רשימה).
    """

    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {i}/{n}")
        if progress_cb:
            # Map chunk progress proportionally into the 82–93% range
            pct = 82 + int(11 * i / n)
            progress_cb(pct, f"🔄 מסכם חלק {i} מתוך {n}...")
        resp = _generate_with_retry(client, f"{partial_prompt}\n\nחלק {i}:\n{chunk}")
        partial_summaries.append(resp.text)

    if progress_cb:
        progress_cb(95, "🔗 מאחד את כל החלקים לסיכום מלא ומבחן...")

    merge_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        "להלן סיכומי ביניים של חלקי השיעור. "
        "בנה מהם סיכום מלא, פרקים ומבחן אמריקאי כפי שנדרש:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )
    final_response = _generate_with_retry(client, merge_prompt)
    return _parse_response(_response_text(final_response))


# ── Async wrappers ────────────────────────────────────────────────────────────────

async def summarize_audio(
    audio_path: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Async: upload audio directly to Gemini (GEMINI_DIRECT mode)."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _summarize_audio_sync, audio_path, progress_cb),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")


async def summarize_transcript(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Async: summarize a text transcript (WHISPER_LOCAL / WHISPER_API mode)."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _summarize_text_sync, transcript, progress_cb),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")


# ── Ask about lesson (chat) ───────────────────────────────────────────────────

_ASK_SYSTEM_PROMPT = """
אתה עוזר לימודי חכם. להלן הקשר של שיעור.
ענה אך ורק על בסיס תוכן השיעור שלהלן.
אם התשובה לא נמצאת בתוכן, אמור זאת בכנות.
ענה בעברית.
"""


def _ask_sync(context: str, question: str) -> str:
    """Synchronous: send a question with lesson context to Gemini."""
    client = _get_client()
    prompt = f"{_ASK_SYSTEM_PROMPT}\n\nתוכן השיעור:\n{context}\n\nשאלת התלמיד: {question}"
    response = _generate_with_retry(client, prompt)
    return response.text


_ASK_TIMEOUT = 120.0  # 2 minutes — chat answers should be fast


async def ask_about_lesson(context: str, question: str) -> str:
    """Async: answer a student question based on the lesson content."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _ask_sync, context, question),
            timeout=_ASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 2 דקות — נסה שוב")
