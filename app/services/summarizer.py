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

# ── Critique prompt ───────────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """
אתה מומחה להערכת שאלות בחינה ברמה אקדמית.
קיבלת רשימת שאלות אמריקאיות מתוך מבחן על שיעור אקדמי.
דרג כל שאלה בסולם 1-5 לפי ארבעה קריטריונים:

- clarity    (1-5): האם השאלה ברורה וחד-משמעית?
- difficulty (1-5): האם השאלה דורשת הבנה אמיתית (לא שינון)? 1=שינון טהור, 5=ניתוח/סינתזה
- distractors(1-5): האם כל שלוש האפשרויות השגויות מייצגות טעות חשיבה שכיחה?
- accuracy   (1-5): האם התשובה הנכונה אכן נכונה ומוצדקת?

avg = ממוצע ארבעת הציונים.

══════════════════════════════════════════════════
דוגמאות few-shot:
══════════════════════════════════════════════════

שאלה גרועה (avg נמוך):
  ❌ "מה השנה בה פורסמה תיאוריית היחסות המיוחדת?"
  → clarity:5, difficulty:1, distractors:2, accuracy:5 → avg:3.25
  feedback: "שאלת שינון ישיר. תלמיד עם זיכרון טוב יענה נכון ללא הבנה"

שאלה טובה (avg גבוה):
  ✅ "מדוע זמן מקומי יאט עבור משקיף הנע במהירות גבוהה יחסית למשקיף אחר?"
  → clarity:5, difficulty:5, distractors:4, accuracy:5 → avg:4.75
  feedback: "דורשת הבנת dilat time. האפשרויות השגויות מייצגות בלבולים קלאסיים"

שאלה בינונית (avg גבולי):
  🟡 "איזה מהבאים הוא יתרון של TCP על UDP?"
  → clarity:4, difficulty:3, distractors:3, accuracy:5 → avg:3.75
  feedback: "ניתן לשפר את האפשרויות השגויות"

══════════════════════════════════════════════════

החזר JSON בלבד (אין טקסט לפני או אחרי):
{
  "questions": [
    {
      "index": 0,
      "question": "טקסט השאלה",
      "clarity": 1-5,
      "difficulty": 1-5,
      "distractors": 1-5,
      "accuracy": 1-5,
      "avg": 1.0-5.0,
      "feedback": "הערה קצרה (עברית)"
    }
  ]
}
"""

_REVISE_PROMPT_HEADER = """
אתה מומחה לכתיבת שאלות בחינה ברמה אקדמית.
קיבלת מבחן שעבר ביקורת — חלק מהשאלות קיבלו ציון ממוצע נמוך מ-THRESHOLD_PLACEHOLDER.
עליך לכתוב מחדש את השאלות שסומנו כ-NEEDS_REVISION, תוך שמירה על השאלות הטובות כמות שהן.

חוקי שכתוב:
  • כל שאלה שתחליף את NEEDS_REVISION חייבת לבדוק הבנה, יישום, ניתוח — לא שינון
  • ארבע האפשרויות יהיו באותו אורך ובאותה מבנה דקדוקי
  • כל אפשרות שגויה מייצגת טעות חשיבה שכיחה
  • ההסבר מציין מדוע כל אחת מהאפשרויות השגויות שגויה

מבחן מקורי עם סימון NEEDS_REVISION:
"""
# NOTE: We intentionally do NOT use .format() for this prompt because both
# the exam JSON and the summary may contain curly braces, which would cause
# KeyError. We build the final prompt by simple string concatenation instead.
_REVISE_PROMPT_SUFFIX = """

החזר JSON בלבד (אין טקסט לפני או אחרי):
{"quiz": [<רשימת כל השאלות המעודכנות, כולל OK, כולל NEEDS_REVISION>]}
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

    Valid JSON single-char escapes: \\ \" / b f n r t  (slash needs no backslash prefix)
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


# ── Exam critique helpers ─────────────────────────────────────────────────────────

def critique_exam(exam: list, summary: str) -> dict:
    """
    Synchronous: score each question 1-5 on 4 rubrics via Gemini.

    Returns a dict:
      {"questions": [{"index": int, "question": str, "clarity": int,
                      "difficulty": int, "distractors": int, "accuracy": int,
                      "avg": float, "feedback": str}, ...]}

    Runs in a thread executor (same pattern as _summarize_*_sync).
    """
    client = _get_client()

    # Serialize the exam for Gemini
    exam_text = json.dumps(
        [
            {
                "index": i,
                "question": q.question,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "explanation": q.explanation,
            }
            for i, q in enumerate(exam)
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"{_CRITIQUE_PROMPT}\n\nשאלות המבחן:\n{exam_text}"
    response = _generate_with_retry(client, prompt)
    text = _response_text(response).strip()

    # Strip fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()

    text = _sanitize_json_escapes(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"Critique JSON parse error: {exc}\nRaw: {text[:300]}")
        # Non-fatal: return empty critique → revise won't run
        return {"questions": []}

    return data


def revise_exam(exam: list, critique: dict, summary: str) -> list:
    """
    Synchronous: rewrite questions whose avg score < settings.exam_critique_threshold.

    Returns a new list[QuizQuestion] — preserving good questions, replacing bad ones.
    Runs in a thread executor.
    """
    client = _get_client()
    threshold = settings.exam_critique_threshold

    # Build score lookup by index
    score_by_idx = {
        q["index"]: q.get("avg", 5.0)
        for q in critique.get("questions", [])
    }

    # Mark questions for revision
    marked = []
    for i, q in enumerate(exam):
        avg = score_by_idx.get(i, 5.0)
        entry = {
            "index": i,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "avg_score": avg,
            "status": "NEEDS_REVISION" if avg < threshold else "OK",
        }
        marked.append(entry)

    marked_json = json.dumps(marked, ensure_ascii=False, indent=2)

    # Build prompt via concatenation — NOT .format() — because marked_json and summary
    # contain curly braces that would cause KeyError with Python's str.format().
    header = _REVISE_PROMPT_HEADER.replace("THRESHOLD_PLACEHOLDER", str(threshold))
    prompt = (
        header
        + marked_json
        + "\n\nסיכום השיעור להקשר:\n"
        + summary
        + _REVISE_PROMPT_SUFFIX
    )

    response = _generate_with_retry(client, prompt)
    text = _response_text(response).strip()

    # Strip fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()

    text = _sanitize_json_escapes(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error(f"Revise JSON parse error: {exc}\nRaw: {text[:300]}")
        # Non-fatal fallback: return original exam unchanged
        return exam

    revised_questions = []
    for q_data in data.get("quiz", []):
        revised_questions.append(
            QuizQuestion(
                question=q_data.get("question", ""),
                options=q_data.get("options", []),
                correct_answer=q_data.get("correct_answer", ""),
                explanation=q_data.get("explanation", ""),
            )
        )

    if not revised_questions:
        logger.warning("revise_exam: Gemini returned empty quiz — keeping original")
        return exam

    return revised_questions


def _needs_revision(critique: dict, threshold: float) -> bool:
    """Return True if at least one question scored below the threshold."""
    return any(
        q.get("avg", 5.0) < threshold
        for q in critique.get("questions", [])
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

    # NOTE: GEMINI_DIRECT processes the full audio in one pass — Gemini sees all context
    # (tone, pauses, emphasis) and typically produces higher-quality questions.
    # We skip the critique pipeline here because there is no separate text transcript
    # to attach to the critique request (the audio file was already deleted above).
    return result


# ── Text (transcript) mode ────────────────────────────────────────────────────────

_MAX_CHUNK_CHARS = 350_000


def _summarize_text_sync(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """
    Send transcript text to Gemini. Handles chunking for very long classes.

    When settings.enable_exam_critique=True, runs an extra two-pass quality check:
      pass 1 — critique_exam: score each question 1-5 on 4 rubrics
      pass 2 — revise_exam:   rewrite questions that scored below the threshold
                               (skipped if all questions are already above threshold)
    """
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

        return _apply_critique_pipeline(result, progress_cb)

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
            # Map chunk progress proportionally into the 82–88% range
            # (leaving 88-95 for critique + revise)
            pct = 82 + int(6 * i / n)
            progress_cb(pct, f"🔄 מסכם חלק {i} מתוך {n}...")
        resp = _generate_with_retry(client, f"{partial_prompt}\n\nחלק {i}:\n{chunk}")
        partial_summaries.append(resp.text)

    if progress_cb:
        progress_cb(86, "🔗 מאחד את כל החלקים לסיכום מלא ומבחן...")

    merge_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        "להלן סיכומי ביניים של חלקי השיעור. "
        "בנה מהם סיכום מלא, פרקים ומבחן אמריקאי כפי שנדרש:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )
    final_response = _generate_with_retry(client, merge_prompt)
    result = _parse_response(_response_text(final_response))

    return _apply_critique_pipeline(result, progress_cb)


def _apply_critique_pipeline(
    result: LessonResult,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """
    Run the critique → revise pipeline on a finished LessonResult.

    Mutates `result.quiz` and sets `result.exam_critique_log` in-place.
    No-ops if ENABLE_EXAM_CRITIQUE=False or the exam is empty.
    Safe to call from any sync context (runs in a thread executor).
    """
    if not settings.enable_exam_critique or not result.quiz:
        return result

    threshold = settings.exam_critique_threshold
    logger.info(
        f"Critique pipeline: scoring {len(result.quiz)} questions "
        f"(threshold={threshold})"
    )

    # ── Pass 1: Critique ──────────────────────────────────────────────────────
    if progress_cb:
        progress_cb(90, "🔍 בודק איכות שאלות המבחן...")

    critique = critique_exam(result.quiz, result.summary)
    result.exam_critique_log = critique  # always save for debugging

    # Log per-question scores
    for q in critique.get("questions", []):
        logger.info(
            f"  Q{q.get('index', '?')}: avg={q.get('avg', '?'):.2f} — "
            f"{q.get('feedback', '')[:60]}"
        )

    # ── Pass 2: Revise (only if needed) ──────────────────────────────────────
    if _needs_revision(critique, threshold):
        low_count = sum(
            1 for q in critique.get("questions", []) if q.get("avg", 5.0) < threshold
        )
        logger.info(
            f"Revising {low_count}/{len(result.quiz)} questions "
            f"(below threshold {threshold})"
        )
        if progress_cb:
            progress_cb(95, f"✏️ משפר {low_count} שאלות שלא עמדו בסף האיכות...")

        revised_quiz = revise_exam(result.quiz, critique, result.summary)
        result.quiz = revised_quiz
    else:
        logger.info("All questions above threshold — skipping revise pass")
        if progress_cb:
            progress_cb(95, "✅ כל שאלות המבחן עברו את בדיקת האיכות")

    return result


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
