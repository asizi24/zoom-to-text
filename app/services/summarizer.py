"""
Summarization + Quiz generation using Google Gemini.

Two modes:
  ┌─ GEMINI_DIRECT ─────────────────────────────────────────────────────────────┐
  │  Audio file → uploaded to Gemini Files API → model processes natively       │
  │  • ~2-3 min for a 2-hour class                                              │
  │  • Best accuracy (hears tone, emphasis, speaker pauses)                     │
  │  • Direct path is used when audio fits in one 1M-token Gemini request.      │
  │    Recordings longer than ``_GEMINI_DIRECT_MAX_SECONDS`` are routed through │
  │    a chunked path: split → transcribe each chunk via Gemini → run the      │
  │    resulting text through ``summarize_transcript`` (same pipeline used by   │
  │    WHISPER modes).                                                          │
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
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types

from app.config import settings
from app.models import (
    ActionItem,
    Chapter,
    Decision,
    Flashcard,
    LessonResult,
    Objection,
    OpenQuestion,
    PerSpeakerSentiment,
    QuizQuestion,
    SentimentAnalysis,
    ToneShift,
)
from app.services.llm_providers import get_provider

logger = logging.getLogger(__name__)

_ProgressCallback = Callable[[int, str], None]


def _is_gemini_provider() -> bool:
    """True if the current provider is Gemini (use existing code path)."""
    return settings.llm_provider == "gemini"

# ── Prompt ────────────────────────────────────────────────────────────────────────

_PROMPT_BODY = """
החזר אך ורק JSON תקין. אל תוסיף שום טקסט לפני או אחרי ה-JSON.
אל תשתמש בגושי קוד markdown. רק JSON גולמי.

הפורמט הנדרש:
{
  "summary": "comprehensive lecture summary (3-5 paragraphs explaining why topics matter)",
  "chapters": [
    {
      "title": "topic title",
      "start_time": "[MM:SS]",
      "content": "detailed explanation of the topic (at least 3-4 sentences)",
      "key_points": ["key point 1", "key point 2", "key point 3"]
    }
  ],
  "quiz": [
    {
      "question": "question testing genuine understanding, not mere recall",
      "bloom_level": 4,
      "options": ["A. first option", "B. second option", "C. third option", "D. fourth option"],
      "correct_answer": "A. first option",
      "explanation": "explanation of why this is correct and why each wrong option is incorrect"
    }
  ],
  "language": "<detected ISO 639-1 code, e.g. he or en>"
}

הנחיות לסיכום ופרקים:
1. סיכום: כסה את כל הנושאים המרכזיים, הסבר את ה'למה' לא רק את ה'מה'
2. פרקים: חלק לפי נושאים לוגיים כפי שהוצגו בשיעור

══════════════════════════════════════════════════
Critical rule — preserve technical terms verbatim:
══════════════════════════════════════════════════
Keep technical terms (frameworks, protocols, APIs, medical/legal terms, proper nouns,
abbreviations) exactly as spoken — do NOT translate or transliterate them.

Examples:
  ✅ "הגדרנו useState בקומפוננטת React"        (not: "הגדרנו שימוש-במצב ברכיב תגובה")
  ✅ "The API returns JSON with status code 200" (not: "The interface returns jason …")
  ✅ "הריצו pip install fastapi בטרמינל"       (not: "התקינו את החבילה המהירה")
  ✅ "TCP guarantees delivery, UDP does not"     (not: "Transmission Control Protocol…")
  ✅ "בתרופות מסוג ACE-inhibitors"              (not: "בתרופות מעכבות-אנזים-הממיר")

Applies to all output fields: summary, chapters.content, key_points, quiz.question,
quiz.options, quiz.correct_answer, quiz.explanation.

══════════════════════════════════════════════════
הנחיה קריטית — ציטוט timestamps (חובה):
══════════════════════════════════════════════════
כל פרק חייב לכלול שדה start_time בפורמט "[MM:SS]" — הזמן שבו הנושא
מתחיל בהקלטה. הוא יהפוך ל-link שמקפיץ את הנגן לנקודה הנכונה בממשק.

אם מקור הקלט הוא תמלול עם סימוני [MM:SS]:
  • קח את ה-[MM:SS] של הפיסקה הראשונה שעוסקת בנושא כ-start_time.
  • שלב timestamps נוספים בתוך key_points ו-content כשאתה מתייחס לרגע ספציפי.

אם מקור הקלט הוא קובץ אודיו ישיר:
  • אמוד את הדקה והשנייה שבהן כל נושא מתחיל ורשום כ-start_time.
  • הוסף timestamps גם בתוך key_points לדוגמאות ספציפיות.

דוגמאות:
  ✅ start_time: "[03:15]"
  ✅ "המרצה הציג את המושג בפרק השני [15:42] והרחיב עליו בהמשך [23:10]"
  ✅ key_points: ["הגדרה של useState נמצאת ב-[05:22]", "דוגמה מעשית ב-[18:45]"]
  ✅ explanation: "ראה הסבר המרצה ב-[42:15] — הוא מדגים שם את הטעות הזו"
אל תמציא timestamps שאינם נסמכים על המקור.

══════════════════════════════════════════════════
הנחיות למבחן — קרא בעיון ופעל לפיהן במדויק:
══════════════════════════════════════════════════

צור בדיוק 8-10 שאלות אמריקאיות עם ההתפלגות הבאה לפי רמות בלום:
  • 2 שאלות הבנה/זיהוי (רמה 1-2): הגדרת מושג, זיהוי מאפיין, השלמת עובדה
  • 4 שאלות ניתוח/יישום (רמה 3-4): למה X גורם ל-Y, מה יקרה אם, השוואת גישות, יישום עיקרון במצב חדש
  • 2+ שאלות הערכה/סינתזה (רמה 5-6): ביקורת על גישה, הסקת מסקנה שלא נאמרה במפורש, פתרון בעיה חדשה

══════════════════════════════════════════════════
דוגמאות לשאלות ברמות בלום גבוהות — כך נראית שאלה טובה:
══════════════════════════════════════════════════

דוגמה 1 — bloom_level 3 (יישום):
  שאלה: "מפתח פותח חיבור חדש ל-DB בכל קריאת API. האפליקציה מקבלת 200 req/s. מה יקרה?"
  תשובה נכונה: "זמן התגובה יגדל — TCP handshake + TLS + auth חוזרים 200 פעמים בשנייה"
  ✅ יישום עיקרון (connection cost) על מצב חדש — לא זכירת הגדרה

דוגמה 2 — bloom_level 4 (ניתוח):
  שאלה: "שירות A מחכה ל-B שמחכה ל-C (עונה תוך 10ms או 500ms לסירוגין).
          מדוע timeout ב-B בלבד אינו פותר את בעיית ה-latency של A?"
  תשובה נכונה: "B חוסם thread תוך המתנה ל-C גם כשה-timeout קצר מ-50ms"
  ✅ ניתוח שרשרת תלויות — לא עובדה שנאמרה במפורש

דוגמה 3 — bloom_level 5 (הערכה):
  שאלה: "ארכיטקט מציע להחליף REST ב-gRPC בין microservices. אילו גורמים מצדיקים זאת?"
  תשובה נכונה: "gRPC מתאים כש-throughput גבוה ו-type-safety קריטיים, אך מסרבך debugging
                 ומחייב client-library לכל שפה — trade-off, לא תשובה חד-ערכית"
  ✅ הערכת trade-offs — לא מידע ישיר מהמצגת
══════════════════════════════════════════════════

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


def _system_prompt() -> str:
    """Return the synthesis system prompt with the configured language instruction."""
    lang = settings.lecture_language
    if lang == "auto":
        lang_line = (
            "זהה את שפת ההרצאה והפק את כל הפלט (summary, chapters, quiz) "
            "**באותה שפה בדיוק**. עברית → ענה בעברית. English → respond in English. וכן הלאה."
        )
    elif lang == "he":
        lang_line = "הפק פלט מובנה ומקיף **בעברית**."
    else:
        lang_line = f"Produce all output in the lecture's language (forced: {lang})."
    return (
        "אתה מומחה לחינוך וניתוח שיעורים אקדמיים.\n"
        f"המשימה שלך: לנתח את הקלטת השיעור ולהפיק פלט מובנה ומקיף. {lang_line}\n"
        + _PROMPT_BODY
    )

# ── Critique prompt ───────────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """
אתה מומחה להערכת שאלות בחינה ברמה אקדמית.
קיבלת רשימת שאלות אמריקאיות מתוך מבחן על שיעור אקדמי.
דרג כל שאלה בסולם 1-5 לפי ארבעה קריטריונים:

- clarity    (1-5): האם השאלה ברורה וחד-משמעית?
- difficulty (1-5): האם השאלה דורשת הבנה אמיתית (לא שינון)? 1=שינון טהור, 5=ניתוח/סינתזה
- distractors(1-5): האם כל שלוש האפשרויות השגויות מייצגות טעות חשיבה שכיחה?
- accuracy   (1-5): האם התשובה הנכונה אכן נכונה ומוצדקת?
- bloom_alignment (1-5): האם השאלה אכן דורשת את התהליך הקוגניטיבי שהוצהר ב-bloom_level?
                          1=שאלת שינון שמוצגת כניתוח, 5=ההתאמה מושלמת

avg = ממוצע חמשת הציונים (clarity + difficulty + distractors + accuracy + bloom_alignment).

══════════════════════════════════════════════════
דוגמאות few-shot:
══════════════════════════════════════════════════

שאלה גרועה (avg נמוך):
  ❌ "מה השנה בה פורסמה תיאוריית היחסות המיוחדת?"
  → clarity:5, difficulty:1, distractors:2, accuracy:5, bloom_alignment:1 → avg:2.8
  feedback: "שאלת שינון ישיר. תלמיד עם זיכרון טוב יענה נכון ללא הבנה"

שאלה טובה (avg גבוה):
  ✅ "מדוע זמן מקומי יאט עבור משקיף הנע במהירות גבוהה יחסית למשקיף אחר?"
  → clarity:5, difficulty:5, distractors:4, accuracy:5, bloom_alignment:5 → avg:4.8
  feedback: "דורשת הבנת dilat time. האפשרויות השגויות מייצגות בלבולים קלאסיים"

שאלה בינונית (avg גבולי):
  🟡 "איזה מהבאים הוא יתרון של TCP על UDP?"
  → clarity:4, difficulty:3, distractors:3, accuracy:5, bloom_alignment:3 → avg:3.6
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
      "bloom_alignment": 1-5,
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
  • **שמר מונחים טכניים באנגלית כפי שהם** — React, API, TCP, JSON, JWT,
    ACE-inhibitors וכו' לא מתורגמים. השאלה נכתבת בעברית עם מונחים
    אנגליים משובצים, כמו שהמרצה דיבר.

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

def _generate_with_retry(client: genai.Client, contents, max_retries: int = 3, config=None):
    """
    Exponential backoff with error classification.
    Rate-limit (429/quota) → longer backoff + user-friendly raise.
    Server errors (5xx) → standard backoff + retry.
    Other errors → fail immediately.

    `config` overrides the default GenerateContentConfig (used by the extraction
    call to lower temperature to 0.2). Pass None to use the standard config.
    """
    cfg = config if config is not None else _GEN_CONFIG
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=settings.gemini_model,
                contents=contents,
                config=cfg,
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
            start_time=c.get("start_time"),
        )
        for c in data.get("chapters", [])
    ]
    quiz = [
        QuizQuestion(
            question=q.get("question", ""),
            options=q.get("options", []),
            correct_answer=q.get("correct_answer", ""),
            explanation=q.get("explanation", ""),
            bloom_level=q.get("bloom_level"),
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
                "bloom_level": q.bloom_level,
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
            "bloom_level": q.bloom_level,
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
                bloom_level=q_data.get("bloom_level"),
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
        response = _generate_with_retry(client, [_system_prompt(), audio_file])
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
        prompt = _system_prompt() + "\n\nתמלול השיעור:\n" + transcript
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
        _system_prompt() + "\n\n"
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


# ── Extraction (Call 2 of the two-call schema upgrade pipeline) ───────────────────

_EXTRACTION_TEMPERATURE = 0.2

_EXTRACTION_PROMPT = """
You are extracting structured business artifacts from a recording. The
recording may be a lecture, a meeting, or an open discussion.

Return STRICT JSON with these top-level keys (ALL OPTIONAL — return [] or
null when a category is irrelevant):

  - action_items:        [{owner, task, deadline?, priority?, source_quote?}]
  - decisions:           [{decision, context?, stakeholders?, source_quote?}]
  - open_questions:      [{question, raised_by?, context?}]
  - sentiment_analysis:  {overall_tone, per_speaker_sentiment[], shifts_in_tone[]}
                         OR null if not applicable (e.g. solo lecture)
  - objections_tracked:  [{objection, raised_by?, response_given?, resolved?}]

Hard rules:
  • DO NOT wrap your output in markdown code fences. Bare JSON only.
  • Every string MUST be in the SOURCE LANGUAGE of the recording.
    Hebrew recording → Hebrew strings. English recording → English.
  • source_quote must be ≤140 chars and verbatim from the audio/transcript.
  • DO NOT invent owners, deadlines, or stakeholders. If unclear, omit
    the optional field for that item.
  • For pure lectures, action_items and objections_tracked are usually [].
  • For meetings, sentiment_analysis is usually populated.

══════════════════════════════════════════════════
Few-shot example 1 — Hebrew product meeting (excerpt):
══════════════════════════════════════════════════
"דן: צריך להחליט עד יום חמישי על המודל החדש. אסף, אתה לוקח את המחקר?
 אסף: כן, אני מסיים עד רביעי. אבל המחיר עוד פתוח — צריך לוודא עם הכספים.
 דן: סבבה. גלית, את בודקת את חוות הדעת המשפטית?"

→ {
  "action_items": [
    {"owner": "אסף", "task": "לסיים את המחקר על המודל החדש",
     "deadline": "יום רביעי", "priority": "high",
     "source_quote": "אסף, אתה לוקח את המחקר? כן, אני מסיים עד רביעי"},
    {"owner": "גלית", "task": "לבדוק חוות דעת משפטית",
     "source_quote": "גלית, את בודקת את חוות הדעת המשפטית?"}
  ],
  "decisions": [],
  "open_questions": [
    {"question": "מה המחיר של המודל החדש?", "raised_by": "אסף",
     "context": "צריך לוודא עם הכספים"}
  ],
  "sentiment_analysis": {
    "overall_tone": "constructive",
    "per_speaker_sentiment": [
      {"speaker": "דן", "sentiment": "positive"},
      {"speaker": "אסף", "sentiment": "neutral"}
    ],
    "shifts_in_tone": []
  },
  "objections_tracked": []
}

══════════════════════════════════════════════════
Few-shot example 2 — English physics lecture (excerpt):
══════════════════════════════════════════════════
"So when we apply Bell's theorem here, we get an inequality that any
 local hidden-variable theory must satisfy. The experiments by Aspect
 in 1982 violated this inequality decisively. Some of you might be
 wondering — does this prove non-locality? That's still debated."

→ {
  "action_items": [],
  "decisions": [],
  "open_questions": [
    {"question": "Does Bell inequality violation actually prove non-locality?",
     "context": "Mentioned by the lecturer as still debated"}
  ],
  "sentiment_analysis": null,
  "objections_tracked": []
}

══════════════════════════════════════════════════

Now extract from the recording above. Return ONLY the JSON object.
"""


def _parse_extraction_response(text: str) -> dict:
    """
    Parse the extraction call's JSON response into a dict of Pydantic objects.

    Returns keys: action_items, decisions, open_questions, sentiment_analysis,
    objections_tracked. Missing keys default to [] / None.

    Raises ValueError on unparseable JSON — callers handle this via best-effort
    skip (the task still completes with synthesis-only fields).
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()

    json_start = stripped.find("{")
    json_end = stripped.rfind("}")
    if json_start != -1 and json_end > json_start:
        stripped = stripped[json_start : json_end + 1]

    stripped = _sanitize_json_escapes(stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extraction call returned invalid JSON: {exc}") from exc

    sentiment = None
    s_raw = data.get("sentiment_analysis")
    if s_raw:
        sentiment = SentimentAnalysis(
            overall_tone=s_raw.get("overall_tone", ""),
            per_speaker_sentiment=[
                PerSpeakerSentiment(
                    speaker=p.get("speaker", ""),
                    sentiment=p.get("sentiment", ""),
                    rationale=p.get("rationale"),
                )
                for p in s_raw.get("per_speaker_sentiment", [])
            ],
            shifts_in_tone=[
                ToneShift(
                    at=t.get("at", ""),
                    **{"from": t.get("from", ""), "to": t.get("to", "")},
                    trigger=t.get("trigger"),
                )
                for t in s_raw.get("shifts_in_tone", [])
            ],
        )

    return {
        "action_items": [
            ActionItem(
                owner=a.get("owner", ""),
                task=a.get("task", ""),
                deadline=a.get("deadline"),
                priority=a.get("priority"),
                source_quote=a.get("source_quote"),
            )
            for a in data.get("action_items", [])
        ],
        "decisions": [
            Decision(
                decision=d.get("decision", ""),
                context=d.get("context"),
                stakeholders=d.get("stakeholders", []),
                source_quote=d.get("source_quote"),
            )
            for d in data.get("decisions", [])
        ],
        "open_questions": [
            OpenQuestion(
                question=q.get("question", ""),
                raised_by=q.get("raised_by"),
                context=q.get("context"),
            )
            for q in data.get("open_questions", [])
        ],
        "sentiment_analysis": sentiment,
        "objections_tracked": [
            Objection(
                objection=o.get("objection", ""),
                raised_by=o.get("raised_by"),
                response_given=o.get("response_given"),
                resolved=o.get("resolved"),
            )
            for o in data.get("objections_tracked", [])
        ],
    }


def _extraction_config() -> "types.GenerateContentConfig":
    """Lower-temperature config for the factual extraction call."""
    return types.GenerateContentConfig(
        temperature=_EXTRACTION_TEMPERATURE,
        max_output_tokens=32768,
        **({} if _thinking_cfg is None else {"thinking_config": _thinking_cfg}),
    )


def _extract_artifacts_text_sync(transcript: str) -> dict:
    """Sync: send transcript to Gemini for extraction-only call (Call 2). Dict only."""
    parsed, _raw = _extract_text_capture(transcript)
    return parsed


def _extract_artifacts_audio_sync(audio_file) -> dict:
    """Sync: send pre-uploaded Gemini audio for extraction. Dict only."""
    parsed, _raw = _extract_audio_capture(audio_file)
    return parsed


def _extract_text_capture(transcript: str) -> tuple[dict, str]:
    """Sync extraction call on text. Returns (parsed dict, raw response text)."""
    client = _get_client()
    prompt = f"{_EXTRACTION_PROMPT}\n\nRecording transcript:\n{transcript[:_MAX_CHUNK_CHARS]}"
    response = _generate_with_retry(client, prompt, config=_extraction_config())
    raw = _response_text(response)
    return _parse_extraction_response(raw), raw


def _extract_audio_capture(audio_file) -> tuple[dict, str]:
    """Sync extraction call on uploaded audio. Returns (parsed dict, raw response text)."""
    client = _get_client()
    response = _generate_with_retry(
        client, [_EXTRACTION_PROMPT, audio_file], config=_extraction_config()
    )
    raw = _response_text(response)
    return _parse_extraction_response(raw), raw


# ── Diarization (Task 1.2 — Gemini-based text speaker labeling) ──────────────────

_DIARIZATION_TEMPERATURE = 0.1

_DIARIZATION_PROMPT = """
You are labeling speakers in a transcript. Your job is purely mechanical:
identify turn boundaries, tag each utterance with a speaker label, and
propose human names ONLY when the transcript itself contains explicit
self-introduction or addressed names.

Return STRICT JSON with these keys (NO markdown fences):
{
  "speaker_count": <int>,
  "speaker_map": {"Speaker A": "<name>", ...},   // {} when no names detected
  "diarized_transcript": "Speaker A: ...\\nSpeaker B: ...\\n..."
}

Hard rules:
  • Output language MUST match the source language. Hebrew → Hebrew. English → English.
  • DO NOT invent names. Only populate speaker_map from explicit cues
    ("שלום, אני אסף...", "Hi, this is Dan", "אסף, מה דעתך?").
  • Preserve the EXACT WORDS of the original transcript — do not paraphrase,
    summarize, or skip. Just split into turns and add the tag prefix.
  • Use sequential labels: Speaker A, Speaker B, Speaker C, …
  • When unsure whether a single utterance changes speaker, keep it under
    the previous tag.

══════════════════════════════════════════════════
Few-shot example 1 — Hebrew product meeting (3 speakers, 2 named):
══════════════════════════════════════════════════
INPUT:
"שלום לכולם, אני אסף מהצוות. רציתי להציג היום את ההתקדמות. דן, אתה רוצה
להוסיף משהו? כן, אני דן, ואני אדבר על ה-API. גם אני רוצה להגיב — ההצעה
נשמעת טוב אבל הסיכון כספי גבוה."

OUTPUT:
{
  "speaker_count": 3,
  "speaker_map": {"Speaker A": "אסף", "Speaker B": "דן"},
  "diarized_transcript": "Speaker A: שלום לכולם, אני אסף מהצוות. רציתי להציג היום את ההתקדמות. דן, אתה רוצה להוסיף משהו?\\nSpeaker B: כן, אני דן, ואני אדבר על ה-API.\\nSpeaker C: גם אני רוצה להגיב — ההצעה נשמעת טוב אבל הסיכון כספי גבוה."
}

══════════════════════════════════════════════════
Few-shot example 2 — English physics lecture (mainly 1 lecturer):
══════════════════════════════════════════════════
INPUT:
"So when we apply Bell's theorem here, we get an inequality that any
local hidden-variable theory must satisfy. Question — does the experiment
prove non-locality? That's still debated."

OUTPUT:
{
  "speaker_count": 2,
  "speaker_map": {},
  "diarized_transcript": "Speaker A: So when we apply Bell's theorem here, we get an inequality that any local hidden-variable theory must satisfy.\\nSpeaker B: Question — does the experiment prove non-locality?\\nSpeaker A: That's still debated."
}

══════════════════════════════════════════════════

Now diarize the transcript above. Return ONLY the JSON object.
"""


def _diarization_config() -> "types.GenerateContentConfig":
    """Lower-temperature config for the mechanical diarization call.

    `response_mime_type="application/json"` forces Gemini into strict-JSON
    mode, eliminating the missing-comma / unescaped-newline failure modes
    we saw graceful-skipping in prod (2026-05-01).
    """
    return types.GenerateContentConfig(
        temperature=_DIARIZATION_TEMPERATURE,
        max_output_tokens=65536,
        response_mime_type="application/json",
        **({} if _thinking_cfg is None else {"thinking_config": _thinking_cfg}),
    )


def _parse_diarization_response(text: str) -> tuple[str, dict[str, str]]:
    """
    Parse the diarization call's JSON. Returns (diarized_transcript, speaker_map).

    Missing `speaker_map` defaults to {} (single-speaker recordings).
    Raises ValueError on unparseable JSON — caller falls back to raw transcript.
    """
    stripped = text.strip()

    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()

    json_start = stripped.find("{")
    json_end = stripped.rfind("}")
    if json_start != -1 and json_end > json_start:
        stripped = stripped[json_start : json_end + 1]

    stripped = _sanitize_json_escapes(stripped)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"diarization call returned invalid JSON: {exc}") from exc

    diarized = data.get("diarized_transcript", "")
    speaker_map = data.get("speaker_map") or {}
    if not isinstance(speaker_map, dict):
        speaker_map = {}
    return diarized, speaker_map


def _diarize_transcript_sync(transcript: str) -> tuple[str, dict[str, str]]:
    """
    Sync: send transcript to Gemini for speaker labeling.

    Returns (diarized_transcript, speaker_map). Raises ValueError on bad JSON
    or any underlying provider error — callers handle via best-effort skip.
    """
    client = _get_client()
    prompt = f"{_DIARIZATION_PROMPT}\n\nTranscript to diarize:\n{transcript[:_MAX_CHUNK_CHARS]}"
    response = _generate_with_retry(client, prompt, config=_diarization_config())
    raw = _response_text(response)
    return _parse_diarization_response(raw)


# ── Synthesis call wrappers that capture raw text (for raw_llm_response) ─────────

def _synthesize_text_capture(transcript: str) -> tuple[LessonResult, str]:
    """
    Single Gemini synthesis call on a transcript. Returns (parsed result, raw text).

    For very long transcripts (> _MAX_CHUNK_CHARS), the chunked path runs and
    raw text reflects the final-merge response only (chunk responses are not
    captured to keep raw_llm_response a manageable size).
    """
    client = _get_client()

    if len(transcript) <= _MAX_CHUNK_CHARS:
        prompt = _system_prompt() + "\n\nתמלול השיעור:\n" + transcript
        last_raw = ""
        last_error: Exception | None = None
        for attempt in range(2):
            response = _generate_with_retry(client, prompt)
            last_raw = _response_text(response)
            try:
                return _parse_response(last_raw), last_raw
            except RuntimeError as e:
                last_error = e
                if attempt == 0:
                    logger.warning("JSON parse failed on first attempt, retrying...")
        raise last_error  # type: ignore[misc]

    # Chunked path — produce partials, merge, then capture raw of final merge
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
        resp = _generate_with_retry(client, f"{partial_prompt}\n\nחלק {i}:\n{chunk}")
        partial_summaries.append(resp.text)

    merge_prompt = (
        _system_prompt() + "\n\n"
        "להלן סיכומי ביניים של חלקי השיעור. "
        "בנה מהם סיכום מלא, פרקים ומבחן אמריקאי כפי שנדרש:\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )
    final_response = _generate_with_retry(client, merge_prompt)
    raw = _response_text(final_response)
    return _parse_response(raw), raw


def _synthesize_audio_capture(audio_file, progress_cb: _ProgressCallback | None = None) -> tuple[LessonResult, str]:
    """Synthesis call on a pre-uploaded audio file. Returns (parsed, raw text)."""
    client = _get_client()
    if progress_cb:
        progress_cb(72, "✍️ Gemini כותב את הסיכום והמבחן — עוד רגע...")

    last_raw = ""
    last_error: Exception | None = None
    for attempt in range(2):
        response = _generate_with_retry(client, [_system_prompt(), audio_file])
        last_raw = _response_text(response)
        try:
            return _parse_response(last_raw), last_raw
        except RuntimeError as e:
            last_error = e
            if attempt == 0:
                logger.warning("JSON parse failed on first attempt, retrying...")
    raise last_error  # type: ignore[misc]


# ── Audio upload + cleanup helpers (split out of _summarize_audio_sync) ──────────

def _upload_audio_to_gemini(audio_path: str, progress_cb: _ProgressCallback | None = None):
    """Upload audio to Gemini Files API and wait for processing. Returns the file handle."""
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

    return audio_file


def _delete_gemini_file(audio_file) -> None:
    """Best-effort cleanup of an uploaded Gemini audio file."""
    try:
        client = _get_client()
        client.files.delete(name=audio_file.name)
        logger.info("Cleaned up Gemini file upload")
    except Exception:
        pass


# ── Long-audio chunked path (used when direct mode would exceed 1M tokens) ───

# Recording duration above which `summarize_audio` routes through the chunked
# path. Without this guard, generate_content fails with
# "input token count exceeds the maximum number of tokens allowed 1048576"
# when the audio + system prompt together overflow the request budget. 45 min
# is conservative — Gemini's documented audio rate is ~32 tokens/sec, but the
# actual rate at full fidelity is meaningfully higher, so we leave generous
# headroom under the 1M limit.
_GEMINI_DIRECT_MAX_SECONDS = 45 * 60

# Per-chunk duration for the long-audio chunked path. 10 minutes keeps each
# Gemini transcribe request well under the 1M-token budget even at the worst
# observed audio token-rate, while bounding the number of round-trips for a
# 2-hour recording to ~12 — each round-trip is ~60-90 s wall-clock.
_GEMINI_CHUNK_SECONDS = 10 * 60

# Maximum concurrent chunk-transcribe calls. 3 is conservative against Gemini's
# free-tier rate limits while cutting wall-clock for a 2-hour recording from
# ~13 min serial to ~5 min. Bumping this past 5 risks 429s on the Files API.
_GEMINI_CHUNK_PARALLELISM = 3


def _is_token_exceeded_error(exc: BaseException) -> bool:
    """True iff `exc` is Gemini's "input token count exceeds the maximum" error.

    Used by `summarize_audio` to fall back to the chunked path when the direct
    request fails despite a sub-threshold duration probe — happens when the
    audio's token-rate is higher than expected (talk-heavy lectures with little
    silence).
    """
    s = str(exc)
    return (
        "exceeds the maximum number of tokens" in s
        or "1048576" in s
        or "input token count" in s.lower()
    )


_TRANSCRIBE_CHUNK_PROMPT = (
    "תמלל את ההקלטה בעברית מילה במילה. "
    "שמור על מונחים באנגלית (שמות טכנולוגיות, מחלות, ראשי תיבות, מושגים "
    "מקצועיים) בדיוק כפי שנאמרו — אל תתרגם או תתעתק לעברית. "
    "אל תוסיף סיכום או הקדמה — החזר אך ורק את הטקסט המתומלל."
)


def _hard_split_audio_for_gemini(audio_path: str) -> list[str]:
    """Split `audio_path` into ≤``_GEMINI_CHUNK_SECONDS`` time slices via ffmpeg.

    Pure time-based slicing, no silence removal — silence-removal failed silently
    on `.mp4`/`.m4a` audio containers and the upstream
    ``audio_preprocessor.preprocess`` fallback returned the *full* file as a
    single chunk, which then re-tripped the 1M-token Gemini limit. This helper
    intentionally has no fallback: each entry in the returned list is a real
    sub-range of the input.

    Each chunk is created with ``-c copy`` so no re-encoding happens; the
    chunks are written to ``tempfile`` paths the caller is responsible for
    deleting via ``_cleanup_chunk_files``.
    """
    duration = _audio_duration_seconds(audio_path)
    if duration is None or duration <= 0:
        raise RuntimeError(
            "❌ ffprobe לא הצליח למדוד את אורך ההקלטה — לא ניתן לחתוך"
        )

    suffix = Path(audio_path).suffix or ".mp3"
    chunk_s = _GEMINI_CHUNK_SECONDS
    n_chunks = int(duration // chunk_s) + (1 if duration % chunk_s else 0)
    chunks: list[str] = []
    for i in range(n_chunks):
        start = i * chunk_s
        fd, out = tempfile.mkstemp(suffix=suffix, prefix=f"zt_chunk{i:02d}_")
        os.close(fd)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-t", str(chunk_s),
                    "-i", audio_path,
                    "-c", "copy",
                    "-loglevel", "error",
                    out,
                ],
                check=True,
                timeout=600,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            # Clean up any chunks already produced before re-raising
            _cleanup_chunk_files(chunks + [out])
            raise
        chunks.append(out)
    return chunks


def _cleanup_chunk_files(paths: list[str]) -> None:
    """Best-effort delete of chunk temp files. Mirrors `cleanup_audio` style."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"chunk cleanup: failed to delete {p}: {exc}")


def _audio_duration_seconds(path: str) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on any failure.

    Used by `summarize_audio` to decide whether the recording fits in a single
    Gemini request or needs to be chunked. Returning None means "unknown" — the
    caller falls back to the direct path (preserving prior behavior on hosts
    where ffprobe is missing).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as exc:
        logger.warning(f"ffprobe duration failed for {path}: {exc}")
        return None


def _transcribe_audio_chunk_via_gemini(chunk_path: str) -> str:
    """Upload one audio chunk and ask Gemini for a verbatim Hebrew transcript.

    Each chunk is ≤13 minutes (per `audio_preprocessor.CHUNK_SECONDS`) so it
    comfortably fits inside the 1M-token request budget alongside the
    transcribe prompt.
    """
    client = _get_client()
    logger.info(f"Transcribing chunk via Gemini: {chunk_path}")
    audio_file = client.files.upload(file=chunk_path)

    # Match the direct-path budget (`_upload_audio_to_gemini`) — Gemini's
    # Files API can take several minutes to finish PROCESSING a 10-minute
    # chunk on busy days, and 3 minutes is not enough headroom.
    waited, max_wait = 0, 300
    while audio_file.state.name == "PROCESSING":
        if waited >= max_wait:
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass
            raise RuntimeError("⏱️ Gemini לא סיים לעבד חלק מהאודיו תוך 5 דקות")
        time.sleep(5)
        waited += 5
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass
        raise RuntimeError("❌ Gemini נכשל בעיבוד חלק מהאודיו")

    transcribe_cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=65536,
        **({} if _thinking_cfg is None else {"thinking_config": _thinking_cfg}),
    )
    try:
        response = _generate_with_retry(
            client, [audio_file, _TRANSCRIBE_CHUNK_PROMPT], config=transcribe_cfg
        )
        return _response_text(response).strip()
    finally:
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass


async def _summarize_audio_chunked(
    audio_path: str,
    progress_cb: _ProgressCallback | None,
) -> LessonResult:
    """Long-audio path: chunk → Gemini-transcribe each piece → summarize text.

    Triggered by `summarize_audio` when the recording exceeds
    `_GEMINI_DIRECT_MAX_SECONDS`. Uses pure time-based ffmpeg slicing
    (`_hard_split_audio_for_gemini`) — silence-removal pre-processing has
    been observed to fail silently on `.mp4`/`.m4a` audio containers, leaving
    the original full file as a single "chunk" that then re-trips the
    1M-token limit. The merged transcript is then funneled through
    `summarize_transcript`, so the user-facing result includes diarization,
    synthesis, extraction, and the exam-critique pass — same as WHISPER modes.
    """
    if progress_cb:
        progress_cb(52, "📦 ההקלטה ארוכה — מחלק לחלקים לתמלול...")

    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(
        None, _hard_split_audio_for_gemini, audio_path
    )
    total = max(len(chunks), 1)

    # Parallelism: up to `_GEMINI_CHUNK_PARALLELISM` chunks in flight at once.
    # Order is preserved by indexing into a results list rather than appending.
    sem = asyncio.Semaphore(_GEMINI_CHUNK_PARALLELISM)
    completed = {"count": 0}

    async def _transcribe_one(idx: int, chunk_path: str) -> str:
        async with sem:
            text = await loop.run_in_executor(
                None, _transcribe_audio_chunk_via_gemini, chunk_path
            )
        completed["count"] += 1
        if progress_cb:
            pct = 55 + int(20 * completed["count"] / total)
            progress_cb(
                pct,
                f"🎙️ תמלול {completed['count']}/{total} עם Gemini...",
            )
        return text

    try:
        transcripts = await asyncio.gather(
            *(_transcribe_one(i, c) for i, c in enumerate(chunks))
        )
    finally:
        _cleanup_chunk_files(chunks)

    full_transcript = "\n\n".join(t for t in transcripts if t)
    if not full_transcript.strip():
        raise RuntimeError(
            "❌ תמלול Gemini החזיר טקסט ריק לכל חלקי ההקלטה — נסה שוב"
        )

    if progress_cb:
        progress_cb(78, "🤖 מסכם את התמלול המאוחד...")

    result = await summarize_transcript(full_transcript, progress_cb)
    # Preserve the merged transcript so the History tab shows it like the
    # WHISPER paths do — without this, callers would see an empty transcript
    # field on a result that was built from a real transcript.
    result.transcript = full_transcript
    return result


# ── Result merging (synthesis Call 1 + extraction Call 2) ────────────────────────

def _merge_results(
    synthesis: LessonResult,
    extraction: dict | None,
    raw_summary: str | None,
    raw_extraction: str | None,
) -> LessonResult:
    """
    Layer extraction-call fields onto the synthesis LessonResult.

    `extraction=None` means the extraction call failed entirely (graceful skip)
    — synthesis fields are preserved as-is and extraction fields stay default.

    `raw_llm_response` is populated only when settings.llm_debug_raw_responses
    is True (per spec §7).
    """
    if extraction:
        synthesis.action_items       = extraction.get("action_items", [])
        synthesis.decisions          = extraction.get("decisions", [])
        synthesis.open_questions     = extraction.get("open_questions", [])
        synthesis.sentiment_analysis = extraction.get("sentiment_analysis")
        synthesis.objections_tracked = extraction.get("objections_tracked", [])

    if settings.llm_debug_raw_responses:
        from app.models import RawLLMResponse
        synthesis.raw_llm_response = RawLLMResponse(
            summary_call=raw_summary,
            extraction_call=raw_extraction,
        )

    return synthesis


# ── Async wrappers ────────────────────────────────────────────────────────────────

async def summarize_audio(
    audio_path: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """
    Async: upload audio directly to LLM (GEMINI_DIRECT mode).

    Two-call pipeline (Task 1.1):
      Call 1 (synthesis)   — summary + chapters + quiz + content_type
      Call 2 (extraction)  — action_items, decisions, sentiment, etc.
    The two calls run in parallel via asyncio.gather. Synthesis failure is
    fatal; extraction failure is best-effort (graceful skip with [] / None).

    Long-audio guard: when the recording exceeds
    ``_GEMINI_DIRECT_MAX_SECONDS``, the request would overflow Gemini's 1M
    input-token budget. We route through the chunked path
    (`_summarize_audio_chunked`) which transcribes each piece via Gemini and
    runs the resulting text through ``summarize_transcript``.
    """
    if not _is_gemini_provider():
        provider = get_provider()
        await provider.upload_audio(audio_path)
        raise RuntimeError("unreachable")

    duration = _audio_duration_seconds(audio_path)
    if duration is not None and duration > _GEMINI_DIRECT_MAX_SECONDS:
        logger.info(
            f"summarize_audio: audio is {duration / 60:.1f} min — exceeds "
            f"{_GEMINI_DIRECT_MAX_SECONDS / 60:.0f}-min direct threshold; "
            f"routing through chunked path"
        )
        return await _summarize_audio_chunked(audio_path, progress_cb)

    loop = asyncio.get_running_loop()

    # Step 1: Upload + wait (sync, in executor) — both calls share this handle.
    audio_file = await loop.run_in_executor(
        None, _upload_audio_to_gemini, audio_path, progress_cb
    )

    try:
        # Step 2: Synthesis + Extraction in parallel.
        synthesis_future = loop.run_in_executor(
            None, _synthesize_audio_capture, audio_file, progress_cb
        )
        extraction_future = loop.run_in_executor(
            None, _extract_audio_capture, audio_file
        )

        synthesis_outcome, extraction_outcome = await asyncio.wait_for(
            asyncio.gather(
                synthesis_future, extraction_future, return_exceptions=True
            ),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        await loop.run_in_executor(None, _delete_gemini_file, audio_file)
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")

    # Step 3: Cleanup — always.
    await loop.run_in_executor(None, _delete_gemini_file, audio_file)

    # Step 4: Synthesis is required.
    if isinstance(synthesis_outcome, BaseException):
        # Resilience: if the direct path tripped the 1M-token limit despite a
        # sub-threshold ffprobe reading, fall back to the chunked path rather
        # than failing the task. Without this, audio with an unexpectedly
        # high token-rate (talk-heavy lectures, dense narration) would still
        # fail at progress=72%.
        if _is_token_exceeded_error(synthesis_outcome):
            logger.warning(
                f"summarize_audio: direct path hit token limit "
                f"({duration / 60 if duration else '?'} min audio); "
                "falling back to chunked path"
            )
            return await _summarize_audio_chunked(audio_path, progress_cb)
        raise synthesis_outcome
    synthesis_result, raw_summary = synthesis_outcome  # type: ignore[misc]

    # Step 5: Extraction is best-effort.
    extraction_dict, raw_extraction = None, None
    if isinstance(extraction_outcome, BaseException):
        logger.warning(
            f"Extraction call failed (graceful skip): {extraction_outcome}"
        )
    else:
        extraction_dict, raw_extraction = extraction_outcome  # type: ignore[misc]

    return _merge_results(synthesis_result, extraction_dict, raw_summary, raw_extraction)


async def summarize_transcript(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
    audio_path: str | None = None,
) -> LessonResult:
    """
    Async: summarize a text transcript (WHISPER_LOCAL / WHISPER_API mode).

    Pipeline:
      1. (Optional) Diarize the transcript — Task 1.2 (Gemini) / Task 2.2 (Pyannote).
         Provider selected by DIARIZATION_PROVIDER env var (default: gemini).
         Pyannote requires audio_path; falls back to raw transcript if missing.
         Skipped when ENABLE_DIARIZATION=False or for non-Gemini LLM providers.
         On failure, falls back to the raw transcript (graceful skip).
      2. Run synthesis + extraction in parallel via asyncio.gather (Task 1.1).
         Synthesis failure propagates; extraction failure is best-effort.
      3. Merge results, then run the exam critique pipeline.
    """
    if not _is_gemini_provider():
        return await _summarize_transcript_via_provider(transcript, progress_cb)

    loop = asyncio.get_running_loop()

    # Step 1: Diarization (best-effort).
    diarized_transcript: str | None = None
    speaker_map: dict[str, str] | None = None
    text_for_downstream = transcript

    if settings.enable_diarization:
        try:
            if settings.diarization_provider == "pyannote":
                if audio_path is None:
                    logger.warning(
                        "DIARIZATION_PROVIDER=pyannote but audio_path not provided — skipping diarization"
                    )
                else:
                    from app.services.diarization import pyannote_provider
                    diarized_transcript, speaker_map = await loop.run_in_executor(
                        None, pyannote_provider.diarize_audio, audio_path, transcript
                    )
            else:
                diarized_transcript, speaker_map = await loop.run_in_executor(
                    None, _diarize_transcript_sync, transcript
                )
            if diarized_transcript:
                text_for_downstream = diarized_transcript
        except Exception as exc:
            logger.warning(
                f"Diarization call failed (graceful skip): {exc}"
            )
            diarized_transcript = None
            speaker_map = None

    # Step 2: Synthesis + Extraction in parallel on the (possibly diarized) text.
    synthesis_future = loop.run_in_executor(
        None, _synthesize_text_capture, text_for_downstream
    )
    extraction_future = loop.run_in_executor(
        None, _extract_text_capture, text_for_downstream
    )

    try:
        synthesis_outcome, extraction_outcome = await asyncio.wait_for(
            asyncio.gather(
                synthesis_future, extraction_future, return_exceptions=True
            ),
            timeout=_GEMINI_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 10 דקות — נסה שוב")

    if isinstance(synthesis_outcome, BaseException):
        raise synthesis_outcome
    synthesis_result, raw_summary = synthesis_outcome  # type: ignore[misc]

    extraction_dict, raw_extraction = None, None
    if isinstance(extraction_outcome, BaseException):
        logger.warning(
            f"Extraction call failed (graceful skip): {extraction_outcome}"
        )
    else:
        extraction_dict, raw_extraction = extraction_outcome  # type: ignore[misc]

    merged = _merge_results(
        synthesis_result, extraction_dict, raw_summary, raw_extraction
    )
    # Persist diarization output (Task 1.2)
    merged.diarized_transcript = diarized_transcript
    merged.speaker_map = speaker_map
    return _apply_critique_pipeline(merged, progress_cb)


async def _summarize_transcript_via_provider(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
) -> LessonResult:
    """Run the full transcript→summary pipeline through a non-Gemini provider."""
    provider = get_provider()

    if progress_cb:
        progress_cb(82, f"🤖 שולח ל-{provider.name} — מייצר סיכום ומבחן...")

    prompt = _system_prompt() + "\n\nתמלול השיעור:\n" + transcript[:_MAX_CHUNK_CHARS]
    text = await provider.generate_text(prompt)
    return _parse_response(text)


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
    if not _is_gemini_provider():
        provider = get_provider()
        prompt = f"{_ASK_SYSTEM_PROMPT}\n\nתוכן השיעור:\n{context}\n\nשאלת התלמיד: {question}"
        return await provider.generate_text(prompt, timeout=_ASK_TIMEOUT)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _ask_sync, context, question),
            timeout=_ASK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise TimeoutError("⏱️ Gemini לא הגיב תוך 2 דקות — נסה שוב")


# ── Streaming multi-turn chat ─────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """\
אתה עוזר לימודי חכם המסייע לתלמיד להבין שיעור.
כללים:
- ענה אך ורק על בסיס תוכן השיעור שניתן לך להלן.
- אם התשובה אינה בתוכן, אמור זאת בכנות ואל תמציא.
- כשאתה מסתמך על חלק ספציפי מהשיעור, ציטט אותו קצרות עם גרשיים ״...״.
- אם בתוכן יש סימוני זמן בפורמט [MM:SS], **שלב אותם בתשובה** כשאתה
  מצביע על נקודה בהקלטה — למשל "המרצה מסביר את זה ב-[12:34]".
  הסימנים יהפכו ל-link שקופץ לנקודה הזאת בנגן.
  אל תמציא timestamps שלא מופיעים במקור.
- ענה בעברית בסגנון ידידותי ומסביר פנים.
- תשובות קצרות ומדויקות עדיפות על פני תשובות ארוכות ומעורפלות.\
"""

# Maximum history turns sent to Gemini per request (user+model pairs)
_MAX_HISTORY_TURNS = 10


def _build_chat_contents(context: str, history: list[dict], question: str) -> list:
    """
    Build the contents list for a multi-turn Gemini chat call.
    The lesson context is embedded in the first user turn as a system-style prefix
    (google-genai's generate_content_stream does not accept a system role directly).
    """
    contents = []

    # First turn: system context as a user message (Gemini ignores "system" role)
    context_turn = (
        f"{_CHAT_SYSTEM_PROMPT}\n\n"
        f"תוכן השיעור:\n{context[:40_000]}"  # cap at 40 k chars (~30 k tokens)
    )
    contents.append({"role": "user", "parts": [{"text": context_turn}]})
    contents.append({"role": "model", "parts": [{"text": "הבנתי. אני מוכן לענות על שאלות על השיעור הזה."}]})

    # Recent history (trim to avoid overly long context)
    trimmed = history[-(2 * _MAX_HISTORY_TURNS):]  # 2× because user+model per turn
    for msg in trimmed:
        role = msg.get("role", "user")
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    # Current user question
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


def _stream_chat_sync(
    context: str,
    history: list[dict],
    question: str,
    out_queue,  # thread-safe queue.Queue
) -> None:
    """
    Blocking: calls Gemini generate_content_stream, puts each text chunk into
    out_queue, and puts None as sentinel on completion or Exception on error.
    """
    import queue as _q_mod
    client = _get_client()
    contents = _build_chat_contents(context, history, question)
    try:
        for chunk in client.models.generate_content_stream(
            model=settings.gemini_model,
            contents=contents,
        ):
            text = getattr(chunk, "text", None)
            if text:
                out_queue.put(text)
    except Exception as exc:
        out_queue.put(exc)
    finally:
        out_queue.put(None)  # sentinel — always sent


async def stream_chat_response(context: str, history: list[dict], question: str):
    """
    Async generator that yields text chunks from a streaming LLM chat call.

    For Gemini: uses the original sync-SDK + thread-queue bridge.
    For OpenRouter / Ollama: uses the provider's native streaming.
    """
    if not _is_gemini_provider():
        provider = get_provider()
        contents = _build_chat_contents(context, history, question)
        async for chunk in provider.stream_text(contents):
            yield chunk
        return

    # Original Gemini path — UNCHANGED (matches existing tests)
    import queue as _q_mod

    loop = asyncio.get_running_loop()
    q: _q_mod.Queue = _q_mod.Queue()

    # Fire off the sync streaming in a thread — don't await, let it run concurrently
    loop.run_in_executor(None, _stream_chat_sync, context, history, question, q)

    while True:
        # Block (in executor) until a chunk or sentinel arrives
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ── Flashcards generation ─────────────────────────────────────────────────────
# One extra Gemini call per lesson. Invoked by the processor after exam+critique
# (~95% → 98% progress). Output is stored in LessonResult.flashcards and exported
# to .apkg/.csv on demand.

_FLASHCARDS_PROMPT = """
אתה מומחה לתכנון חומרי לימוד לשיטת חזרה מרווחת (spaced repetition) — בסגנון Anki.
קיבלת סיכום שיעור ותמלול. המשימה: הפק 15-25 כרטיסיות תרגול איכותיות בעברית.

כללי כתיבת כרטיסייה:
  ✅ "Front" — שאלה ממוקדת, מושג לזיהוי, או הנחיה קצרה (לא יותר ממשפט אחד)
  ✅ "Back" — הסבר קצר, 1-3 משפטים. צריך לעמוד בזכות עצמו (לא "ראה סעיף...").
  ✅ כל כרטיסייה בודקת רעיון אחד בלבד (atomic)
  ✅ **שמור מונחים טכניים באנגלית כפי שהם** — React, API, TCP, JWT וכו'
     לא מתורגמים גם אם יש תרגום עברי מקובל.
  ✅ תייג כל כרטיסייה ב-1-3 תגיות נושא קצרות (מילה אחת כל אחת, ללא רווחים
     ולא סימנים — השתמש ב-underscore אם צריך להצמיד שתי מילים)

דוגמאות טובות:
{"front": "מה תפקיד useState ב-React?", "back": "מחזיר זוג [ערך, setState] שמאפשר לקומפוננטה לנהל state מקומי ולגרום ל-re-render בעת שינוי.", "tags": ["React", "hooks"]}
{"front": "מהו ההבדל המרכזי בין TCP ל-UDP?", "back": "TCP מבטיח מסירה מסודרת ואמינה באמצעות handshake ו-acknowledgments. UDP שולח בלי אישור — מהיר יותר אבל יכול לאבד packets.", "tags": ["networking"]}

אסור:
  ❌ כרטיסיות "רשימה" ("מנה 5 עקרונות של X") — קשה לזכור, תפצל ל-5 כרטיסיות
  ❌ כרטיסיות שהתשובה בהן "כן/לא"
  ❌ שאלה שהתשובה שלה נכתבה מילה במילה ב-Front (רמז עצמי)
  ❌ תרגום מונחים טכניים לעברית

החזר JSON גולמי בלבד, ללא markdown fences, בפורמט הזה:
{
  "flashcards": [
    {"front": "...", "back": "...", "tags": ["...", "..."]}
  ]
}
"""


def _parse_flashcards_response(text: str) -> list[Flashcard]:
    """Parse Gemini flashcards JSON response into a list of Flashcard objects."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()

    # Anchor on our known root key to skip any thinking preamble
    m = re.search(r'\{\s*"flashcards"\s*:', stripped)
    if m:
        stripped = stripped[m.start():]
    json_end = stripped.rfind("}")
    if json_end != -1:
        stripped = stripped[: json_end + 1]

    stripped = _sanitize_json_escapes(stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.error(f"Flashcards JSON parse error: {exc}\nRaw (first 300): {text[:300]}")
        return []

    cards = []
    for c in data.get("flashcards", []):
        front = (c.get("front") or "").strip()
        back  = (c.get("back") or "").strip()
        if not front or not back:
            continue
        tags = [str(t).strip() for t in c.get("tags", []) if str(t).strip()]
        cards.append(Flashcard(front=front, back=back, tags=tags))
    return cards


def _generate_flashcards_sync(summary: str, transcript: str | None) -> list[Flashcard]:
    """Blocking: one Gemini call to produce 15-25 flashcards. Runs in executor."""
    client = _get_client()

    # Build context — prefer summary; include transcript head for richer detail
    context_parts = [f"סיכום השיעור:\n{summary}"]
    if transcript:
        # Cap at 30k chars — flashcards don't need the full 2-hour transcript
        context_parts.append(f"\nקטע מהתמלול:\n{transcript[:30_000]}")
    context = "\n\n".join(context_parts)

    prompt = f"{_FLASHCARDS_PROMPT}\n\n{context}"
    response = _generate_with_retry(client, prompt)
    text = _response_text(response)
    return _parse_flashcards_response(text)


_FLASHCARDS_TIMEOUT = 180.0  # 3 minutes


async def generate_flashcards(
    summary: str,
    transcript: str | None = None,
) -> list[Flashcard]:
    """Async: generate 15-25 flashcards from a lesson summary (+ optional transcript)."""
    if not summary.strip():
        return []

    if not _is_gemini_provider():
        provider = get_provider()
        context_parts = [f"סיכום השיעור:\n{summary}"]
        if transcript:
            context_parts.append(f"\nקטע מהתמלול:\n{transcript[:30_000]}")
        joined_context = "\n\n".join(context_parts)
        prompt = f"{_FLASHCARDS_PROMPT}\n\n{joined_context}"
        try:
            text = await asyncio.wait_for(
                provider.generate_text(prompt),
                timeout=_FLASHCARDS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Flashcards generation timed out — returning empty list")
            return []
        return _parse_flashcards_response(text)

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _generate_flashcards_sync, summary, transcript),
            timeout=_FLASHCARDS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Flashcards generation timed out — returning empty list")
        return []
