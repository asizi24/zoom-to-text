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

All Gemini calls use:
  • client.aio — native async; timeouts genuinely cancel the request instead
    of orphaning an executor thread.
  • Structured output (response_schema) — Gemini's constrained decoding
    guarantees schema-valid JSON, so there is no fence-stripping, regex
    anchoring, or escape sanitization anywhere in this module.

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
from pathlib import Path
from typing import Awaitable, Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.config import settings
from app.models import Chapter, Flashcard, LessonResult, QuizQuestion
from app.services.errors import PipelineError

logger = logging.getLogger(__name__)

_ProgressCallback = Callable[[int, str], Awaitable[None]]

# ── Prompt ────────────────────────────────────────────────────────────────────────
# The JSON *shape* is enforced by response_schema (constrained decoding), so the
# prompt spends its budget on content quality: field semantics, code-switching,
# timestamps, and exam design.

_SYSTEM_PROMPT = """
אתה מומחה לחינוך וניתוח שיעורים אקדמיים.
המשימה שלך: לנתח את הקלטת השיעור ולהפיק פלט מובנה ומקיף **בעברית**.

תוכן השדות (המבנה נאכף אוטומטית — אתה אחראי לאיכות התוכן):
  • summary — סיכום מקיף של השיעור כולו (3-5 פסקאות). כסה את כל הנושאים
    המרכזיים והסבר את ה'למה', לא רק את ה'מה'.
  • chapters — חלוקה לפי נושאים לוגיים כפי שהוצגו בשיעור. לכל פרק:
    title (כותרת נושא), content (הסבר מפורט, לפחות 3-4 משפטים),
    key_points (2-4 נקודות מרכזיות).
  • quiz — מבחן אמריקאי לפי ההנחיות המפורטות בהמשך. לכל שאלה 4 options,
    correct_answer שזהה מילה-במילה לאחת האפשרויות, ו-explanation.
  • language — קוד השפה של הפלט ("he").

══════════════════════════════════════════════════
הנחיה קריטית — שמירה על מונחים באנגלית (Code-Switching):
══════════════════════════════════════════════════
בשיעורי הנדסה, מדעי המחשב, רפואה, משפטים וכד' המרצה מערבב עברית עם
מונחים מקצועיים באנגלית. **חובה לשמור על המונחים האנגליים בדיוק כפי
שנאמרו — לא לתרגם, לא לתעתק לעברית, לא להחליף במילה עברית.**

דוגמאות חובה:
  ✅ "הגדרנו useState בקומפוננטת React"        (לא: "הגדרנו שימוש-במצב ברכיב תגובה")
  ✅ "ה-API מחזיר JSON עם status code 200"     (לא: "הממשק מחזיר ג'ייסון עם קוד מצב 200")
  ✅ "הריצו pip install fastapi בטרמינל"       (לא: "התקינו את החבילה המהירה")
  ✅ "TCP מבטיח delivery, UDP לא"               (לא: "פרוטוקול הבקרה מבטיח משלוח")
  ✅ "בתרופות מסוג ACE-inhibitors"              (לא: "בתרופות מעכבות-אנזים-הממיר")

חל על כל חלקי הפלט: summary, chapters.content, key_points, quiz.question,
quiz.options, quiz.correct_answer, quiz.explanation.

השתמש בעברית לחיבור (מילות קישור, פעלים, מבנה משפט), אך שמור שמות של
טכנולוגיות, שפות, פרוטוקולים, מחלות, תרופות, חוקים, ראשי תיבות
ושמות לועזיים — כפי שהם.

══════════════════════════════════════════════════
הנחיה קריטית — ציטוט timestamps:
══════════════════════════════════════════════════
אם התמלול שלפניך מכיל סימוני זמן בפורמט [MM:SS] (למשל "[12:34] ..."),
שלב אותם בתגובותיך כשאתה מתייחס לקטע ספציפי. הסימן מופיע בתחילת
הקטע הרלוונטי והוא יהפוך ל-link בממשק.

דוגמאות:
  ✅ "המרצה הציג את המושג בפרק השני [15:42] והרחיב עליו בהמשך [23:10]"
  ✅ key_points: ["הגדרה של useState נמצאת ב-[05:22]", "דוגמה מעשית ב-[18:45]"]
  ✅ explanation: "ראה הסבר המרצה ב-[42:15] — הוא מדגים שם את הטעות הזו"
אל תמציא timestamps אם הם לא מופיעים במקור.

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

_SYSTEM_PROMPT_EN = """
You are an expert in education and lecture analysis.
Your task is to analyze the lecture recording and produce a structured, comprehensive output in English.

Field requirements (the schema is enforced automatically — you are responsible
for content quality):
  • summary — a thorough lecture summary (3-5 paragraphs). Cover all main
    topics and explain why, not just what.
  • chapters — logical topic sections as presented in lecture. Each chapter
    should include title, content (detailed explanation, at least 3-4 sentences),
    and key_points (2-4 bullet points).
  • quiz — a multiple-choice exam following the rules below. Each question
    must include 4 options, a correct_answer identical to one of the options,
    and an explanation.
  • language — the output language code ("en").

══════════════════════════════════════════════════
Critical instruction — preserve English technical terms:
══════════════════════════════════════════════════
In technical lectures, the instructor may mix local language with English
technical terms. Preserve those terms exactly as spoken — do not translate,
transliterate, or substitute them.

Required examples:
  ✅ "We defined useState in the React component"        (not: "We defined state usage in the component")
  ✅ "The API returns JSON with status code 200"         (not: "The interface returns JSON with code 200")
  ✅ "Run pip install fastapi in the terminal"           (not: "Install the fast package")
  ✅ "TCP guarantees delivery, UDP does not"              (not: "The control protocol guarantees delivery")
  ✅ "ACE-inhibitors are a class of drugs"               (not: "enzyme-converting inhibitor drugs")

This applies to all output fields: summary, chapters.content, key_points,
quiz.question, quiz.options, quiz.correct_answer, quiz.explanation.

Use English for the output but preserve technical names and acronyms in English.

══════════════════════════════════════════════════
Critical instruction — include timestamps:
══════════════════════════════════════════════════
If the transcript includes timestamps in the format [MM:SS], include them
when referring to specific segments. The timestamp appears at the start of the
relevant segment and becomes a link in the UI.

Examples:
  ✅ "The instructor introduced the concept at [15:42] and expanded on it later [23:10]"
  ✅ key_points: ["Definition of useState appears at [05:22]", "Practical example at [18:45]"]
  ✅ explanation: "See the instructor's example at [42:15] — it demonstrates this mistake."
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
בשדה index החזר את האינדקס המקורי של השאלה כפי שניתן לך.
בשדה feedback כתוב הערה קצרה בעברית.

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
"""

_CRITIQUE_PROMPT_EN = """
You are an expert at evaluating academic exam questions.
You received a list of multiple-choice questions from a test on a lecture.
Rate each question on a scale of 1-5 in four criteria:

- clarity    (1-5): Is the question clear and unambiguous?
- difficulty (1-5): Does the question require real understanding (not rote recall)? 1=recall, 5=analysis/synthesis
- distractors(1-5): Do the three wrong options represent common reasoning mistakes?
- accuracy   (1-5): Is the correct answer actually correct and properly justified?

avg = the average of the four scores.
Return the original index of each question in the index field.
Use the feedback field for a short comment in English.

════════════════════════════════════════════════════════════════════════════════
Few-shot examples:
════════════════════════════════════════════════════════════════════════════════

Poor question (low avg):
  ❌ "What year was special relativity published?"
  → clarity:5, difficulty:1, distractors:2, accuracy:5 → avg:3.25
  feedback: "Simple recall question. A student with good memory can answer it without understanding."

Good question (high avg):
  ✅ "Why does local time slow down for an observer moving at high speed compared to a stationary observer?"
  → clarity:5, difficulty:5, distractors:4, accuracy:5 → avg:4.75
  feedback: "Requires understanding of time dilation. The wrong options represent classic misconceptions."

Mediocre question (borderline avg):
  🟡 "Which of the following is an advantage of TCP over UDP?"
  → clarity:4, difficulty:3, distractors:3, accuracy:5 → avg:3.75
  feedback: "The distractors can be improved."
"""

_REVISE_PROMPT_HEADER = """
אתה מומחה לכתיבת שאלות בחינה ברמה אקדמית.
קיבלת מבחן שעבר ביקורת — חלק מההשאלות קיבלו ציון ממוצע נמוך מ-THRESHOLD_PLACEHOLDER.
עליך לכתוב מחדש את השאלות שסומנו כ-NEEDS_REVISION, תוך שמירה על השאלות הטובות כמות שהן.
החזר את רשימת כל השאלות המעודכנות — כולל OK, כולל NEEDS_REVISION.

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

_REVISE_PROMPT_HEADER_EN = """
You are an expert in writing academic exam questions.
You received a reviewed exam — some questions scored below THRESHOLD_PLACEHOLDER.
Rewrite only the questions marked NEEDS_REVISION, while preserving the good questions unchanged.
Return the full updated exam list including OK and NEEDS_REVISION statuses.

Rewrite rules:
  • Any question you replace as NEEDS_REVISION must test understanding, application, or analysis — not recall.
  • All 4 options must use the same length and grammatical structure.
  • Each wrong option must represent a plausible common thinking error.
  • The explanation must explicitly state why each of the 3 wrong options is incorrect.
  • Preserve technical terms in English exactly as written — React, API, TCP, JSON, JWT,
    ACE-inhibitors, etc. Do not translate them.

Original exam with NEEDS_REVISION markings:
"""


# ── Response schemas (constrained decoding) ───────────────────────────────────────
# Dedicated schemas rather than the app models: LessonResult carries fields the
# model must never fill (transcript, flashcards, exam_critique_log), and here
# every field is required so Gemini always emits it.

class _ChapterSchema(BaseModel):
    title: str
    content: str
    key_points: list[str]


class _QuizSchema(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str


class _LessonSchema(BaseModel):
    summary: str
    chapters: list[_ChapterSchema]
    quiz: list[_QuizSchema]
    language: str


class _CritiqueItemSchema(BaseModel):
    index: int
    question: str
    clarity: int
    difficulty: int
    distractors: int
    accuracy: int
    avg: float
    feedback: str


class _CritiqueSchema(BaseModel):
    questions: list[_CritiqueItemSchema]


class _RevisedQuizSchema(BaseModel):
    quiz: list[_QuizSchema]


class _FlashcardSchema(BaseModel):
    front: str
    back: str
    tags: list[str]


class _FlashcardsSchema(BaseModel):
    flashcards: list[_FlashcardSchema]


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

# thinking_budget=0 disables Gemini 2.5 Flash reasoning preamble — combined with
# response_schema, the response body is exactly the requested JSON.
_BASE_KWARGS = dict(
    temperature=0.3,
    max_output_tokens=65536,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)


def _is_english_language(language: str | None) -> bool:
    return str(language or "he").strip().lower() == "en"


def _json_config(system_instruction: str | None, schema: type[BaseModel]) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        **_BASE_KWARGS,
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
    )


def _lesson_config(language: str | None = "he") -> types.GenerateContentConfig:
    return _json_config(
        _SYSTEM_PROMPT_EN if _is_english_language(language) else _SYSTEM_PROMPT,
        _LessonSchema,
    )


def _critique_config(language: str | None = "he") -> types.GenerateContentConfig:
    return _json_config(
        _CRITIQUE_PROMPT_EN if _is_english_language(language) else _CRITIQUE_PROMPT,
        _CritiqueSchema,
    )


def _revise_config(language: str | None = "he") -> types.GenerateContentConfig:
    return _json_config(
        _REVISE_PROMPT_HEADER_EN if _is_english_language(language) else _REVISE_PROMPT_HEADER,
        _RevisedQuizSchema,
    )


def _flashcards_config(language: str | None = "he") -> types.GenerateContentConfig:
    return _json_config(
        _FLASHCARDS_PROMPT_EN if _is_english_language(language) else _FLASHCARDS_PROMPT,
        _FlashcardsSchema,
    )


_LESSON_CONFIG = _json_config(_SYSTEM_PROMPT, _LessonSchema)
_CRITIQUE_CONFIG = _json_config(_CRITIQUE_PROMPT, _CritiqueSchema)
_REVISE_CONFIG = _json_config(_REVISE_PROMPT_HEADER, _RevisedQuizSchema)

# Timeout for each individual Gemini call
_GEMINI_TIMEOUT = 600.0   # 10 minutes — full lesson generation
_ASK_TIMEOUT = 120.0      # 2 minutes — chat answers should be fast
_FLASHCARDS_TIMEOUT = 180.0


# ── Generation helpers ────────────────────────────────────────────────────────────

async def _generate(
    contents,
    config: types.GenerateContentConfig,
    max_retries: int = 3,
    timeout: float = _GEMINI_TIMEOUT,
):
    """
    One Gemini call with typed retry classification and a real timeout.
    429 → long backoff; 5xx → standard backoff; anything else fails immediately.
    asyncio.timeout cancels the underlying HTTP request — no orphaned work.
    """
    client = _get_client()
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout):
                return await client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=contents,
                    config=config,
                )
        except TimeoutError as exc:
            raise PipelineError(
                f"⏱️ Gemini לא הגיב תוך {int(timeout // 60)} דקות — נסה שוב",
                detail=f"generate_content timed out after {timeout}s",
            ) from exc
        except genai_errors.APIError as exc:
            code = exc.code or 0
            is_rate_limit = code == 429
            is_retriable = is_rate_limit or code >= 500
            if attempt == max_retries - 1 or not is_retriable:
                if is_rate_limit:
                    raise PipelineError(
                        "⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות",
                        detail=str(exc),
                    ) from exc
                raise PipelineError(
                    "שגיאה בתקשורת עם Gemini — נסה שוב",
                    detail=str(exc),
                ) from exc
            wait = (4 if is_rate_limit else 2) ** attempt
            logger.warning(
                f"Gemini {'rate-limit' if is_rate_limit else 'server'} error "
                f"(attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {exc}"
            )
            await asyncio.sleep(wait)


async def _generate_structured(
    contents,
    config: types.GenerateContentConfig,
    timeout: float = _GEMINI_TIMEOUT,
):
    """
    Generate with a response_schema config and return response.parsed (a
    pydantic instance). Constrained decoding makes invalid output rare, but
    truncation (max_output_tokens) can still yield parsed=None — retry once.
    """
    for attempt in range(2):
        response = await _generate(contents, config, timeout=timeout)
        parsed = response.parsed
        if parsed is not None:
            return parsed
        logger.warning(
            f"Gemini returned unparseable structured output (attempt {attempt + 1}/2)"
        )
    raise PipelineError(
        "🔄 Gemini החזיר תוצאה לא תקינה — נסה שוב",
        detail="response.parsed was None after 2 attempts (truncated output?)",
    )


def _to_lesson_result(parsed: _LessonSchema) -> LessonResult:
    return LessonResult(
        summary=parsed.summary,
        chapters=[Chapter(**c.model_dump()) for c in parsed.chapters],
        quiz=[QuizQuestion(**q.model_dump()) for q in parsed.quiz],
        language=parsed.language or "he",
    )


# ── Exam critique pipeline ────────────────────────────────────────────────────────

async def critique_exam(exam: list, summary: str, language: str = "he") -> dict:
    """
    Score each question 1-5 on 4 rubrics via Gemini.

    Returns a dict:
      {"questions": [{"index": int, "question": str, "clarity": int,
                      "difficulty": int, "distractors": int, "accuracy": int,
                      "avg": float, "feedback": str}, ...]}

    Failures are non-fatal: an empty critique means the revise pass is skipped.
    """
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

    try:
        parsed: _CritiqueSchema = await _generate_structured(
            f"שאלות המבחן:\n{exam_text}", _critique_config(language)
        )
    except PipelineError as exc:
        logger.error(f"Critique pass failed (non-fatal): {exc}")
        return {"questions": []}
    return parsed.model_dump()


async def revise_exam(exam: list, critique: dict, summary: str, language: str = "he") -> list:
    """
    Rewrite questions whose avg score < settings.exam_critique_threshold.
    Returns a new list[QuizQuestion] — preserving good questions, replacing bad
    ones. Falls back to the original exam on any failure.
    """
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
        marked.append({
            "index": i,
            "question": q.question,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "avg_score": avg,
            "status": "NEEDS_REVISION" if avg < threshold else "OK",
        })

    marked_json = json.dumps(marked, ensure_ascii=False, indent=2)

    # Build prompt via concatenation — NOT .format() — because marked_json and summary
    # contain curly braces that would cause KeyError with Python's str.format().
    header = _REVISE_PROMPT_HEADER.replace("THRESHOLD_PLACEHOLDER", str(threshold))
    prompt = header + marked_json + "\n\nסיכום השיעור להקשר:\n" + summary

    try:
        parsed: _RevisedQuizSchema = await _generate_structured(
            prompt,
            _revise_config(language),
        )
    except PipelineError as exc:
        logger.error(f"Revise pass failed (non-fatal, keeping original exam): {exc}")
        return exam

    revised = [
        QuizQuestion(**q.model_dump())
        for q in parsed.quiz
    ]
    if not revised:
        logger.warning("revise_exam: Gemini returned empty quiz — keeping original")
        return exam
    return revised


def _needs_revision(critique: dict, threshold: float) -> bool:
    """Return True if at least one question scored below the threshold."""
    return any(
        q.get("avg", 5.0) < threshold
        for q in critique.get("questions", [])
    )


async def _apply_critique_pipeline(
    result: LessonResult,
    progress_cb: _ProgressCallback | None = None,
    language: str = "he",
) -> LessonResult:
    """
    Run the critique → revise pipeline on a finished LessonResult.

    Mutates `result.quiz` and sets `result.exam_critique_log` in-place.
    No-ops if ENABLE_EXAM_CRITIQUE=False or the exam is empty.
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
        await progress_cb(90, "🔍 בודק איכות שאלות המבחן...")

    critique = await critique_exam(result.quiz, result.summary, language)
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
            await progress_cb(95, f"✏️ משפר {low_count} שאלות שלא עמדו בסף האיכות...")

        result.quiz = await revise_exam(result.quiz, critique, result.summary, language)
    else:
        logger.info("All questions above threshold — skipping revise pass")
        if progress_cb:
            await progress_cb(95, "✅ כל שאלות המבחן עברו את בדיקת האיכות")

    return result


# ── Direct audio mode ─────────────────────────────────────────────────────────────

async def summarize_audio(
    audio_path: str,
    progress_cb: _ProgressCallback | None = None,
    language: str = "he",
) -> LessonResult:
    """Upload audio to Gemini and get summary + quiz in one call (GEMINI_DIRECT)."""
    client = _get_client()

    audio_prompt = (
        "Analyze the attached lecture recording and produce the required output."
        if _is_english_language(language)
        else "נתח את הקלטת השיעור המצורפת והפק את הפלט הנדרש."
    )

    logger.info(f"Uploading audio to Gemini Files API: {audio_path}")
    try:
        async with asyncio.timeout(_GEMINI_TIMEOUT):
            audio_file = await client.aio.files.upload(file=audio_path)

            if progress_cb:
                await progress_cb(55, "✅ האודיו הועלה ל-Gemini. ממתין לעיבוד הקובץ...")

            max_wait = 300
            waited = 0
            while audio_file.state.name == "PROCESSING":
                if waited >= max_wait:
                    raise PipelineError("⏱️ Gemini לא סיים לעבד את קובץ האודיו תוך 5 דקות")
                await asyncio.sleep(5)
                waited += 5
                audio_file = await client.aio.files.get(name=audio_file.name)

            if audio_file.state.name == "FAILED":
                raise PipelineError("❌ Gemini נכשל בעיבוד קובץ האודיו")
    except TimeoutError as exc:
        raise PipelineError(
            "⏱️ העלאת האודיו ל-Gemini לא הסתיימה תוך 10 דקות — נסה שוב",
            detail="Files API upload/processing timed out",
        ) from exc

    if progress_cb:
        await progress_cb(65, "🔄 Gemini עיבד את הקובץ. מייצר סיכום, פרקים ומבחן...")

    logger.info("Audio processed by Gemini. Generating summary + quiz...")

    if progress_cb:
        await progress_cb(72, "✍️ Gemini כותב את הסיכום והמבחן — עוד רגע...")

    try:
        parsed = await _generate_structured(
            [audio_prompt, audio_file],
            _lesson_config(language),
        )
    finally:
        try:
            await client.aio.files.delete(name=audio_file.name)
            logger.info("Cleaned up Gemini file upload")
        except Exception:
            pass

    # NOTE: GEMINI_DIRECT processes the full audio in one pass — Gemini sees all context
    # (tone, pauses, emphasis) and typically produces higher-quality questions.
    # We skip the critique pipeline here because there is no separate text transcript
    # to attach to the critique request (the audio file was already deleted above).
    return _to_lesson_result(parsed)


# ── Text (transcript) mode ────────────────────────────────────────────────────────

_MAX_CHUNK_CHARS = 350_000


async def summarize_transcript(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
    language: str = "he",
) -> LessonResult:
    """
    Summarize a text transcript (WHISPER_LOCAL / WHISPER_API / IVRIT_AI modes).
    Handles chunking for very long classes, then runs the critique pipeline.
    """
    transcript_prefix = (
        "Transcript:\n" if _is_english_language(language) else "תמלול השיעור:\n"
    )
    if len(transcript) <= _MAX_CHUNK_CHARS:
        parsed = await _generate_structured(
            f"{transcript_prefix}{transcript}", _lesson_config(language)
        )
        result = _to_lesson_result(parsed)
    else:
        result = await _summarize_long_transcript(transcript, progress_cb, language)

    return await _apply_critique_pipeline(result, progress_cb, language)


async def _summarize_long_transcript(
    transcript: str,
    progress_cb: _ProgressCallback | None = None,
    language: str = "he",
) -> LessonResult:
    """Long transcript: chunk → plain-text partial summaries → structured merge."""
    chunks = [
        transcript[i: i + _MAX_CHUNK_CHARS]
        for i in range(0, len(transcript), _MAX_CHUNK_CHARS)
    ]
    n = len(chunks)
    logger.info(f"Transcript is {len(transcript):,} chars — chunking into {n} parts")

    # Intermediate summaries are plain text (no schema) — they only feed the
    # final merge prompt, so JSON here would be pointless overhead.
    partial_config = types.GenerateContentConfig(**_BASE_KWARGS)
    partial_prompt = (
        "Here is a part of the lecture transcript. Summarize the key points in free text for this part only: one to two paragraphs followed by a list of key points. Preserve English technical terms and any [MM:SS] timestamps as they appear."
        if _is_english_language(language)
        else (
            "להלן חלק מתמלול שיעור. סכם בטקסט חופשי את הנקודות המרכזיות "
            "בחלק זה בלבד: פסקה-שתיים של סיכום ואז רשימת נקודות מפתח. "
            "שמור מונחים באנגלית וסימוני זמן [MM:SS] כפי שהם."
        )
    )

    partial_summaries: list[str] = []
    chunk_label = "Part" if _is_english_language(language) else "חלק"
    progress_template = (
        "🔄 Summarizing part {i} of {n}..."
        if _is_english_language(language)
        else "🔄 מסכם חלק {i} מתוך {n}..."
    )
    merge_intro = (
        "Here are intermediate summaries of the lecture parts. Build a complete summary, chapters, and a multiple-choice quiz as required:\n\n"
        if _is_english_language(language)
        else "להלן סיכומי ביניים של חלקי השיעור. בנה מהם סיכום מלא, פרקים ומבחן אמריקאי כפי שנדרש:\n\n"
    )

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {i}/{n}")
        if progress_cb:
            await progress_cb(82 + int(6 * i / n), progress_template.format(i=i, n=n))
        resp = await _generate(f"{partial_prompt}\n\n{chunk_label} {i}:\n{chunk}", partial_config)
        partial_summaries.append(resp.text or "")

    if progress_cb:
        await progress_cb(86, "🔗 מאחד את כל החלקים לסיכום מלא ומבחן..." if not _is_english_language(language) else "🔗 Merging all parts into a full summary and quiz...")

    merge_prompt = merge_intro + "\n\n---\n\n".join(partial_summaries)
    parsed = await _generate_structured(merge_prompt, _lesson_config(language))
    return _to_lesson_result(parsed)


# ── Ask about lesson (chat) ───────────────────────────────────────────────────

_ASK_SYSTEM_PROMPT = """
אתה עוזר לימודי חכם. להלן הקשר של שיעור.
ענה אך ורק על בסיס תוכן השיעור שלהלן.
אם התשובה לא נמצאת בתוכן, אמור זאת בכנות.
ענה בעברית.
"""


async def ask_about_lesson(context: str, question: str) -> str:
    """Answer a student question based on the lesson content."""
    config = types.GenerateContentConfig(
        **_BASE_KWARGS, system_instruction=_ASK_SYSTEM_PROMPT
    )
    response = await _generate(
        f"תוכן השיעור:\n{context}\n\nשאלת התלמיד: {question}",
        config,
        timeout=_ASK_TIMEOUT,
    )
    return response.text or ""


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


def _build_chat_contents(history: list[dict], question: str) -> list:
    """Build the contents list for a multi-turn chat call. The system prompt and
    lesson context travel in system_instruction — no fake user/model turn pair."""
    contents = []
    # Recent history (trim to avoid overly long context)
    trimmed = history[-(2 * _MAX_HISTORY_TURNS):]  # 2× because user+model per turn
    for msg in trimmed:
        contents.append({
            "role": msg.get("role", "user"),
            "parts": [{"text": msg.get("content", "")}],
        })
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


async def stream_chat_response(context: str, history: list[dict], question: str):
    """
    Async generator yielding text chunks from a streaming Gemini chat call.
    Native client.aio streaming — no executor thread, no queue bridge.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        **_BASE_KWARGS,
        # Lesson context rides in the system instruction (capped at 40k chars
        # ≈ 30k tokens) so history stays purely conversational.
        system_instruction=f"{_CHAT_SYSTEM_PROMPT}\n\nתוכן השיעור:\n{context[:40_000]}",
    )
    stream = await client.aio.models.generate_content_stream(
        model=settings.gemini_model,
        contents=_build_chat_contents(history, question),
        config=config,
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text


# ── Flashcards generation ─────────────────────────────────────────────────────
# One extra Gemini call per lesson. Invoked by the processor after exam+critique
# (~95% → 98% progress). Output is stored in LessonResult.flashcards and exported
# to .apkg/.csv on demand.

_FLASHCARDS_PROMPT = """
אתה מומחה לתכנון חומרי לימוד לשיטת חזרה מרווחת (spaced repetition) — בסגנון Anki.
קיבלת סיכום שיעור ותמלול. המשימה: הפק 15-25 כרטיסיות תרגול איכותיות בעברית.

כללי כתיבת כרטיסייה:
  ✅ "front" — שאלה ממוקדת, מושג לזיהוי, או הנחיה קצרה (לא יותר ממשפט אחד)
  ✅ "back" — הסבר קצר, 1-3 משפטים. צריך לעמוד בזכות עצמו (לא "ראה סעיף...").
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
  ❌ שאלה שהתשובה שלה נכתבה מילה במילה ב-front (רמז עצמי)
  ❌ תרגום מונחים טכניים לעברית
"""

_FLASHCARDS_CONFIG = _json_config(_FLASHCARDS_PROMPT, _FlashcardsSchema)


async def generate_flashcards(
    summary: str,
    transcript: str | None = None,
    language: str = "he",
) -> list[Flashcard]:
    """Generate 15-25 flashcards from a lesson summary (+ optional transcript).

    Never raises — flashcards are a bonus step; failures return an empty list
    and the processor keeps the completed lesson.
    """
    if not summary.strip():
        return []

    # Build context — prefer summary; include transcript head for richer detail
    context_parts = [f"סיכום השיעור:\n{summary}"]
    if transcript:
        # Cap at 30k chars — flashcards don't need the full 2-hour transcript
        context_parts.append(f"\nקטע מהתמלול:\n{transcript[:30_000]}")

    try:
        parsed: _FlashcardsSchema = await _generate_structured(
            "\n\n".join(context_parts),
            _flashcards_config(language),
            timeout=_FLASHCARDS_TIMEOUT,
        )
    except PipelineError as exc:
        logger.warning(f"Flashcards generation failed (non-fatal): {exc}")
        return []

    return [
        Flashcard(
            front=c.front.strip(),
            back=c.back.strip(),
            tags=[t.strip() for t in c.tags if t.strip()],
        )
        for c in parsed.flashcards
        if c.front.strip() and c.back.strip()
    ]


# ── Local Ollama summarization (no Gemini dependency) ───────────────────────

_LOCAL_SYSTEM_PROMPT = """
אתה מומחה לחינוך וניתוח שיעורים אקדמיים.
המשימה שלך: לנתח את תמלול השיעור ולהפיק פלט מובנה **בעברית** בלבד.

תוכן השדות הנדרש (החזר JSON חוקי בלבד):
  • summary — סיכום מקיף של השיעור כולו (3-5 פסקאות). כסה את כל הנושאים המרכזיים.
  • chapters — חלוקה לוגית: [{title, content (3-4 משפטים), key_points (2-4)}]
  • quiz — מבחן אמריקאי 8-10 שאלות: [{question, options(4), correct_answer, explanation}]
  • language — "he"

⚠️ חשוב ביותר: החזר **רק** אובייקט JSON חוקי. אל תעטוף ב-```json או ``` או כל markdown אחר.
התחלה חייבת להיות { והסיום } — לא קודם ולא אחריו.

שמור מונחים באנגלית כפי שהם: React, API, TCP, JSON וכו' לא מתורגמים.

══════════════════════════════════════════════════
הנחיה קריטית — ציטוט timestamps:
אם בתמלול יש סימוני זמן [MM:SS], שלב אותם בתשובות כשאתה מתייחס לקטע ספציפי.
אל תמציא timestamps שלא מופיעים במקור.
"""


async def summarize_transcript_with_ollama(
    transcript: str,
    ollama_host: str = "http://localhost:11434",
    model: str = "llama3.2",
    timeout: float = 600.0,
) -> LessonResult:
    """
    Summarize a text transcript using a local Ollama instance instead of Gemini.

    Constructs a prompt from the existing _SYSTEM_PROMPT content (adapted for
    plain-text JSON output), calls ollama.complete(), and parses the raw JSON
    into a LessonResult. Handles markdown fence stripping if the model still
    hallucinates them.
    """
    from app.services.llm import OllamaProvider

    provider = OllamaProvider(
        host=ollama_host,
        model=model,
        timeout=timeout,
    )

    prompt = (
        f"להלן תמלול השיעור:\n\n{transcript}"
    )

    try:
        raw = await provider.complete(
            prompt=prompt,
            system=_LOCAL_SYSTEM_PROMPT,
            temperature=0.3,
            timeout=timeout,
        )
    except Exception as exc:
        raise PipelineError(
            "⚠️ נכשל ביצירת סיכום עם Ollama — נסה שוב",
            detail=f"ollama complete failed: {exc}",
        ) from exc

    if not raw:
        raise PipelineError(
            "⚠️ Ollama החזיר תגובה ריקה — ודא שהמודל מותקן ופועל",
            detail="ollama returned empty response",
        )

    response_text = raw.strip()

    # Strip markdown fences that the LLM may have hallucinated around JSON output
    for _ in range(3):  # loop handles nested or double-wrapped cases
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
    response_text = response_text.strip()

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    cleaned_text = match.group(0) if match else response_text

    try:
        data = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            "⚠️ Ollama החזר פלט לא תקין — נסה שוב",
            detail=f"ollama returned invalid JSON (first 200 chars): {cleaned_text[:200]}",
        ) from exc

    # Build LessonResult from the raw dict
    try:
        return _lesson_from_dict(data)
    except Exception as exc:
        raise PipelineError(
            "⚠️ מבנה הפלט מ-Ollama לא תואם — נסה שוב",
            detail=f"failed to parse ollama result into LessonResult: {exc}",
        ) from exc


def _lesson_from_dict(data: dict) -> LessonResult:
    """Convert a raw dict (from JSON parse) into a LessonResult."""
    summary = data.get("summary", "")
    chapters_data = data.get("chapters", [])
    quiz_data = data.get("quiz", [])

    chapters = [
        Chapter(
            title=ch.get("title", ""),
            content=ch.get("content", ""),
            key_points=ch.get("key_points", []),
        )
        for ch in chapters_data
    ]

    quiz = [
        QuizQuestion(
            question=q.get("question", ""),
            options=q.get("options", []),
            correct_answer=q.get("correct_answer", ""),
            explanation=q.get("explanation", ""),
        )
        for q in quiz_data
    ]

    return LessonResult(
        summary=summary,
        chapters=chapters,
        quiz=quiz,
        language=data.get("language", "he"),
    )
