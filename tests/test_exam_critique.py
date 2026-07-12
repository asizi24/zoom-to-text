"""
Tests for the exam self-critique pipeline (Feature 1).

The critique pipeline works in three phases:
  1. generate_exam  — done by summarize_transcript / summarize_audio
  2. critique_exam  — Gemini scores each question 1-5 on 4 rubrics
  3. revise_exam    — if avg score < EXAM_CRITIQUE_THRESHOLD, Gemini rewrites low-scoring questions

Gemini is mocked at the _generate_structured seam: since the refactor to
structured output (response_schema), the summarizer receives parsed pydantic
instances — there is no raw-JSON parsing left to exercise.

The key invariant under test:
  A trivial question ("מה השנה הנוכחית?") has low difficulty/distractors scores,
  bringing its average below 3.5. After critique + revise, that question must
  not appear in the final exam.
"""
import pytest

from app.models import Chapter, QuizQuestion
from app.services import summarizer
from app.services.summarizer import (
    _CritiqueItemSchema,
    _CritiqueSchema,
    _LessonSchema,
    _ChapterSchema,
    _QuizSchema,
    _RevisedQuizSchema,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

TRIVIAL_QUESTION = QuizQuestion(
    question="מה השנה הנוכחית?",
    options=["א. 2024", "ב. 2023", "ג. 2025", "ד. 2026"],
    correct_answer="א. 2024",
    explanation="שנת 2024 הייתה השנה הנוכחית בזמן ההרצאה",
)

GOOD_QUESTION = QuizQuestion(
    question="מדוע שימוש ב-connection pooling מפחית latency בבקשות DB?",
    options=[
        "א. מונע יצירת חיבורים חדשים לכל בקשה ומנצל חיבורים קיימים",
        "ב. מדחס את הנתונים לפני שליחה לשרת ה-DB",
        "ג. מסנן שאילתות כפולות ברמת ה-ORM",
        "ד. מאחסן תוצאות שאילתות בזיכרון cache מקומי",
    ],
    correct_answer="א. מונע יצירת חיבורים חדשים לכל בקשה ומנצל חיבורים קיימים",
    explanation="יצירת TCP connection + TLS handshake + auth עולה 50-200ms. Pool שומר חיבורים פתוחים.",
)

SAMPLE_SUMMARY = "השיעור עסק בארכיטקטורת מסדי נתונים ואופטימיזציית ביצועים."

SAMPLE_CHAPTERS = [
    Chapter(title="מבוא", content="הסבר מבוא", key_points=["נקודה א"])
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quiz_schema(q: QuizQuestion) -> _QuizSchema:
    return _QuizSchema(
        question=q.question,
        options=q.options,
        correct_answer=q.correct_answer,
        explanation=q.explanation,
    )


def _critique_schema(questions: list[QuizQuestion], scores: list[float]) -> _CritiqueSchema:
    """Build a parsed critique response for mocking."""
    items = []
    for i, (q, avg) in enumerate(zip(questions, scores)):
        s = round(avg)
        items.append(_CritiqueItemSchema(
            index=i,
            question=q.question,
            clarity=s,
            difficulty=s,
            distractors=s,
            accuracy=s,
            avg=avg,
            feedback="טריוויאלית — שינון ישיר" if avg < 3.5 else "שאלה טובה",
        ))
    return _CritiqueSchema(questions=items)


def _lesson_schema(quiz: list[QuizQuestion]) -> _LessonSchema:
    return _LessonSchema(
        summary=SAMPLE_SUMMARY,
        chapters=[_ChapterSchema(**c.model_dump()) for c in SAMPLE_CHAPTERS],
        quiz=[_quiz_schema(q) for q in quiz],
        language="he",
    )


def _patch_structured(monkeypatch, responses: list):
    """Patch _generate_structured to return canned parsed objects in order.
    Returns the call-count dict."""
    calls = {"n": 0}

    async def fake_structured(contents, config, timeout=None, **kwargs):
        resp = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return resp

    monkeypatch.setattr(summarizer, "_generate_structured", fake_structured)
    return calls


# ── Unit tests for critique_exam ──────────────────────────────────────────────

class TestCritiqueExam:
    """critique_exam() must score each question and return structured data."""

    @pytest.mark.asyncio
    async def test_returns_per_question_scores(self, monkeypatch):
        """critique_exam returns a dict with a 'questions' list, one entry per input question."""
        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(monkeypatch, [_critique_schema(exam, [2.5, 4.25])])

        result = await summarizer.critique_exam(exam, SAMPLE_SUMMARY)

        assert "questions" in result
        assert len(result["questions"]) == 2

    @pytest.mark.asyncio
    async def test_trivial_question_scores_below_threshold(self, monkeypatch):
        """The trivial question ends up with avg < 3.5."""
        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(monkeypatch, [_critique_schema(exam, [2.5, 4.25])])

        result = await summarizer.critique_exam(exam, SAMPLE_SUMMARY)

        avgs = [q["avg"] for q in result["questions"]]
        assert avgs[0] < 3.5, f"Trivial question avg should be < 3.5, got {avgs[0]}"
        assert avgs[1] >= 3.5, f"Good question avg should be >= 3.5, got {avgs[1]}"


# ── Unit tests for revise_exam ────────────────────────────────────────────────

class TestReviseExam:
    """revise_exam() must replace low-scoring questions and return updated exam."""

    CRITIQUE = {
        "questions": [
            {"index": 0, "question": TRIVIAL_QUESTION.question, "avg": 2.5, "feedback": "טריוויאלית"},
            {"index": 1, "question": GOOD_QUESTION.question, "avg": 4.25, "feedback": "טובה"},
        ]
    }

    @pytest.mark.asyncio
    async def test_trivial_question_removed_after_revise(self, monkeypatch):
        """Revised exam must not contain the trivial question."""
        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(
            monkeypatch, [_RevisedQuizSchema(quiz=[_quiz_schema(GOOD_QUESTION)])]
        )

        result = await summarizer.revise_exam(exam, self.CRITIQUE, SAMPLE_SUMMARY)

        question_texts = [q.question for q in result]
        assert TRIVIAL_QUESTION.question not in question_texts, (
            f"Trivial question should have been removed, but found: {question_texts}"
        )

    @pytest.mark.asyncio
    async def test_good_questions_preserved_after_revise(self, monkeypatch):
        """Revised exam keeps high-scoring questions intact."""
        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(
            monkeypatch, [_RevisedQuizSchema(quiz=[_quiz_schema(GOOD_QUESTION)])]
        )

        result = await summarizer.revise_exam(exam, self.CRITIQUE, SAMPLE_SUMMARY)

        question_texts = [q.question for q in result]
        assert GOOD_QUESTION.question in question_texts

    @pytest.mark.asyncio
    async def test_empty_revision_keeps_original_exam(self, monkeypatch):
        """If Gemini returns an empty quiz, the original exam is preserved."""
        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(monkeypatch, [_RevisedQuizSchema(quiz=[])])

        result = await summarizer.revise_exam(exam, self.CRITIQUE, SAMPLE_SUMMARY)

        assert len(result) == 2


# ── Integration test: full pipeline rejects trivial question ──────────────────

class TestExamCritiquePipeline:
    """
    End-to-end: summarize_transcript with ENABLE_EXAM_CRITIQUE=True must
    call critique → revise and return an exam without trivial questions.
    """

    @pytest.mark.asyncio
    async def test_pipeline_removes_trivial_question_when_critique_enabled(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", True, raising=False)
        monkeypatch.setattr(settings, "exam_critique_threshold", 3.5, raising=False)

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(monkeypatch, [
            _lesson_schema(exam),                                   # 1. initial generation
            _critique_schema(exam, [2.5, 4.25]),                    # 2. critique
            _RevisedQuizSchema(quiz=[_quiz_schema(GOOD_QUESTION)]), # 3. revise
        ])

        result = await summarizer.summarize_transcript("תמלול לדוגמה קצר")

        question_texts = [q.question for q in result.quiz]
        assert TRIVIAL_QUESTION.question not in question_texts, (
            f"Trivial question must be removed after critique+revise. Got: {question_texts}"
        )
        assert GOOD_QUESTION.question in question_texts, (
            f"Good question must be preserved. Got: {question_texts}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_skips_critique_when_disabled(self, monkeypatch):
        """When ENABLE_EXAM_CRITIQUE=False, only one Gemini call is made."""
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", False, raising=False)

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        calls = _patch_structured(monkeypatch, [_lesson_schema(exam)])

        result = await summarizer.summarize_transcript("תמלול לדוגמה קצר")

        # When disabled, exactly 1 call (no critique, no revise)
        assert calls["n"] == 1, (
            f"Expected 1 Gemini call when critique disabled, got {calls['n']}"
        )
        # Original trivial question should still be there (not filtered)
        question_texts = [q.question for q in result.quiz]
        assert TRIVIAL_QUESTION.question in question_texts

    @pytest.mark.asyncio
    async def test_critique_log_saved_in_result(self, monkeypatch):
        """exam_critique_log must appear in LessonResult when critique runs."""
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", True, raising=False)
        monkeypatch.setattr(settings, "exam_critique_threshold", 3.5, raising=False)

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        _patch_structured(monkeypatch, [
            _lesson_schema(exam),
            _critique_schema(exam, [2.5, 4.25]),
            _RevisedQuizSchema(quiz=[_quiz_schema(GOOD_QUESTION)]),
        ])

        result = await summarizer.summarize_transcript("תמלול לדוגמה קצר")

        assert hasattr(result, "exam_critique_log"), (
            "LessonResult must have exam_critique_log field when critique runs"
        )
        assert result.exam_critique_log is not None, "exam_critique_log must not be None"
