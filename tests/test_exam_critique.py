"""
Tests for the exam self-critique pipeline (Feature 1).

TDD approach — these tests were written BEFORE the implementation.

The critique pipeline works in three phases:
  1. generate_exam  — already done by summarize_transcript / summarize_audio
  2. critique_exam  — Gemini scores each question 1-5 on 4 rubrics
  3. revise_exam    — if avg score < EXAM_CRITIQUE_THRESHOLD, Gemini rewrites low-scoring questions

The key invariant under test:
  A trivial question ("מה השנה הנוכחית?") has low difficulty/distractors scores,
  bringing its average below 3.5. After critique + revise, that question must
  not appear in the final exam.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from app.models import Chapter, LessonResult, QuizQuestion


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

def _mock_response(text: str) -> MagicMock:
    """Wrap a text string in a mock Gemini response object."""
    mock = MagicMock()
    mock.text = text
    part = MagicMock()
    part.thought = False
    part.text = text
    mock.candidates = [MagicMock()]
    mock.candidates[0].content.parts = [part]
    return mock


def _critique_json(questions: list[QuizQuestion], scores: list[float]) -> str:
    """Build a valid critique JSON string for mocking."""
    items = []
    for i, (q, avg) in enumerate(zip(questions, scores)):
        # Round scores so individual criteria are plausible integers
        s = round(avg)
        items.append({
            "index": i,
            "question": q.question,
            "clarity": s,
            "difficulty": s,
            "distractors": s,
            "accuracy": s,
            "avg": avg,
            "feedback": "טריוויאלית — שינון ישיר" if avg < 3.5 else "שאלה טובה",
        })
    return json.dumps({"questions": items}, ensure_ascii=False)


def _revised_exam_json(questions: list[QuizQuestion]) -> str:
    """Build a valid revised exam JSON string."""
    return json.dumps(
        {"quiz": [q.model_dump() for q in questions]},
        ensure_ascii=False,
    )


# ── Unit tests for critique_exam ──────────────────────────────────────────────

class TestCritiqueExam:
    """critique_exam() must score each question and return structured data."""

    def test_returns_per_question_scores(self, monkeypatch):
        """critique_exam returns a dict with a 'questions' list, one entry per input question."""
        from app.services.summarizer import critique_exam

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        critique_text = _critique_json(exam, [2.5, 4.25])

        with patch("app.services.summarizer._generate_with_retry") as mock_gen:
            mock_gen.return_value = _mock_response(critique_text)
            result = critique_exam(exam, SAMPLE_SUMMARY)

        assert "questions" in result
        assert len(result["questions"]) == 2

    def test_trivial_question_scores_below_threshold(self, monkeypatch):
        """The trivial question ends up with avg < 3.5."""
        from app.services.summarizer import critique_exam

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        critique_text = _critique_json(exam, [2.5, 4.25])

        with patch("app.services.summarizer._generate_with_retry") as mock_gen:
            mock_gen.return_value = _mock_response(critique_text)
            result = critique_exam(exam, SAMPLE_SUMMARY)

        avgs = [q["avg"] for q in result["questions"]]
        assert avgs[0] < 3.5, f"Trivial question avg should be < 3.5, got {avgs[0]}"
        assert avgs[1] >= 3.5, f"Good question avg should be >= 3.5, got {avgs[1]}"


# ── Unit tests for revise_exam ────────────────────────────────────────────────

class TestReviseExam:
    """revise_exam() must replace low-scoring questions and return updated exam."""

    def test_trivial_question_removed_after_revise(self):
        """Revised exam must not contain the trivial question."""
        from app.services.summarizer import revise_exam

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        critique = {
            "questions": [
                {"index": 0, "question": TRIVIAL_QUESTION.question, "avg": 2.5, "feedback": "טריוויאלית"},
                {"index": 1, "question": GOOD_QUESTION.question, "avg": 4.25, "feedback": "טובה"},
            ]
        }

        # Mock Gemini to return an exam without the trivial question
        revised_text = _revised_exam_json([GOOD_QUESTION])

        with patch("app.services.summarizer._generate_with_retry") as mock_gen:
            mock_gen.return_value = _mock_response(revised_text)
            result = revise_exam(exam, critique, SAMPLE_SUMMARY)

        question_texts = [q.question for q in result]
        assert TRIVIAL_QUESTION.question not in question_texts, (
            f"Trivial question should have been removed, but found: {question_texts}"
        )

    def test_good_questions_preserved_after_revise(self):
        """Revised exam keeps high-scoring questions intact."""
        from app.services.summarizer import revise_exam

        exam = [TRIVIAL_QUESTION, GOOD_QUESTION]
        critique = {
            "questions": [
                {"index": 0, "question": TRIVIAL_QUESTION.question, "avg": 2.5, "feedback": "טריוויאלית"},
                {"index": 1, "question": GOOD_QUESTION.question, "avg": 4.25, "feedback": "טובה"},
            ]
        }

        revised_text = _revised_exam_json([GOOD_QUESTION])

        with patch("app.services.summarizer._generate_with_retry") as mock_gen:
            mock_gen.return_value = _mock_response(revised_text)
            result = revise_exam(exam, critique, SAMPLE_SUMMARY)

        question_texts = [q.question for q in result]
        assert GOOD_QUESTION.question in question_texts


# ── Integration test: full pipeline rejects trivial question ──────────────────

class TestExamCritiquePipeline:
    """
    End-to-end: _summarize_text_sync with ENABLE_EXAM_CRITIQUE=True must
    call critique → revise and return an exam without trivial questions.
    """

    def test_pipeline_removes_trivial_question_when_critique_enabled(self, monkeypatch):
        """
        Given a transcript that causes Gemini to produce a trivial question,
        the full pipeline (with critique enabled) must return a result
        where that trivial question does not appear.
        """
        from app.services import summarizer
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", True, raising=False)
        monkeypatch.setattr(settings, "exam_critique_threshold", 3.5, raising=False)

        # Build the three Gemini responses in sequence:
        # 1. Initial generation → exam with trivial question
        initial_exam = LessonResult(
            summary=SAMPLE_SUMMARY,
            chapters=SAMPLE_CHAPTERS,
            quiz=[TRIVIAL_QUESTION, GOOD_QUESTION],
            language="he",
        )
        initial_json = json.dumps(
            {
                "summary": initial_exam.summary,
                "chapters": [c.model_dump() for c in initial_exam.chapters],
                "quiz": [q.model_dump() for q in initial_exam.quiz],
                "language": "he",
            },
            ensure_ascii=False,
        )

        # 2. Critique → trivial question scores < 3.5
        critique_text = _critique_json([TRIVIAL_QUESTION, GOOD_QUESTION], [2.5, 4.25])

        # 3. Revise → exam without trivial question
        revised_text = _revised_exam_json([GOOD_QUESTION])

        call_count = {"n": 0}
        responses = [initial_json, critique_text, revised_text]

        def fake_generate(client, contents, max_retries=3):
            resp_text = responses[call_count["n"] % len(responses)]
            call_count["n"] += 1
            return _mock_response(resp_text)

        monkeypatch.setattr(summarizer, "_generate_with_retry", fake_generate)

        result = summarizer._summarize_text_sync("תמלול לדוגמה קצר")

        question_texts = [q.question for q in result.quiz]
        assert TRIVIAL_QUESTION.question not in question_texts, (
            f"Trivial question must be removed after critique+revise. Got: {question_texts}"
        )
        assert GOOD_QUESTION.question in question_texts, (
            f"Good question must be preserved. Got: {question_texts}"
        )

    def test_pipeline_skips_critique_when_disabled(self, monkeypatch):
        """When ENABLE_EXAM_CRITIQUE=False, only one Gemini call is made."""
        from app.services import summarizer
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", False, raising=False)

        initial_exam = LessonResult(
            summary=SAMPLE_SUMMARY,
            chapters=SAMPLE_CHAPTERS,
            quiz=[TRIVIAL_QUESTION, GOOD_QUESTION],
            language="he",
        )
        initial_json = json.dumps(
            {
                "summary": initial_exam.summary,
                "chapters": [c.model_dump() for c in initial_exam.chapters],
                "quiz": [q.model_dump() for q in initial_exam.quiz],
                "language": "he",
            },
            ensure_ascii=False,
        )

        call_count = {"n": 0}

        def fake_generate(client, contents, max_retries=3):
            call_count["n"] += 1
            return _mock_response(initial_json)

        monkeypatch.setattr(summarizer, "_generate_with_retry", fake_generate)

        result = summarizer._summarize_text_sync("תמלול לדוגמה קצר")

        # When disabled, exactly 1 call (no critique, no revise)
        assert call_count["n"] == 1, (
            f"Expected 1 Gemini call when critique disabled, got {call_count['n']}"
        )
        # Original trivial question should still be there (not filtered)
        question_texts = [q.question for q in result.quiz]
        assert TRIVIAL_QUESTION.question in question_texts

    def test_critique_log_saved_in_result(self, monkeypatch):
        """exam_critique_log must appear in LessonResult when critique runs."""
        from app.services import summarizer
        from app.config import settings

        monkeypatch.setattr(settings, "enable_exam_critique", True, raising=False)
        monkeypatch.setattr(settings, "exam_critique_threshold", 3.5, raising=False)

        initial_exam = LessonResult(
            summary=SAMPLE_SUMMARY,
            chapters=SAMPLE_CHAPTERS,
            quiz=[TRIVIAL_QUESTION, GOOD_QUESTION],
            language="he",
        )
        initial_json = json.dumps(
            {
                "summary": initial_exam.summary,
                "chapters": [c.model_dump() for c in initial_exam.chapters],
                "quiz": [q.model_dump() for q in initial_exam.quiz],
                "language": "he",
            },
            ensure_ascii=False,
        )
        critique_text = _critique_json([TRIVIAL_QUESTION, GOOD_QUESTION], [2.5, 4.25])
        revised_text = _revised_exam_json([GOOD_QUESTION])

        call_count = {"n": 0}
        responses = [initial_json, critique_text, revised_text]

        def fake_generate(client, contents, max_retries=3):
            resp_text = responses[call_count["n"] % len(responses)]
            call_count["n"] += 1
            return _mock_response(resp_text)

        monkeypatch.setattr(summarizer, "_generate_with_retry", fake_generate)

        result = summarizer._summarize_text_sync("תמלול לדוגמה קצר")

        assert hasattr(result, "exam_critique_log"), (
            "LessonResult must have exam_critique_log field when critique runs"
        )
        assert result.exam_critique_log is not None, "exam_critique_log must not be None"
