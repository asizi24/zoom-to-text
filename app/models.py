"""
Pydantic schemas for request/response validation.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingMode(str, Enum):
    # Upload audio directly to Gemini — fast (2-3 min), requires internet
    GEMINI_DIRECT = "gemini_direct"
    # Transcribe locally with Whisper, then send text to Gemini — slower, more private
    WHISPER_LOCAL = "whisper_local"
    # Transcribe via OpenAI Whisper API, then send text to Gemini
    # Requires OPENAI_API_KEY. Audio is preprocessed (silence removal + chunking).
    WHISPER_API = "whisper_api"
    # Transcribe locally with ivrit-ai's Hebrew-tuned Whisper model.
    # Higher accuracy on spoken Hebrew than vanilla Whisper, but slower than
    # GEMINI_DIRECT. Model is downloaded on first use and cached on disk.
    IVRIT_AI = "ivrit_ai"


# ── Result schemas ──────────────────────────────────────────────────────────────


class QuizQuestion(BaseModel):
    question: str
    options: list[str]  # ["א. ...", "ב. ...", "ג. ...", "ד. ..."]
    correct_answer: str  # Must match one of the options exactly
    explanation: str = ""
    bloom_level: Optional[int] = None  # Bloom level 1-6; None = not labeled


class Chapter(BaseModel):
    title: str
    content: str
    key_points: list[str] = []
    start_time: Optional[str] = (
        None  # [MM:SS] of when this topic begins; None = not available
    )


class Flashcard(BaseModel):
    front: str  # Question or concept prompt
    back: str  # 1-3 sentence answer/explanation
    tags: list[str] = []  # Topic/chapter tags for Anki filtering


# ── Extraction artifacts (Task 1.1 schema upgrade) ──────────────────────────────


class ActionItem(BaseModel):
    owner: str
    task: str
    deadline: Optional[str] = None  # free-form: "EOW", "2026-05-01"
    priority: Optional[str] = None  # "high" | "medium" | "low"
    source_quote: Optional[str] = None


class Decision(BaseModel):
    decision: str
    context: Optional[str] = None
    stakeholders: list[str] = []
    source_quote: Optional[str] = None


class OpenQuestion(BaseModel):
    question: str
    raised_by: Optional[str] = None
    context: Optional[str] = None


class PerSpeakerSentiment(BaseModel):
    speaker: str  # "Speaker A" or named when known
    sentiment: str  # "positive" | "neutral" | "negative" | "mixed"
    rationale: Optional[str] = None


class ToneShift(BaseModel):
    # 'from'/'to' are Python keywords / builtins, so the JSON keys are aliased
    # while the Python attributes use _tone suffix.
    model_config = {"populate_by_name": True}

    at: str  # rough timestamp or paragraph anchor
    from_tone: str = Field(..., alias="from")
    to_tone: str = Field(..., alias="to")
    trigger: Optional[str] = None


class SentimentAnalysis(BaseModel):
    overall_tone: str
    per_speaker_sentiment: list[PerSpeakerSentiment] = []
    shifts_in_tone: list[ToneShift] = []


class Objection(BaseModel):
    objection: str
    raised_by: Optional[str] = None
    response_given: Optional[str] = None
    resolved: Optional[bool] = None


class RawLLMResponse(BaseModel):
    """Raw LLM output captured for offline debugging when LLM_DEBUG_RAW_RESPONSES=true."""

    summary_call: Optional[str] = None
    extraction_call: Optional[str] = None


class KeyTerm(BaseModel):
    term: str
    definition: str
    context: Optional[str] = None


class Highlight(BaseModel):
    """A memorable / quotable / 'aha' moment extracted from the recording."""

    quote: str  # verbatim quote, ≤200 chars
    why: str  # 1-sentence explanation of why this matters
    timestamp: Optional[str] = None  # [MM:SS] or "0:34:12" when available
    speaker: Optional[str] = None  # "Speaker A" or named when known


class MindMapNode(BaseModel):
    """One node in a hierarchical mind-map (recursive)."""

    label: str  # short text (≤80 chars)
    children: list["MindMapNode"] = []


class MindMap(BaseModel):
    """A hierarchical mind-map rooted at a single topic."""

    root: MindMapNode


# Pydantic v2 needs an explicit rebuild for self-referencing list["MindMapNode"]
MindMapNode.model_rebuild()


class LessonResult(BaseModel):
    transcript: Optional[str] = None  # Raw transcript (only in WHISPER_LOCAL mode)
    summary: str = ""
    chapters: list[Chapter] = []
    quiz: list[QuizQuestion] = []
    flashcards: list[Flashcard] = []  # 15-25 spaced-repetition cards
    language: str = "he"
    # Critique pipeline debug log — populated when ENABLE_EXAM_CRITIQUE=True
    exam_critique_log: Optional[dict] = None
    # ── Task 1.1 schema upgrade — all optional, default-empty ──
    content_type: Optional[str] = None  # "lecture" | "meeting" | "discussion"
    action_items: list[ActionItem] = []
    decisions: list[Decision] = []
    open_questions: list[OpenQuestion] = []
    sentiment_analysis: Optional[SentimentAnalysis] = None
    objections_tracked: list[Objection] = []
    raw_llm_response: Optional[RawLLMResponse] = None
    # ── Task 1.2 Gemini-based diarization (WHISPER paths only) ──
    diarized_transcript: Optional[str] = None  # transcript with "Speaker A:" anchors
    speaker_map: Optional[dict[str, str]] = (
        None  # {"Speaker A": "Asaf", ...} when names detected
    )
    # ── Key Terms Glossary — important vocabulary extracted from the recording ──
    key_terms: list[KeyTerm] = []
    # ── Smart Highlights — quotable / memorable moments (Batch B1) ──
    highlights: list[Highlight] = []
    # ── Mind Map — hierarchical topic structure (Batch B1, lazy-generated) ──
    mindmap: Optional[MindMap] = None


# ── Cram Guide schemas (Multi-Lecture Study Guide) ──────────────────────────────


class CramChapter(BaseModel):
    title: str
    summary: str
    key_concepts: list[str] = []


class CramGuideResult(BaseModel):
    overall_summary: str = ""
    key_themes: list[str] = []
    chapters: list[CramChapter] = []
    quiz: list[QuizQuestion] = []
    lecture_count: int = 0


class CramGuideRequest(BaseModel):
    task_ids: list[str] = Field(..., min_length=2, max_length=10)


# ── API request/response schemas ────────────────────────────────────────────────


class TaskCreate(BaseModel):
    url: str = Field(..., description="Zoom recording URL")
    mode: ProcessingMode = ProcessingMode.GEMINI_DIRECT
    # Netscape-format cookie string extracted by the Chrome extension.
    # Required for private/institutional Zoom recordings (e.g. admin-ort-org-il.zoom.us).
    cookies: Optional[str] = Field(
        None, description="Zoom session cookies (Netscape format)"
    )
    language: str = Field(
        "he", description="Audio language hint for Whisper (he, en, auto)"
    )


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = Field(0, ge=0, le=100)
    message: str = ""
    created_at: str
    url: Optional[str] = None
    result: Optional[LessonResult] = None
    error: Optional[str] = None
    # Structured error info (Task 1.5). Keys: stage, code, user_message,
    # technical_details. None for tasks that succeeded or failed before
    # the schema migration.
    error_details: Optional[dict] = None
    # True iff the server has a playable audio file for this task (Feature 7)
    has_audio: bool = False
    # ISO 8601 UTC timestamp of when the task entered status='failed'.
    # Used by the auto-cleanup task (24h TTL) and the UI countdown.
    # None for tasks that never failed.
    failed_at: Optional[str] = None
    # User-editable plain-text notes attached to this task.
    # Empty string when never edited; capped at NOTES_MAX_LEN chars on PUT.
    notes: str = ""


class NotesUpdate(BaseModel):
    """Request body for PUT /api/tasks/{id}/notes."""

    notes: str = Field("", max_length=50_000)


class AskAcrossRequest(BaseModel):
    """Request body for POST /api/ask — Ask-Across-Lectures."""

    question: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(20, ge=1, le=50)


class FlashcardReview(BaseModel):
    """Request body for POST /api/tasks/{id}/flashcards/{idx}/review."""

    grade: str = Field(..., pattern="^(again|hard|good|easy)$")


class TutorRequest(BaseModel):
    """Request body for POST /api/tasks/{id}/tutor — Socratic AI tutor."""

    question: str = Field(..., min_length=1, max_length=2000)


class AudioClipCreate(BaseModel):
    """Request body for POST /api/tasks/{id}/clips — B4 clip sharing."""

    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)
    label: str = Field("", max_length=200)


class WebhookCreate(BaseModel):
    """Request body for POST /api/webhooks — B5 outgoing notifications."""

    kind: str = Field(..., pattern=r"^(slack|discord)$")
    url: str = Field(..., min_length=10, max_length=2000)


class WebhookUpdate(BaseModel):
    """Request body for PATCH /api/webhooks/{id}."""

    enabled: bool


class TaskShareCreate(BaseModel):
    """Request body for POST /api/tasks/{id}/shares — B5 cohort sharing."""

    email: str = Field(..., min_length=3, max_length=320)


# ── B6.2: lesson recipes ────────────────────────────────────────────────────

class RecipeCreate(BaseModel):
    """Request body for POST /api/recipes — saved processing preset."""

    name: str = Field(..., min_length=1, max_length=60)
    mode: str = Field(..., pattern=r"^(gemini_direct|whisper_local|whisper_api)$")
    language: str = Field("he", min_length=2, max_length=8)
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=500)


class RecipeUpdate(BaseModel):
    """Request body for PATCH /api/recipes/{id}."""

    name: Optional[str] = Field(None, min_length=1, max_length=60)
    mode: Optional[str] = Field(
        None, pattern=r"^(gemini_direct|whisper_local|whisper_api)$"
    )
    language: Optional[str] = Field(None, min_length=2, max_length=8)
    tags: Optional[list[str]] = None
    notes: Optional[str] = Field(None, max_length=500)


# ── B6.3: slide deck + alignment ────────────────────────────────────────────

class SlideInput(BaseModel):
    """A single slide as supplied by the client (text extracted browser-side
    from a PDF, or manually authored)."""

    page_index: int = Field(..., ge=0, le=10_000)
    title: str = Field("", max_length=300)
    body: str = Field("", max_length=4000)


class SlideDeckUpload(BaseModel):
    """Request body for POST /api/tasks/{id}/slides — replaces the existing deck."""

    slides: list[SlideInput] = Field(..., max_length=500)


class SlideAlignmentUpdate(BaseModel):
    """Request body for PATCH /api/tasks/{id}/slides/{slide_id}."""

    chapter_index: Optional[int] = Field(None, ge=0, le=200)


# ── B6.4: AI podcast companion ──────────────────────────────────────────────

class PodcastTurn(BaseModel):
    """A single line in the two-host podcast script."""

    speaker: str = Field(..., pattern=r"^(host_a|host_b)$")
    text: str


class PodcastScriptResponse(BaseModel):
    """Response body for GET /api/tasks/{id}/podcast-script."""

    task_id: str
    turns: list[PodcastTurn]
    model: str
