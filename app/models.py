"""
Pydantic schemas for request/response validation.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING      = "pending"
    DOWNLOADING  = "downloading"
    TRANSCRIBING = "transcribing"
    SUMMARIZING  = "summarizing"
    COMPLETED    = "completed"
    FAILED       = "failed"


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
    question:       str
    options:        list[str]        # ["א. ...", "ב. ...", "ג. ...", "ד. ..."]
    correct_answer: str              # Must match one of the options exactly
    explanation:    str = ""


class Chapter(BaseModel):
    title:      str
    content:    str
    key_points: list[str] = []


class Flashcard(BaseModel):
    front: str                  # Question or concept prompt
    back:  str                  # 1-3 sentence answer/explanation
    tags:  list[str] = []       # Topic/chapter tags for Anki filtering


# ── Extraction artifacts (Task 1.1 schema upgrade) ──────────────────────────────

class ActionItem(BaseModel):
    owner:        str
    task:         str
    deadline:     Optional[str] = None  # free-form: "EOW", "2026-05-01"
    priority:     Optional[str] = None  # "high" | "medium" | "low"
    source_quote: Optional[str] = None


class Decision(BaseModel):
    decision:     str
    context:      Optional[str] = None
    stakeholders: list[str] = []
    source_quote: Optional[str] = None


class OpenQuestion(BaseModel):
    question:  str
    raised_by: Optional[str] = None
    context:   Optional[str] = None


class PerSpeakerSentiment(BaseModel):
    speaker:   str                       # "Speaker A" or named when known
    sentiment: str                       # "positive" | "neutral" | "negative" | "mixed"
    rationale: Optional[str] = None


class ToneShift(BaseModel):
    # 'from'/'to' are Python keywords / builtins, so the JSON keys are aliased
    # while the Python attributes use _tone suffix.
    model_config = {"populate_by_name": True}

    at:        str                       # rough timestamp or paragraph anchor
    from_tone: str = Field(..., alias="from")
    to_tone:   str = Field(..., alias="to")
    trigger:   Optional[str] = None


class SentimentAnalysis(BaseModel):
    overall_tone:          str
    per_speaker_sentiment: list[PerSpeakerSentiment] = []
    shifts_in_tone:        list[ToneShift] = []


class Objection(BaseModel):
    objection:      str
    raised_by:      Optional[str] = None
    response_given: Optional[str] = None
    resolved:       Optional[bool] = None


class RawLLMResponse(BaseModel):
    """Raw LLM output captured for offline debugging when LLM_DEBUG_RAW_RESPONSES=true."""
    summary_call:    Optional[str] = None
    extraction_call: Optional[str] = None


class LessonResult(BaseModel):
    transcript:        Optional[str] = None  # Raw transcript (only in WHISPER_LOCAL mode)
    summary:           str = ""
    chapters:          list[Chapter] = []
    quiz:              list[QuizQuestion] = []
    flashcards:        list[Flashcard] = []   # 15-25 spaced-repetition cards
    language:          str = "he"
    # Critique pipeline debug log — populated when ENABLE_EXAM_CRITIQUE=True
    exam_critique_log: Optional[dict] = None
    # ── Task 1.1 schema upgrade — all optional, default-empty ──
    content_type:       Optional[str] = None  # "lecture" | "meeting" | "discussion"
    action_items:       list[ActionItem] = []
    decisions:          list[Decision] = []
    open_questions:     list[OpenQuestion] = []
    sentiment_analysis: Optional[SentimentAnalysis] = None
    objections_tracked: list[Objection] = []
    raw_llm_response:   Optional[RawLLMResponse] = None
    # ── Task 1.2 Gemini-based diarization (WHISPER paths only) ──
    diarized_transcript: Optional[str]            = None  # transcript with "Speaker A:" anchors
    speaker_map:         Optional[dict[str, str]] = None  # {"Speaker A": "Asaf", ...} when names detected


# ── API request/response schemas ────────────────────────────────────────────────

class TaskCreate(BaseModel):
    url:      str = Field(..., description="Zoom recording URL")
    mode:     ProcessingMode = ProcessingMode.GEMINI_DIRECT
    # Netscape-format cookie string extracted by the Chrome extension.
    # Required for private/institutional Zoom recordings (e.g. admin-ort-org-il.zoom.us).
    cookies:  Optional[str] = Field(None, description="Zoom session cookies (Netscape format)")
    language: str = Field("he", description="Audio language hint for Whisper (he, en, auto)")


class TaskResponse(BaseModel):
    task_id:    str
    status:     TaskStatus
    progress:   int = Field(0, ge=0, le=100)
    message:    str = ""
    created_at: str
    url:        Optional[str] = None
    result:     Optional[LessonResult] = None
    error:      Optional[str] = None
    # Structured error info (Task 1.5). Keys: stage, code, user_message,
    # technical_details. None for tasks that succeeded or failed before
    # the schema migration.
    error_details: Optional[dict] = None
    # True iff the server has a playable audio file for this task (Feature 7)
    has_audio:  bool = False
