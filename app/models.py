"""
Pydantic schemas for request/response validation.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Enums ───────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING      = "pending"
    DOWNLOADING  = "downloading"
    TRANSCRIBING = "transcribing"
    SUMMARIZING  = "summarizing"
    COMPLETED    = "completed"
    FAILED       = "failed"
    # User-requested abort. Terminal, but retryable (POST /tasks/{id}/retry)
    # exactly like FAILED. Cooperative: the worker/transcriber poll the
    # cancellation registry and stop at the next safe checkpoint.
    CANCELLED    = "cancelled"


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
    # Transcript only — local ivrit-ai transcription, NO Gemini call at all.
    # Result contains just the transcript; summary/chapters/quiz stay empty.
    TRANSCRIPTION_ONLY = "transcription_only"


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


class LessonResult(BaseModel):
    transcript:        Optional[str] = None  # Raw transcript (Whisper modes + TRANSCRIPTION_ONLY)
    summary:           str = ""
    chapters:          list[Chapter] = []
    quiz:              list[QuizQuestion] = []
    flashcards:        list[Flashcard] = []   # 15-25 spaced-repetition cards
    language:          str = "he"
    # Critique pipeline debug log — populated when ENABLE_EXAM_CRITIQUE=True
    exam_critique_log: Optional[dict] = None


# ── API request/response schemas ────────────────────────────────────────────────

class TaskCreate(BaseModel):
    url:      str = Field(..., max_length=4096, description="Zoom recording URL")
    mode:     ProcessingMode = ProcessingMode.GEMINI_DIRECT
    # Netscape-format cookie string extracted by the Chrome extension.
    # Required for private/institutional Zoom recordings (e.g. admin-ort-org-il.zoom.us).
    cookies:  Optional[str] = Field(None, description="Zoom session cookies (Netscape format)")
    language: str = Field("he", description="Audio language hint for Whisper (he, en, auto)")

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        # The URL is handed to yt-dlp, whose generic extractor will fetch
        # almost anything — reject non-web schemes (file:, data:, javascript:)
        # at the front door instead of relying on downstream behavior.
        v = v.strip()
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v


class TaskResponse(BaseModel):
    task_id:    str
    status:     TaskStatus
    progress:   int = Field(0, ge=0, le=100)
    message:    str = ""
    created_at: str
    url:        Optional[str] = None
    result:     Optional[LessonResult] = None
    error:      Optional[str] = None
    # True iff the server has a playable audio file for this task (Feature 7)
    has_audio:  bool = False
