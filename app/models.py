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


class LessonResult(BaseModel):
    transcript: Optional[str] = None  # Raw transcript (only in WHISPER_LOCAL mode)
    summary:    str = ""
    chapters:   list[Chapter] = []
    quiz:       list[QuizQuestion] = []
    language:   str = "he"


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
