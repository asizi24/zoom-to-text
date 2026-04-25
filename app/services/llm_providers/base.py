"""
Abstract LLM provider interface + shared helpers.

Every concrete provider (Gemini, OpenRouter, Ollama) inherits from
LLMProvider. The interface intentionally returns RAW MODEL TEXT for
generation methods — JSON parsing stays in summarizer.py because it is
specific to the LessonResult schema and has accumulated defensive logic
that would not survive being scattered across providers.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar

from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderUnsupportedError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Audio reference ────────────────────────────────────────────────────────────

@dataclass
class AudioRef:
    """Reference to an audio file living in a provider's storage."""

    provider_name: str
    provider_specific_id: str
    raw: Any = None  # original SDK object, kept for cleanup calls


# ── Abstract provider ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    name: str
    supports_audio_upload: bool
    supports_streaming: bool
    default_model: str

    # ---- text generation -----------------------------------------------------

    @abstractmethod
    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        """Single-shot generation. Returns raw model text. Caller parses."""

    @abstractmethod
    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Yield text deltas. Used by chat-with-recording streaming."""

    # ---- audio upload (Gemini-only by default) ------------------------------

    async def upload_audio(self, path: str) -> AudioRef:
        raise ProviderUnsupportedError(
            provider=self.name,
            stage="upload_audio",
            code="unsupported",
            user_message=f"❌ ה-LLM הנוכחי ({self.name}) לא תומך בהעלאת אודיו. בחר מצב עיבוד אחר.",
        )

    async def generate_text_with_audio(
        self,
        audio_ref: AudioRef,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        raise ProviderUnsupportedError(
            provider=self.name,
            stage="generate_text_with_audio",
            code="unsupported",
            user_message=f"❌ ה-LLM הנוכחי ({self.name}) לא תומך בעיבוד אודיו ישיר.",
        )

    async def cleanup_audio(self, audio_ref: AudioRef) -> None:
        """Default no-op; Gemini overrides to delete the Files-API entry."""
        return None


# ── Shared retry helper ───────────────────────────────────────────────────────

async def _with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """
    Exponential-backoff retry for an async function.

    Retries on every ProviderError EXCEPT ProviderAuthError (terminal) and
    ProviderUnsupportedError (terminal). Re-raises on final attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except (ProviderAuthError, ProviderUnsupportedError):
            raise
        except ProviderError as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            logger.warning(
                f"Provider error (attempt {attempt + 1}/{max_retries}): "
                f"{exc.code} — retrying in {wait}s"
            )
            await asyncio.sleep(wait)
    # Unreachable — loop either returns or raises
    raise last_exc or RuntimeError("retry loop exited unexpectedly")
