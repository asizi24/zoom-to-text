"""
Gemini provider — a thin facade over the existing helpers in
`app.services.summarizer`. The goal of this module is to expose Gemini
through the `LLMProvider` interface WITHOUT changing summarizer.py — every
existing test that monkeypatches summarizer internals continues to work.

We keep the sync google-genai SDK on a thread executor (as summarizer.py
already does) and translate Gemini-specific exceptions into ProviderError
subclasses on the way out.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from app.config import settings

from .base import AudioRef, LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _classify_gemini_exception(exc: Exception) -> ProviderError:
    """Map a raw Gemini SDK / RuntimeError into a ProviderError subclass."""
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "quota" in low or "rate limit" in low or "מכסת" in msg:
        return ProviderRateLimitError(
            provider="gemini",
            stage="generate",
            code="rate_limit",
            user_message="⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות",
            technical_details=msg,
        )
    if "401" in msg or "403" in msg or "api key" in low:
        return ProviderAuthError(
            provider="gemini",
            stage="generate",
            code="auth",
            user_message="🔑 מפתח ה-API של Gemini שגוי או לא הוגדר",
            technical_details=msg,
        )
    return ProviderError(
        provider="gemini",
        stage="generate",
        code="unknown",
        user_message=f"שגיאה ב-Gemini: {msg[:200]}",
        technical_details=msg,
    )


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_audio_upload = True
    supports_streaming = True
    default_model = settings.gemini_model

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        """Delegate to summarizer._generate_with_retry on a thread executor."""
        from app.services import summarizer  # local import — avoids cycle

        loop = asyncio.get_running_loop()

        def _call() -> str:
            try:
                client = summarizer._get_client()
                resp = summarizer._generate_with_retry(client, prompt)
                return summarizer._response_text(resp)
            except Exception as exc:
                raise _classify_gemini_exception(exc)

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _call), timeout=timeout
            )
        except asyncio.TimeoutError:
            raise ProviderTimeoutError(
                provider="gemini",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ Gemini לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        """Delegate to the existing streaming code in summarizer.py."""
        from app.services import summarizer

        client = summarizer._get_client()

        loop = asyncio.get_running_loop()
        import queue as _q
        q: _q.Queue = _q.Queue()

        def _run() -> None:
            try:
                for chunk in client.models.generate_content_stream(
                    model=settings.gemini_model,
                    contents=contents,
                ):
                    text = getattr(chunk, "text", None)
                    if text:
                        q.put(text)
            except Exception as exc:
                q.put(_classify_gemini_exception(exc))
            finally:
                q.put(None)

        loop.run_in_executor(None, _run)

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def upload_audio(self, path: str) -> AudioRef:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _upload():
            client = summarizer._get_client()
            f = client.files.upload(file=path)
            # Poll until processed (mirrors summarizer._summarize_audio_sync)
            import time
            waited = 0
            while f.state.name == "PROCESSING":
                if waited >= 300:
                    raise RuntimeError("⏱️ Gemini לא סיים לעבד את קובץ האודיו תוך 5 דקות")
                time.sleep(5)
                waited += 5
                f = client.files.get(name=f.name)
            if f.state.name == "FAILED":
                raise RuntimeError("❌ Gemini נכשל בעיבוד קובץ האודיו")
            return f

        try:
            f = await loop.run_in_executor(None, _upload)
        except Exception as exc:
            raise _classify_gemini_exception(exc)

        return AudioRef(provider_name="gemini", provider_specific_id=f.name, raw=f)

    async def generate_text_with_audio(
        self,
        audio_ref: AudioRef,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _call() -> str:
            try:
                client = summarizer._get_client()
                resp = summarizer._generate_with_retry(client, [prompt, audio_ref.raw])
                return summarizer._response_text(resp)
            except Exception as exc:
                raise _classify_gemini_exception(exc)

        return await loop.run_in_executor(None, _call)

    async def cleanup_audio(self, audio_ref: AudioRef) -> None:
        from app.services import summarizer
        loop = asyncio.get_running_loop()

        def _delete() -> None:
            try:
                client = summarizer._get_client()
                client.files.delete(name=audio_ref.provider_specific_id)
            except Exception as exc:
                logger.warning(f"Gemini audio cleanup failed: {exc}")

        await loop.run_in_executor(None, _delete)
