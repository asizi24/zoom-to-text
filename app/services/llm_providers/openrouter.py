"""
OpenRouter provider — REST + SSE via httpx.

OpenRouter exposes an OpenAI-compatible /chat/completions endpoint. We use
the `messages` array for both single-shot and multi-turn calls, and SSE
(`stream=true`) for the chat panel.

Audio upload is not supported — caller falls back to one of the WHISPER
modes for transcription, then sends the text via generate_text.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import settings

from .base import LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _classify_http_error(status: int, body: str) -> ProviderError:
    if status in (401, 403):
        return ProviderAuthError(
            provider="openrouter",
            stage="generate",
            code=str(status),
            user_message="🔑 מפתח ה-API של OpenRouter שגוי או לא הוגדר",
            technical_details=body[:300],
        )
    if status == 429:
        return ProviderRateLimitError(
            provider="openrouter",
            stage="generate",
            code="429",
            user_message="⚠️ מכסת ה-API של OpenRouter הוצתה — נסה שוב בעוד כמה דקות",
            technical_details=body[:300],
        )
    return ProviderError(
        provider="openrouter",
        stage="generate",
        code=str(status),
        user_message=f"שגיאה מ-OpenRouter ({status})",
        technical_details=body[:300],
    )


def _contents_to_messages(contents: str | list) -> list[dict]:
    """Convert prompt|contents into OpenAI-style messages."""
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    out = []
    for c in contents:
        # google-genai-style {"role": "user|model", "parts": [{"text": ...}]}
        role = c.get("role", "user")
        if role == "model":
            role = "assistant"
        text_parts = [p.get("text", "") for p in c.get("parts", []) if "text" in p]
        out.append({"role": role, "content": "\n".join(text_parts)})
    return out


class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    supports_audio_upload = False
    supports_streaming = True
    default_model = ""  # populated in __init__ from settings

    def __init__(self) -> None:
        self.default_model = settings.openrouter_model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends these for analytics
            "HTTP-Referer": settings.base_url,
            "X-Title": settings.app_title,
        }

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        url = f"{settings.openrouter_base_url}/chat/completions"
        payload = {
            "model": settings.openrouter_model,
            "messages": _contents_to_messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                if resp.status_code >= 400:
                    raise _classify_http_error(resp.status_code, resp.text)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="openrouter",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ OpenRouter לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )

        return data["choices"][0]["message"]["content"]

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        url = f"{settings.openrouter_base_url}/chat/completions"
        payload = {
            "model": settings.openrouter_model,
            "messages": _contents_to_messages(contents),
            "temperature": temperature,
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=self._headers()
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise _classify_http_error(resp.status_code, body.decode("utf-8", "ignore"))
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {}).get("content")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="openrouter",
                stage="stream_text",
                code="timeout",
                user_message="⏱️ OpenRouter לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
