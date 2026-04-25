"""
Ollama provider — REST + NDJSON streaming via httpx.

Ollama runs models locally (llama3.1, mistral, qwen, etc.). We hit
/api/generate for single-shot text and /api/chat with stream=true for the
chat panel. NDJSON: each line is a JSON object with `response` (text delta)
and `done` (final flag).

This provider is code-only on the current Fly.io deploy — Ollama needs
local GPU / sizable RAM. It's wired to be ready for a future home-server
deploy with no further code changes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from app.config import settings

from .base import LLMProvider
from .errors import (
    ProviderError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def _contents_to_messages(contents: str | list) -> list[dict]:
    """google-genai-style contents → Ollama chat messages."""
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    out = []
    for c in contents:
        role = c.get("role", "user")
        if role == "model":
            role = "assistant"
        parts = [p.get("text", "") for p in c.get("parts", []) if "text" in p]
        out.append({"role": role, "content": "\n".join(parts)})
    return out


class OllamaProvider(LLMProvider):
    name = "ollama"
    supports_audio_upload = False
    supports_streaming = True
    default_model = ""

    def __init__(self) -> None:
        self.default_model = settings.ollama_model

    async def generate_text(
        self,
        prompt: str | list,
        *,
        max_tokens: int = 65536,
        temperature: float = 0.3,
        timeout: float = 600.0,
    ) -> str:
        url = f"{settings.ollama_base_url}/api/generate"
        # Ollama /api/generate takes a flat prompt; for multi-turn use /api/chat
        if isinstance(prompt, list):
            messages = _contents_to_messages(prompt)
            prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        else:
            prompt_text = prompt

        payload = {
            "model": settings.ollama_model,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    raise ProviderError(
                        provider="ollama",
                        stage="generate",
                        code=str(resp.status_code),
                        user_message=f"שגיאה מ-Ollama ({resp.status_code})",
                        technical_details=resp.text[:300],
                    )
                data = resp.json()
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="ollama",
                stage="generate_text",
                code="timeout",
                user_message="⏱️ Ollama לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
        except httpx.ConnectError as exc:
            raise ProviderError(
                provider="ollama",
                stage="generate_text",
                code="connect",
                user_message="❌ לא הצלחתי להתחבר לשרת Ollama — ודא שהוא רץ",
                technical_details=str(exc),
            )

        return data.get("response", "")

    async def stream_text(
        self,
        contents: list[dict],
        *,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> AsyncIterator[str]:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": _contents_to_messages(contents),
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise ProviderError(
                            provider="ollama",
                            stage="stream_text",
                            code=str(resp.status_code),
                            user_message=f"שגיאה מ-Ollama ({resp.status_code})",
                            technical_details=body.decode("utf-8", "ignore")[:300],
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("done"):
                            break
                        delta = chunk.get("message", {}).get("content")
                        if delta:
                            yield delta
        except httpx.TimeoutException:
            raise ProviderTimeoutError(
                provider="ollama",
                stage="stream_text",
                code="timeout",
                user_message="⏱️ Ollama לא הגיב בזמן — נסה שוב",
                technical_details=f"timeout={timeout}s",
            )
