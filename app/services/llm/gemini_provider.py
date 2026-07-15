"""
Gemini provider — the cloud fallback for the Smart Summary Map-Reduce pipeline.

Used on CPU-only machines where a local model is impractical. Deliberately
independent of summarizer._get_client(): the Setup Wizard may store a Gemini
key in the app_config kv table rather than .env, and that key must back Smart
Summary without disturbing the main transcription pipeline's own client.

Clients are cached per API key (constructing a genai.Client is not free).
"""
import asyncio
import logging
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.services.llm.base import LLMError

logger = logging.getLogger(__name__)

# Cache by api_key ("" → Application Default Credentials / Vertex).
_clients: dict[str, genai.Client] = {}


def _get_client(api_key: str) -> genai.Client:
    client = _clients.get(api_key)
    if client is None:
        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        _clients[api_key] = client
    return client


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float = 600.0):
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        client = _get_client(self.api_key)
        try:
            async with asyncio.timeout(timeout or self.timeout):
                resp = await client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
        except TimeoutError as exc:
            raise LLMError(
                "⏱️ Gemini לא הגיב בזמן — נסה שוב",
                detail=f"gemini generate_content timed out after {timeout or self.timeout}s",
            ) from exc
        except genai_errors.APIError as exc:
            code = exc.code or 0
            if code == 429:
                raise LLMError(
                    "⚠️ מכסת ה-API של Gemini הוצתה — נסה שוב בעוד כמה דקות",
                    detail=str(exc),
                ) from exc
            raise LLMError(
                "שגיאה בתקשורת עם Gemini — נסה שוב",
                detail=str(exc),
            ) from exc

        return (resp.text or "").strip()
