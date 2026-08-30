import json
import logging
from pathlib import Path
from typing import Optional

import httpx
from app.config import settings
from app.services.llm.base import LLMError, SummaryProvider

logger = logging.getLogger(__name__)

def _unwrap_ollama_fence(text: str) -> str:
    """Strips markdown code fences from Ollama's JSON response."""
    return text.replace("```json", "").replace("```", "").strip()

class OllamaProvider:
    """Ollama implementation of SummaryProvider."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: float = 60.0,
    ):
        self.host = host.rstrip("/")
        self.app_host = host.rstrip("/") # just for safety in debugging
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
        """
        Ollama implementation of the SummaryProvider protocol.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,                  # force non-streaming response
            "format": "json",
            "max_tokens": 8192,
            "options": {"temperature": temperature, "num_predict": 8192},
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.post(f"{self.host}/api/generate", json=payload)
                resp.raise_for_status()
                result_data = resp.json()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise LLMError(
                    f"⚠️ המודל '{self.model}' לא מותקן ב-OLLAMA — הרץ את אשף ההגדרה שוב",
                    detail=f"ollama /api/generate 404 (model '{self.model}' not pulled)",
                ) from exc
            raise LLMError(
                "שגיאה בתקשורת עם Ollama — נסה שוב",
                detail=f"ollama /api/generate {exc.response.status_code}: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "⚠️ לא ניתן להתחבר לשרת Ollma המקומי — ודא שהוא פועל",
                detail=f"ollama transport error: {exc}",
            ) from exc

        # ── Aggressive brace-extraction to ignore preamble/postamble text ──
        raw_text = _unwrap_ollama_fence((result_data.get("response") or "").strip())

        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            clean_json = raw_text[start_idx : end_idx + 1]
        else:
            raise ValueError(f"Ollama response does not contain a valid JSON object. Raw text: {raw_text[:100]}...")

        # Validate that it's actually parseable JSON before returning
        try:
            json.loads(clean_json)
            return clean_json
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Ollama JSON. Raw text: %s", raw_text)
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc
