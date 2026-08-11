"""
Ollama provider — talks to a local Ollama server over HTTP.

In docker-compose the backend reaches the ollama container at
http://ollama:11434 (Docker's internal DNS). The provider covers the three
things the app needs:
  • complete()     — one-shot generation for a Map-Reduce map/reduce step
  • pull()         — stream model-download progress for the Setup Wizard bar
  • list_models()  — what's already pulled (wizard "already installed" check)
  • is_available() — is the server up at all

Tests inject an httpx.MockTransport via `transport=` — no live server, no
extra test dependency.
"""
import json
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from app.services.llm.base import LLMError

logger = logging.getLogger(__name__)


def _unwrap_ollama_fence(text: str) -> str:
    """Remove an outer ```json / ````markdown fence that some models wrap
    inside the Ollama response string.  Works for both streaming and
    non-streaming responses.

    This is a lightweight regex approach — it does NOT handle nested fences
    or embedded backtick sequences; those are extremely rare in practice.
    """
    # Leading fence:  ```json  or  ```markdown  or  ```
    stripped = text.lstrip()
    match = re.match(r"^```(?:json|markdown)?\s*\n", stripped)
    if match:
        stripped = stripped[match.end():]

    # Trailing fence
    if stripped.endswith("\n```"):
        stripped = stripped[:-4].rstrip("\n")
    elif stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3].rstrip("\n")

    return stripped.strip()


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        timeout: float = 600.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Injected only in tests; None → httpx picks the real transport.
        self._transport = transport

    def _client(self, timeout: Optional[float]) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.host,
            timeout=timeout,
            transport=self._transport,
        )

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
    ) -> str:
        # Ollama defaults to streaming NDJSON for /api/generate — we must
        # disable it so the response body is a single complete JSON object.
        # When "format": "json" is set the model may still wrap its output in
        # stray markdown fences (```json … ```).  We strip those before the
        # caller does json.loads on the cleaned text.
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
            async with self._client(timeout or self.timeout) as client:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                result_data = resp.json()

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise LLMError(
                    f"⚠️ המודל '{self.model}' לא מותקן ב-Ollama — הרץ את אשף ההגדרה שוב",
                    detail=f"ollama /api/generate 404 (model '{self.model}' not pulled)",
                ) from exc
            raise LLMError(
                "שגיאה בתקשורת עם Ollama — נסה שוב",
                detail=f"ollama /api/generate {exc.response.status_code}: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "⚠️ לא ניתן להתחבר לשרת Ollama המקומי — ודא שהוא פועל",
                detail=f"ollama transport error: {exc}",
            ) from exc

        # ── Aggressive brace-extraction to ignore preamble/postamble text ──
        raw_text = (result_data.get("response") or "").strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            clean_json = raw_text[start_idx : end_idx + 1]
        else:
            clean_json = raw_text

        # Validate that it's actually parseable JSON before returning
        try:
            json.loads(clean_json)
            return clean_json
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Ollama JSON. Raw text: %s", raw_text)
            raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc

    async def is_available(self) -> bool:
        """True iff the Ollama server answers — used by the Setup Wizard."""
        try:
            async with self._client(5.0) as client:
                resp = await client.get("/api/version")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        """Names of models already pulled locally (e.g. ['gemma2:9b'])."""
        try:
            async with self._client(10.0) as client:
                resp = await client.get("/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(
                "⚠️ לא ניתן לקבל את רשימת המודלים מ-Ollama",
                detail=f"ollama /api/tags error: {exc}",
            ) from exc
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    async def pull(self, model: Optional[str] = None) -> AsyncIterator[dict]:
        """Stream Ollama's model-download progress.

        Yields the raw NDJSON status dicts Ollama emits, e.g.
          {"status": "pulling manifest"}
          {"status": "downloading", "total": 5500000000, "completed": 1200000000}
          {"status": "success"}
        The Setup Wizard maps these to a percentage. Uses no read timeout — a
        multi-GB pull can run for minutes between chunks.
        """
        name = model or self.model
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=None)
        try:
            async with self._client(timeout) as client:
                async with client.stream(
                    "POST", "/api/pull", json={"name": name, "stream": True}
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("ollama pull: skipping non-JSON line: %s", line[:120])
        except httpx.HTTPError as exc:
            raise LLMError(
                f"⚠️ הורדת המודל '{name}' נכשלה — בדוק את החיבור ל-Ollama",
                detail=f"ollama /api/pull error: {exc}",
            ) from exc
