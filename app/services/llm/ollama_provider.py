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
from typing import AsyncIterator, Optional

import httpx

from app.services.llm.base import LLMError

logger = logging.getLogger(__name__)


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
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        try:
            async with self._client(timeout or self.timeout) as client:
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
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

        return (data.get("response") or "").strip()

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
