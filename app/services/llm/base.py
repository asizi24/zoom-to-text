"""
LLM provider abstraction for the Smart Summary Map-Reduce pipeline.

The pipeline (app/services/smart_summary.py) must run identically on a local
Ollama container OR the hosted Gemini API — the choice is a first-boot Setup
Wizard decision, not a code change. Both providers implement one method:

    async def complete(prompt, *, system=None, temperature=..., timeout=...) -> str

Providers stay framework-light: they raise LLMError on transport/model failure
and know nothing about FastAPI or PipelineError. The caller (smart_summary)
translates LLMError into the user-facing PipelineError envelope.
"""
from typing import Optional, Protocol, runtime_checkable


class LLMError(Exception):
    """Provider-level failure (transport error, missing model, timeout).

    Carries a Hebrew user_message for the UI plus a technical detail for logs.
    """

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail


@runtime_checkable
class SummaryProvider(Protocol):
    """Structural type implemented by OllamaProvider and GeminiProvider."""

    name: str
    model: str

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.3,
        timeout: Optional[float] = None,
    ) -> str:
        """Return the model's completion for `prompt` as plain text."""
        ...
