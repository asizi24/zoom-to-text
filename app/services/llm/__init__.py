"""
LLM provider selection for the Smart Summary feature.

`get_summary_provider()` reads the first-boot Setup Wizard's persisted choice
(app_config kv store) and returns a ready provider. Before the wizard runs it
falls back to settings.summary_backend (the GPU box defaults to local Ollama).
"""
import logging
from pathlib import Path

from app.config import settings
from app.services import runtime_config
from app.services.llm.base import LLMError, SummaryProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError",
    "SummaryProvider",
    "OllamaProvider",
    "GeminiProvider",
    "get_summary_provider",
]


def _gcp_credentials_available() -> bool:
    """True if a GCP service-account key file is present — lets a keyless
    GeminiProvider authenticate via Vertex/ADC instead of an API key."""
    creds = settings.google_application_credentials
    return bool(creds) and Path(creds).exists()


async def get_summary_provider() -> SummaryProvider:
    """Construct the configured Smart Summary provider.

    Precedence: the wizard's kv choice, else settings.summary_backend.
    Raises LLMError if a Gemini backend is selected without any key available.
    """
    backend = (await runtime_config.get(runtime_config.KEY_SUMMARY_BACKEND)) \
        or settings.summary_backend
    backend = (backend or "ollama").strip().lower()

    if backend == "gemini":
        api_key = (
            await runtime_config.get(runtime_config.KEY_GEMINI_API_KEY)
        ) or settings.google_api_key
        model = (
            await runtime_config.get(runtime_config.KEY_GEMINI_MODEL)
        ) or settings.gemini_model
        if not api_key and not _gcp_credentials_available():
            raise LLMError(
                "⚠️ לא הוגדר מפתח Gemini — הרץ את אשף ההגדרה",
                detail="summary_backend=gemini but no gemini_api_key/google_api_key set",
            )
        logger.info("Smart Summary provider: gemini (model=%s)", model)
        return GeminiProvider(api_key=api_key, model=model)

    # Default / "ollama"
    model = (
        await runtime_config.get(runtime_config.KEY_OLLAMA_MODEL)
    ) or settings.ollama_model
    logger.info("Smart Summary provider: ollama (model=%s)", model)
    return OllamaProvider(
        host=settings.ollama_host,
        model=model,
        timeout=settings.ollama_timeout,
    )
