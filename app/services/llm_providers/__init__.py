"""
Factory for LLM providers.

`get_provider()` returns a singleton matching `settings.llm_provider`. The
singleton is rebuilt only when `_reset_provider_cache()` is called (used by
tests; production code calls get_provider() repeatedly without overhead).
"""
from __future__ import annotations

from app.config import settings

from .base import AudioRef, LLMProvider
from .errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnsupportedError,
)

__all__ = [
    "AudioRef",
    "LLMProvider",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnsupportedError",
    "get_provider",
]


_cached: LLMProvider | None = None


def _reset_provider_cache() -> None:
    """Clear the cached provider — used by tests when changing settings."""
    global _cached
    _cached = None


def get_provider() -> LLMProvider:
    """Return the active LLMProvider instance based on settings.llm_provider."""
    global _cached
    if _cached is not None:
        return _cached

    name = settings.llm_provider
    if name == "gemini":
        from .gemini import GeminiProvider
        _cached = GeminiProvider()
    elif name == "openrouter":
        from .openrouter import OpenRouterProvider
        _cached = OpenRouterProvider()
    elif name == "ollama":
        from .ollama import OllamaProvider
        _cached = OllamaProvider()
    else:
        raise ValueError(f"Unknown llm_provider: {name!r}")

    return _cached
