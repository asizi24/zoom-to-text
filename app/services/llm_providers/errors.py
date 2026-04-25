"""
Exception hierarchy for LLM providers.

ProviderError is the root; subclasses convey the kind of failure so callers
(or pipeline orchestrators) can decide whether to surface the user_message,
retry, or fail fast.
"""
from __future__ import annotations


class ProviderError(RuntimeError):
    """Base class for all LLM provider failures."""

    def __init__(
        self,
        *,
        provider: str,
        stage: str,
        code: str,
        user_message: str,
        technical_details: str = "",
    ) -> None:
        self.provider = provider
        self.stage = stage
        self.code = code
        self.user_message = user_message
        self.technical_details = technical_details
        super().__init__(user_message)


class ProviderUnsupportedError(ProviderError):
    """The provider does not support the requested feature (e.g. audio upload)."""


class ProviderRateLimitError(ProviderError):
    """Quota exceeded / 429 — caller should back off or surface to user."""


class ProviderTimeoutError(ProviderError):
    """Provider did not respond within the configured timeout."""


class ProviderAuthError(ProviderError):
    """API key missing or rejected (401/403). Terminal — do not retry."""
