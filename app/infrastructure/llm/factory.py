"""LLM provider factory.

Creates the appropriate LLM provider adapter based on application settings.
Supports Ollama (self-hosted, default) and Google Gemini (per-tenant API key).

Usage::

    # Startup: create the default Ollama provider
    llm_provider = get_llm_provider(settings)

    # Per-request: create a Gemini provider for a specific tenant
    gemini = get_gemini_provider(api_key="AIza...", model="gemini-2.5-flash")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.infrastructure.llm.gemini_adapter import GeminiAdapter
from app.infrastructure.llm.ollama_adapter import OllamaAdapter

if TYPE_CHECKING:
    from app.config import Settings
    from app.domain.interfaces.llm_provider import LLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Create and return the default (Ollama) LLM provider.

    Used at application startup to populate ``app.state.llm_provider``.
    Ollama is the default, self-hosted provider.
    """
    return OllamaAdapter(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_chat_model,
        cf_client_id=settings.cf_ollama_id,
        cf_client_secret=settings.cf_ollama_secret,
    )


def get_gemini_provider(
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> LLMProvider:
    """Create a per-request Gemini provider with a tenant's API key.

    Each call returns a new adapter instance — providers are NOT shared
    between tenants or cached across requests.

    Args:
        api_key: The decrypted Gemini API key from tenant config.
        model: The Gemini model identifier (default: gemini-2.5-flash).

    Returns:
        A GeminiAdapter configured for the given tenant.
    """
    return GeminiAdapter(api_key=api_key, default_model=model)
