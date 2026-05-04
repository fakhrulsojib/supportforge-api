"""LLM provider factory.

Creates the appropriate LLM provider adapter based on application settings.
Currently supports Ollama only; extensible to OpenAI, Anthropic, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.infrastructure.llm.ollama_adapter import OllamaAdapter

if TYPE_CHECKING:
    from app.config import Settings
    from app.domain.interfaces.llm_provider import LLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Create and return the configured LLM provider.

    Currently always returns an OllamaAdapter. When additional providers
    are needed, this factory will inspect ``settings.llm_provider`` to
    select the appropriate adapter.
    """
    return OllamaAdapter(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_chat_model,
        cf_client_id=settings.cf_ollama_id,
        cf_client_secret=settings.cf_ollama_secret,
    )
