"""LLM Provider interface (port).

Abstract base class defining the contract for LLM communication.
Concrete implementations (e.g., OllamaAdapter) live in infrastructure/.

NO framework imports allowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class LLMProvider(ABC):
    """Port for LLM communication.

    All LLM providers (Ollama, OpenAI, Gemini, etc.) must implement
    this interface, enabling zero-change provider swaps via the adapter pattern.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. ``"ollama"``, ``"gemini"``)."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        *,
        think: bool = True,
    ) -> str:
        """Generate a complete response from the LLM.

        Args:
            messages: Chat messages in OpenAI format [{role, content}].
            model: Model override. None uses the configured default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            think: Whether to enable reasoning/thinking mode. Set to
                False for simple tasks where thinking wastes tokens.

        Returns:
            The complete generated text.

        Raises:
            LLMError: On any communication or processing failure.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Stream a response from the LLM token-by-token.

        Args:
            messages: Chat messages in OpenAI format [{role, content}].
            model: Model override. None uses the configured default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Yields:
            Dicts with ``type`` (``"content"`` or ``"thinking"``) and
            ``text`` (the token string). Models without a thinking
            phase yield only ``"content"`` frames.

        Raises:
            LLMError: On any communication or processing failure.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is reachable and responsive.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict[str, object]]:
        """List available chat models from this provider.

        Returns a list of model descriptors. Each dict contains at minimum:
        - ``id`` (str): model identifier for API calls
        - ``name`` (str): display name
        - ``size_gb`` (float): approximate model size in GB (0 if unknown)

        Embedding-only models should be filtered out.

        Returns:
            List of model info dicts.
        """
        ...

    @abstractmethod
    async def list_embedding_models(self) -> list[dict[str, object]]:
        """List available embedding models from this provider.

        Returns a list of embedding model descriptors. Each dict contains
        at minimum:
        - ``id`` (str): model identifier for API calls
        - ``name`` (str): display name
        - ``size_gb`` (float): approximate model size in GB (0 if unknown)

        Chat-only models should be filtered out.

        Returns:
            List of embedding model info dicts.
        """
        ...
