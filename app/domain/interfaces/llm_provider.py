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

    All LLM providers (Ollama, OpenAI, Anthropic, etc.) must implement
    this interface, enabling zero-change provider swaps via the adapter pattern.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a complete response from the LLM.

        Args:
            messages: Chat messages in OpenAI format [{role, content}].
            model: Model override. None uses the configured default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

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
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM token-by-token.

        Args:
            messages: Chat messages in OpenAI format [{role, content}].
            model: Model override. None uses the configured default.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Yields:
            Token strings as they are generated.

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
