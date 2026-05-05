"""Ollama LLM adapter using OpenAI-compatible API.

Connects to a self-hosted Ollama instance behind Cloudflare Access.
Authentication is handled by injecting CF service token headers
into every request via httpx.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import structlog
from openai import AsyncOpenAI

from app.core.exceptions import LLMError
from app.domain.interfaces.llm_provider import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)


class OllamaAdapter(LLMProvider):
    """Concrete LLM provider backed by a self-hosted Ollama instance.

    Uses the OpenAI-compatible API endpoint (``/v1/chat/completions``)
    with Cloudflare Access service token headers for authentication.

    Attributes:
        base_url: Ollama API base URL.
        default_model: Default chat model name.
        _client: AsyncOpenAI client instance.
    """

    def __init__(
        self,
        base_url: str,
        default_model: str,
        cf_client_id: str = "",
        cf_client_secret: str = "",
    ) -> None:
        self.base_url = base_url
        self.default_model = default_model

        # Build httpx client with Cloudflare Access headers
        headers: dict[str, str] = {}
        if cf_client_id and cf_client_secret:
            headers["CF-Access-Client-Id"] = cf_client_id
            headers["CF-Access-Client-Secret"] = cf_client_secret

        self._http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        self._client = AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama accepts any string  # noqa: S106
            http_client=self._http_client,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a complete response from Ollama."""
        resolved_model = model or self.default_model
        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content  # type: ignore[union-attr]
            return content or ""
        except httpx.ConnectError as e:
            logger.error("ollama_connection_failed", error=str(e))
            msg = f"Cannot connect to Ollama at {self.base_url}: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            logger.error("ollama_timeout", error=str(e))
            msg = f"Ollama request timed out: {e}"
            raise LLMError(msg) from e
        except Exception as e:
            logger.error("ollama_generate_error", error=str(e), error_type=type(e).__name__)
            msg = f"Ollama generation failed: {e}"
            raise LLMError(msg) from e

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from Ollama."""
        resolved_model = model or self.default_model
        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except httpx.ConnectError as e:
            logger.error("ollama_stream_connection_failed", error=str(e))
            msg = f"Cannot connect to Ollama at {self.base_url}: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            logger.error("ollama_stream_timeout", error=str(e))
            msg = f"Ollama stream timed out: {e}"
            raise LLMError(msg) from e
        except Exception as e:
            logger.error("ollama_stream_error", error=str(e), error_type=type(e).__name__)
            msg = f"Ollama streaming failed: {e}"
            raise LLMError(msg) from e

    async def health_check(self) -> bool:
        """Check if Ollama is reachable by hitting the base URL."""
        try:
            response = await self._http_client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200  # noqa: TRY300
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http_client.aclose()
