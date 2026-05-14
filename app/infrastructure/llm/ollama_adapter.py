"""Ollama LLM adapter using native /api/chat endpoint.

Connects to a self-hosted Ollama instance behind Cloudflare Access.
Authentication is handled by injecting CF service token headers
into every request via httpx.

Uses the native Ollama API (not the OpenAI-compatible endpoint)
to avoid Cloudflare WAF blocks on OpenAI-style request headers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import structlog

from app.core.exceptions import LLMError
from app.domain.interfaces.llm_provider import LLMProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)


class OllamaAdapter(LLMProvider):
    """Concrete LLM provider backed by a self-hosted Ollama instance.

    Uses the native ``/api/chat`` endpoint with Cloudflare Access
    service token headers for authentication.

    Attributes:
        base_url: Ollama API base URL.
        default_model: Default chat model name.
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
            # qwen3 has a "thinking" phase (30–60s) before first token.
            # read timeout must be high enough to cover that delay.
            timeout=httpx.Timeout(300.0, connect=10.0),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        *,
        think: bool = True,
    ) -> str:
        """Generate a complete response from Ollama.

        Uses the native ``/api/chat`` endpoint with ``stream=false``.
        Set ``think=False`` to disable reasoning mode for models like
        qwen3, saving tokens and latency on simple tasks.
        """
        resolved_model = model or self.default_model
        try:
            payload: dict = {
                "model": resolved_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if not think:
                payload["think"] = False

            response = await self._http_client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")

            # Some models emit residual <think>…</think> tags in content
            # even when thinking is disabled.  Strip them out.
            if "</think>" in content:
                content = content.split("</think>", 1)[-1]
            return content.strip()
        except httpx.ConnectError as e:
            logger.error("ollama_connection_failed", error=str(e))
            msg = f"Cannot connect to Ollama at {self.base_url}: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            logger.error("ollama_timeout", error=str(e))
            msg = f"Ollama request timed out: {e}"
            raise LLMError(msg) from e
        except httpx.HTTPStatusError as e:
            logger.error("ollama_http_error", status=e.response.status_code, error=str(e))
            msg = f"Ollama API error ({e.response.status_code}): {e}"
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
        max_tokens: int = 8192,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Stream response tokens from Ollama.

        Uses the native ``/api/chat`` endpoint with ``stream=true``.
        Ollama sends newline-delimited JSON objects. For reasoning models
        (e.g. qwen3), tokens may appear in ``message.thinking`` (internal
        reasoning) or ``message.content`` (visible answer). Both are
        yielded with a ``type`` discriminator.
        """
        resolved_model = model or self.default_model
        try:
            async with self._http_client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": resolved_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        thinking = msg.get("thinking", "")
                        content = msg.get("content", "")
                        if thinking:
                            yield {"type": "thinking", "text": thinking}
                        if content:
                            yield {"type": "content", "text": content}
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as e:
            logger.error("ollama_stream_connection_failed", error=str(e))
            msg = f"Cannot connect to Ollama at {self.base_url}: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            logger.error("ollama_stream_timeout", error=str(e))
            msg = f"Ollama stream timed out: {e}"
            raise LLMError(msg) from e
        except httpx.HTTPStatusError as e:
            logger.error("ollama_stream_http_error", status=e.response.status_code)
            msg = f"Ollama stream error ({e.response.status_code}): {e}"
            raise LLMError(msg) from e
        except LLMError:
            raise
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
