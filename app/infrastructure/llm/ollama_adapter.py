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
            # Generous read timeout to cover slow models and long generations.
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

        Uses streaming internally to avoid Cloudflare 524 timeouts on
        long-running generations (e.g. reasoning models).  Collects all
        content tokens and returns the full text as a single string.
        """
        resolved_model = model or self.default_model
        try:
            payload: dict = {
                "model": resolved_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            # NOTE: We intentionally do NOT send the top-level "think"
            # parameter.  Thinking models (e.g. qwen3) default to
            # thinking-on, which is fine.  Non-thinking models (e.g.
            # phi4-mini) will return a 400 error if they receive it.
            # Omitting it keeps the adapter model-agnostic.

            content_parts: list[str] = []

            async with self._http_client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        # Collect content tokens only; thinking tokens
                        # (if the model produces them) are skipped.
                        content = msg.get("content", "")
                        if content:
                            content_parts.append(content)
                    except json.JSONDecodeError:
                        continue

            result = "".join(content_parts)

            # Safety net: strip residual <think>…</think> tags that some
            # reasoning models may leak into the content field.
            if "</think>" in result:
                result = result.split("</think>", 1)[-1]
            return result.strip()
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
        Ollama sends newline-delimited JSON objects. For reasoning models,
        tokens may appear in ``message.thinking`` (internal reasoning)
        or ``message.content`` (visible answer). Both are yielded with
        a ``type`` discriminator. Non-thinking models only produce
        ``content`` frames.
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
