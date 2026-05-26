"""Gemini LLM adapter using Google's OpenAI-compatible endpoint.

Connects to Google's Generative AI API via the ``openai`` Python SDK,
using the OpenAI-compatible base URL.  Each adapter instance is scoped
to a single tenant's API key — there is no shared state between tenants.

Supported models:
    - gemini-2.5-flash (default)
    - gemini-2.5-flash-lite

Thinking (reasoning) is enabled by default — Gemini 2.5 models natively
support it via the ``reasoning_effort`` parameter mapped to their
internal ``thinking_level``.  When not explicitly set, Gemini uses its
default thinking behavior.

Security:
    - API keys are NEVER logged.
    - Each adapter instance is per-request and per-tenant.
    - No credential sharing between tenants.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from openai import AsyncOpenAI

from app.core.exceptions import LLMError
from app.domain.interfaces.llm_provider import LLMProvider, ToolAwareResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# Hardcoded model catalog — Gemini/Gemma models available on the
# Google AI free tier.  Ordered by recommended usage (newest first).
# Gemini 2.0 Flash/Lite omitted — deprecated June 1, 2026.
_GEMINI_CHAT_MODELS: list[dict[str, object]] = [
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "size_gb": 0},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite", "size_gb": 0},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash (Preview)", "size_gb": 0},
    {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite", "size_gb": 0},
    {"id": "gemma-4-31b-it", "name": "Gemma 4 31B IT", "size_gb": 0},
    {"id": "gemma-4-26b-a4b-it", "name": "Gemma 4 26B A4B IT", "size_gb": 0},
]

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiAdapter(LLMProvider):
    """Concrete LLM provider backed by Google Gemini via OpenAI-compat API.

    Uses the ``openai.AsyncOpenAI`` client pointed at Google's
    Generative AI endpoint.  Authentication is via a per-tenant
    API key injected at construction time.

    Attributes:
        default_model: Default chat model name.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.5-flash",
    ) -> None:
        self.default_model = default_model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=_GEMINI_BASE_URL,
        )
        logger.info(
            "gemini_adapter_created",
            default_model=default_model,
            base_url=_GEMINI_BASE_URL,
        )

    @property
    def provider_name(self) -> str:  # noqa: D102
        return "gemini"

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        *,
        think: bool = True,
    ) -> str:
        """Generate a complete response from Gemini.

        Uses the OpenAI-compatible chat completions API.
        Thinking is enabled by default (Gemini's native default level).
        """
        resolved_model = model or self.default_model
        logger.info(
            "gemini_generate_start",
            model=resolved_model,
            message_count=len(messages),
            temperature=temperature,
        )
        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except LLMError:
            raise
        except Exception as e:
            logger.error(
                "gemini_generate_error",
                error=str(e),
                error_type=type(e).__name__,
                model=resolved_model,
            )
            msg = f"Gemini generation failed: {e}"
            raise LLMError(msg) from e

    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
    ) -> ToolAwareResponse:
        """Generate with function calling via Gemini OpenAI-compat API."""
        effective_model = model or self.default_model
        try:
            response = await self._client.chat.completions.create(
                model=effective_model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
            choice = response.choices[0]
            if choice.message.tool_calls:
                parsed_calls = []
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"_raw_invalid_json": tc.function.arguments}
                        
                    parsed_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        )
                    )
                return ToolAwareResponse(
                    tool_calls=parsed_calls,
                    model_used=effective_model,
                )
            return ToolAwareResponse(
                content=choice.message.content or "",
                model_used=effective_model,
            )
        except Exception as exc:
            raise LLMError(f"Gemini tool call failed: {exc}") from exc

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Stream response tokens from Gemini.

        Uses the OpenAI-compatible streaming API.  Yields dicts with
        ``type`` (``"content"``) and ``text`` fields, matching the
        OllamaAdapter stream contract.

        Gemini 2.5 models may also yield thinking content via the
        ``reasoning_content`` field on the delta — those are yielded
        as ``{"type": "thinking", "text": ...}``.
        """
        resolved_model = model or self.default_model
        logger.info(
            "gemini_stream_start",
            model=resolved_model,
            message_count=len(messages),
            temperature=temperature,
        )
        try:
            stream_response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream_response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Thinking/reasoning content (Gemini 2.5 thinking feature)
                thinking = getattr(delta, "reasoning_content", None)
                if thinking:
                    yield {"type": "thinking", "text": thinking}

                # Regular content
                content = delta.content
                if content:
                    yield {"type": "content", "text": content}
        except GeneratorExit:
            # Consumer closed the generator — not an error
            return
        except LLMError:
            raise
        except Exception as e:
            logger.error(
                "gemini_stream_error",
                error=str(e),
                error_type=type(e).__name__,
                model=resolved_model,
            )
            msg = f"Gemini streaming failed: {e}"
            raise LLMError(msg) from e

    async def health_check(self) -> bool:
        """Check if the Gemini API key is valid by listing models."""
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, object]]:
        """Return the hardcoded list of supported Gemini chat models."""
        return list(_GEMINI_CHAT_MODELS)

    async def list_embedding_models(self) -> list[dict[str, object]]:
        """Return the hardcoded list of supported Gemini embedding models."""
        from app.infrastructure.llm.gemini_embedding_adapter import (
            _GEMINI_EMBEDDING_MODELS,
        )
        return list(_GEMINI_EMBEDDING_MODELS)

    async def close(self) -> None:
        """Close the underlying OpenAI async client."""
        logger.info("gemini_adapter_closing")
        await self._client.close()
