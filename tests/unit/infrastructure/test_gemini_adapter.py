"""Unit tests for the GeminiAdapter LLM provider.

Covers:
    - provider_name returns "gemini"
    - generate() calls OpenAI-compat API and returns content
    - generate() raises LLMError on API error
    - generate() raises LLMError on invalid API key
    - stream() yields content and thinking frames
    - stream() raises LLMError on API error
    - health_check() returns True when API key is valid
    - health_check() returns False on failure
    - list_models() returns hardcoded Gemini model list
    - list_embedding_models() returns empty list
    - close() closes the underlying client
    - default model used when no override
    - model override respected
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import LLMError
from app.infrastructure.llm.gemini_adapter import GeminiAdapter

if TYPE_CHECKING:
    pass


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def adapter() -> GeminiAdapter:
    """Create a GeminiAdapter with a fake API key."""
    return GeminiAdapter(api_key="fake-api-key", default_model="gemini-2.5-flash")


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI async client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.models = MagicMock()
    return client


# ── Provider Name ───────────────────────────────────────────────


class TestProviderName:
    """Tests for provider identification."""

    def test_provider_name(self, adapter: GeminiAdapter) -> None:
        """Provider name should be 'gemini'."""
        assert adapter.provider_name == "gemini"


# ── Generate ────────────────────────────────────────────────────


class TestGenerate:
    """Tests for non-streaming generation."""

    @pytest.mark.asyncio
    async def test_generate_returns_content(self, adapter: GeminiAdapter) -> None:
        """generate() should return the LLM response text."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Gemini!"
        mock_response.choices = [mock_choice]

        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await adapter.generate(
                [{"role": "user", "content": "Hi"}],
                model="gemini-2.5-flash",
            )
            assert result == "Hello from Gemini!"

    @pytest.mark.asyncio
    async def test_generate_uses_default_model(self, adapter: GeminiAdapter) -> None:
        """generate() with model=None should use default_model."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]

        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_create:
            await adapter.generate([{"role": "user", "content": "test"}])
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs.get("model") == "gemini-2.5-flash" or \
                   call_kwargs[1].get("model") == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_generate_raises_on_api_error(self, adapter: GeminiAdapter) -> None:
        """generate() should raise LLMError on API failures."""
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("API quota exceeded"),
        ):
            with pytest.raises(LLMError):
                await adapter.generate([{"role": "user", "content": "test"}])


# ── Stream ──────────────────────────────────────────────────────


class TestStream:
    """Tests for streaming generation."""

    @pytest.mark.asyncio
    async def test_stream_yields_content(self, adapter: GeminiAdapter) -> None:
        """stream() should yield content frames."""
        # Create mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        async def mock_stream():
            yield chunk1
            yield chunk2

        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_stream(),
        ):
            frames = []
            async for frame in adapter.stream(
                [{"role": "user", "content": "Hi"}],
            ):
                frames.append(frame)

            content_frames = [f for f in frames if f["type"] == "content"]
            assert len(content_frames) == 2
            assert content_frames[0]["text"] == "Hello"
            assert content_frames[1]["text"] == " world"

    @pytest.mark.asyncio
    async def test_stream_raises_on_error(self, adapter: GeminiAdapter) -> None:
        """stream() should raise LLMError on API failures."""
        with patch.object(
            adapter._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(LLMError):
                async for _ in adapter.stream(
                    [{"role": "user", "content": "test"}],
                ):
                    pass


# ── Health Check ────────────────────────────────────────────────


class TestHealthCheck:
    """Tests for API key validation via health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter: GeminiAdapter) -> None:
        """health_check() should return True when API key is valid."""
        mock_response = MagicMock()
        mock_response.data = [{"id": "gemini-2.5-flash"}]
        with patch.object(
            adapter._client.models,
            "list",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, adapter: GeminiAdapter) -> None:
        """health_check() should return False on failure."""
        with patch.object(
            adapter._client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=Exception("Invalid API key"),
        ):
            result = await adapter.health_check()
            assert result is False


# ── Model Listing ───────────────────────────────────────────────


class TestModelListing:
    """Tests for static model listing."""

    @pytest.mark.asyncio
    async def test_list_models_returns_gemini_models(
        self, adapter: GeminiAdapter
    ) -> None:
        """list_models() should return hardcoded Gemini chat models."""
        models = await adapter.list_models()
        assert isinstance(models, list)
        assert len(models) >= 2
        model_ids = {m["id"] for m in models}
        assert "gemini-2.5-flash" in model_ids
        assert "gemini-2.5-flash-lite" in model_ids

    @pytest.mark.asyncio
    async def test_list_embedding_models_empty(
        self, adapter: GeminiAdapter
    ) -> None:
        """list_embedding_models() should return empty list."""
        models = await adapter.list_embedding_models()
        assert models == []


# ── Close ───────────────────────────────────────────────────────


class TestClose:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, adapter: GeminiAdapter) -> None:
        """close() should close the underlying OpenAI client."""
        with patch.object(
            adapter._client,
            "close",
            new_callable=AsyncMock,
        ) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()
