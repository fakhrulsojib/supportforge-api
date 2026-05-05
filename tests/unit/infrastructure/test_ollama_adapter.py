"""Tests for Ollama LLM adapter.

Uses mocked httpx responses to test all code paths without
requiring a live Ollama instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import LLMError
from app.infrastructure.llm.factory import get_llm_provider
from app.infrastructure.llm.ollama_adapter import OllamaAdapter


@pytest.fixture
def adapter() -> OllamaAdapter:
    """Create an OllamaAdapter with test configuration."""
    return OllamaAdapter(
        base_url="https://test-ollama.example.com",
        default_model="test-model",
        cf_client_id="test-cf-id",
        cf_client_secret="test-cf-secret",
    )


class TestOllamaAdapterInit:
    """Test suite for adapter initialization."""

    def test_init_with_cf_headers(self) -> None:
        """Adapter should inject CF Access headers into httpx client."""
        adapter = OllamaAdapter(
            base_url="https://ollama.test.com",
            default_model="llama3",
            cf_client_id="cf-id-123",
            cf_client_secret="cf-secret-456",
        )
        headers = adapter._http_client.headers
        assert headers.get("CF-Access-Client-Id") == "cf-id-123"
        assert headers.get("CF-Access-Client-Secret") == "cf-secret-456"

    def test_init_without_cf_headers(self) -> None:
        """Adapter should work without CF headers (local Ollama)."""
        adapter = OllamaAdapter(
            base_url="http://localhost:11434",
            default_model="llama3",
        )
        headers = adapter._http_client.headers
        assert "CF-Access-Client-Id" not in headers

    def test_default_model_stored(self) -> None:
        adapter = OllamaAdapter(
            base_url="http://localhost:11434",
            default_model="qwen3:4b",
        )
        assert adapter.default_model == "qwen3:4b"


class TestGenerate:
    """Test suite for generate() method."""

    @pytest.mark.asyncio
    async def test_generate_success(self, adapter: OllamaAdapter) -> None:
        """Successful generation should return content string."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello, world!"))]

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            result = await adapter.generate([{"role": "user", "content": "Hi"}])
            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_uses_default_model(self, adapter: OllamaAdapter) -> None:
        """Generate should use default_model when no model override is given."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            await adapter.generate([{"role": "user", "content": "test"}])
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_generate_uses_override_model(self, adapter: OllamaAdapter) -> None:
        """Generate should use override model when provided."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            await adapter.generate([{"role": "user", "content": "test"}], model="custom-model")
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_generate_empty_content_returns_empty(self, adapter: OllamaAdapter) -> None:
        """None content should be converted to empty string."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            result = await adapter.generate([{"role": "user", "content": "test"}])
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, adapter: OllamaAdapter) -> None:
        """Connection errors should be mapped to LLMError."""
        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMError, match="Cannot connect"):
                await adapter.generate([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, adapter: OllamaAdapter) -> None:
        """Timeout errors should be mapped to LLMError."""
        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.TimeoutException("Request timed out")
            with pytest.raises(LLMError, match="timed out"):
                await adapter.generate([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generate_generic_error(self, adapter: OllamaAdapter) -> None:
        """Other exceptions should be mapped to LLMError."""
        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = ValueError("Model not found")
            with pytest.raises(LLMError, match="generation failed"):
                await adapter.generate([{"role": "user", "content": "test"}])


class TestStream:
    """Test suite for stream() method."""

    @pytest.mark.asyncio
    async def test_stream_success(self, adapter: OllamaAdapter) -> None:
        """Streaming should yield token chunks."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]

        async def mock_stream():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream()
            tokens = []
            async for token in adapter.stream([{"role": "user", "content": "test"}]):
                tokens.append(token)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, adapter: OllamaAdapter) -> None:
        """Stream connection errors should raise LLMError."""
        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMError, match="Cannot connect"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, adapter: OllamaAdapter) -> None:
        """Stream timeout should raise LLMError."""
        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = httpx.TimeoutException("Timed out")
            with pytest.raises(LLMError, match="timed out"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass


class TestHealthCheck:
    """Test suite for health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, adapter: OllamaAdapter) -> None:
        """Healthy Ollama should return True."""
        mock_response = MagicMock(status_code=200)
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await adapter.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, adapter: OllamaAdapter) -> None:
        """Non-200 response should return False."""
        mock_response = MagicMock(status_code=503)
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, adapter: OllamaAdapter) -> None:
        """Connection error should return False (not raise)."""
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = await adapter.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, adapter: OllamaAdapter) -> None:
        """Timeout should return False (not raise)."""
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Timed out")
            result = await adapter.health_check()
            assert result is False


class TestFactory:
    """Test suite for LLM provider factory."""

    def test_factory_returns_ollama_adapter(self) -> None:
        """Factory should return an OllamaAdapter."""
        from app.config import Settings

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        provider = get_llm_provider(settings)
        assert isinstance(provider, OllamaAdapter)

    def test_factory_configures_base_url(self) -> None:
        """Factory should pass settings to the adapter."""
        from app.config import Settings

        settings = Settings(
            ollama_base_url="https://custom.ollama.com",
            ollama_chat_model="custom-model",
            _env_file=None,  # type: ignore[call-arg]
        )
        provider = get_llm_provider(settings)
        assert isinstance(provider, OllamaAdapter)
        assert provider.base_url == "https://custom.ollama.com"
        assert provider.default_model == "custom-model"


class TestHealthCheckGeneric:
    """Additional health check edge cases."""

    @pytest.mark.asyncio
    async def test_health_check_generic_exception(self, adapter: OllamaAdapter) -> None:
        """Generic exceptions should return False (not raise)."""
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected")
            result = await adapter.health_check()
            assert result is False


class TestClose:
    """Test suite for close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, adapter: OllamaAdapter) -> None:
        """close() should call aclose on the underlying httpx client."""
        with patch.object(adapter._http_client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()


class TestStreamGeneric:
    """Additional stream edge cases."""

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, adapter: OllamaAdapter) -> None:
        """Generic exceptions during streaming should raise LLMError."""

        async def mock_stream():
            raise RuntimeError("Unexpected error")
            yield  # type: ignore[misc]  # Make this a generator  # noqa: E501

        with patch.object(adapter._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream()
            with pytest.raises(LLMError, match="streaming failed"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass

