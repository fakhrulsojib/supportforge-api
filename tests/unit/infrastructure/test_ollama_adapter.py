"""Tests for Ollama LLM adapter.

Uses mocked httpx responses to test all code paths without
requiring a live Ollama instance.
"""

from __future__ import annotations

import json
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
    """Test suite for generate() method (uses streaming internally)."""

    @staticmethod
    def _mock_stream_response(lines: list[str]):
        """Build a mock streaming response context manager."""
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def _aiter():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter

        ctx_manager = AsyncMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=mock_response)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)
        return ctx_manager

    @pytest.mark.asyncio
    async def test_generate_success(self, adapter: OllamaAdapter) -> None:
        """Successful generation should return collected content string."""
        lines = [
            json.dumps({"message": {"content": "Hello, "}}),
            json.dumps({"message": {"content": "world!"}}),
        ]
        ctx = self._mock_stream_response(lines)
        with patch.object(adapter._http_client, "stream", return_value=ctx):
            result = await adapter.generate([{"role": "user", "content": "Hi"}])
            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_generate_uses_default_model(self, adapter: OllamaAdapter) -> None:
        """Generate should use default_model when no model override is given."""
        lines = [json.dumps({"message": {"content": "ok"}})]
        ctx = self._mock_stream_response(lines)
        with patch.object(adapter._http_client, "stream", return_value=ctx) as mock_stream:
            await adapter.generate([{"role": "user", "content": "test"}])
            call_kwargs = mock_stream.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_generate_uses_override_model(self, adapter: OllamaAdapter) -> None:
        """Generate should use override model when provided."""
        lines = [json.dumps({"message": {"content": "ok"}})]
        ctx = self._mock_stream_response(lines)
        with patch.object(adapter._http_client, "stream", return_value=ctx) as mock_stream:
            await adapter.generate(
                [{"role": "user", "content": "test"}], model="custom-model"
            )
            call_kwargs = mock_stream.call_args
            body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert body["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_generate_empty_content_returns_empty(self, adapter: OllamaAdapter) -> None:
        """No content tokens should return empty string."""
        lines = [json.dumps({"message": {}})]
        ctx = self._mock_stream_response(lines)
        with patch.object(adapter._http_client, "stream", return_value=ctx):
            result = await adapter.generate([{"role": "user", "content": "test"}])
            assert result == ""

    @pytest.mark.asyncio
    async def test_generate_strips_think_tags(self, adapter: OllamaAdapter) -> None:
        """Residual </think> tags in content should be stripped."""
        lines = [
            json.dumps({"message": {"content": "<think>reasoning</think>\nThe answer."}}),
        ]
        ctx = self._mock_stream_response(lines)
        with patch.object(adapter._http_client, "stream", return_value=ctx):
            result = await adapter.generate([{"role": "user", "content": "test"}])
            assert result == "The answer."

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, adapter: OllamaAdapter) -> None:
        """Connection errors should be mapped to LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMError, match="Cannot connect"):
                await adapter.generate([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generate_timeout_error(self, adapter: OllamaAdapter) -> None:
        """Timeout errors should be mapped to LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.TimeoutException("Request timed out")
            with pytest.raises(LLMError, match="timed out"):
                await adapter.generate([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_generate_generic_error(self, adapter: OllamaAdapter) -> None:
        """Other exceptions should be mapped to LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = ValueError("Model not found")
            with pytest.raises(LLMError, match="generation failed"):
                await adapter.generate([{"role": "user", "content": "test"}])


class TestStream:
    """Test suite for stream() method (native /api/chat with stream=true)."""

    @pytest.mark.asyncio
    async def test_stream_content_only(self, adapter: OllamaAdapter) -> None:
        """Streaming should yield content frames for standard models."""
        lines = [
            json.dumps({"message": {"content": "Hello "}}),
            json.dumps({"message": {"content": "world!"}}),
            json.dumps({"message": {"content": ""}}),  # empty — should be skipped
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = self._aiter_lines(lines)

        ctx_manager = AsyncMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=mock_response)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter._http_client, "stream", return_value=ctx_manager):
            tokens = []
            async for frame in adapter.stream([{"role": "user", "content": "test"}]):
                tokens.append(frame)

        assert len(tokens) == 2
        assert tokens[0] == {"type": "content", "text": "Hello "}
        assert tokens[1] == {"type": "content", "text": "world!"}

    @pytest.mark.asyncio
    async def test_stream_thinking_and_content(self, adapter: OllamaAdapter) -> None:
        """Reasoning models should yield both thinking and content frames."""
        lines = [
            json.dumps({"message": {"thinking": "Let me think...", "content": ""}}),
            json.dumps({"message": {"thinking": "", "content": "The answer."}}),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = self._aiter_lines(lines)

        ctx_manager = AsyncMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=mock_response)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter._http_client, "stream", return_value=ctx_manager):
            tokens = []
            async for frame in adapter.stream([{"role": "user", "content": "test"}]):
                tokens.append(frame)

        assert len(tokens) == 2
        assert tokens[0] == {"type": "thinking", "text": "Let me think..."}
        assert tokens[1] == {"type": "content", "text": "The answer."}

    @pytest.mark.asyncio
    async def test_stream_connection_error(self, adapter: OllamaAdapter) -> None:
        """Stream connection errors should raise LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMError, match="Cannot connect"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass

    @pytest.mark.asyncio
    async def test_stream_timeout_error(self, adapter: OllamaAdapter) -> None:
        """Stream timeout should raise LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = httpx.TimeoutException("Timed out")
            with pytest.raises(LLMError, match="timed out"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, adapter: OllamaAdapter) -> None:
        """Generic exceptions during streaming should raise LLMError."""
        with patch.object(adapter._http_client, "stream") as mock_stream:
            mock_stream.side_effect = RuntimeError("Unexpected error")
            with pytest.raises(LLMError, match="streaming failed"):
                async for _ in adapter.stream([{"role": "user", "content": "test"}]):
                    pass

    @pytest.mark.asyncio
    async def test_stream_malformed_json_skipped(self, adapter: OllamaAdapter) -> None:
        """Malformed JSON lines should be skipped without raising."""
        lines = [
            "not-valid-json",
            json.dumps({"message": {"content": "valid"}}),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = self._aiter_lines(lines)

        ctx_manager = AsyncMock()
        ctx_manager.__aenter__ = AsyncMock(return_value=mock_response)
        ctx_manager.__aexit__ = AsyncMock(return_value=False)

        with patch.object(adapter._http_client, "stream", return_value=ctx_manager):
            tokens = []
            async for frame in adapter.stream([{"role": "user", "content": "test"}]):
                tokens.append(frame)

        assert len(tokens) == 1
        assert tokens[0] == {"type": "content", "text": "valid"}

    @staticmethod
    def _aiter_lines(lines: list[str]):
        """Create a callable that returns an async line iterator."""
        async def _iter():
            for line in lines:
                yield line
        return _iter


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

    @pytest.mark.asyncio
    async def test_health_check_generic_exception(self, adapter: OllamaAdapter) -> None:
        """Generic exceptions should return False (not raise)."""
        with patch.object(adapter._http_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected")
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


class TestClose:
    """Test suite for close() method."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, adapter: OllamaAdapter) -> None:
        """close() should call aclose on the underlying httpx client."""
        with patch.object(adapter._http_client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()
