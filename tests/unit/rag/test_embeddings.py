"""Tests for the EmbeddingService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import LLMError
from app.rag.embeddings import EmbeddingService


@pytest.fixture
def service() -> EmbeddingService:
    """Create an EmbeddingService with test configuration."""
    return EmbeddingService(
        base_url="https://test-ollama.example.com",
        model="nomic-embed-text",
        cf_client_id="test-cf-id",
        cf_client_secret="test-cf-secret",
    )


class TestEmbeddingServiceInit:
    """Test suite for service initialization."""

    def test_init_with_cf_headers(self) -> None:
        service = EmbeddingService(
            base_url="https://test.com",
            model="embed-model",
            cf_client_id="id-123",
            cf_client_secret="secret-456",
        )
        headers = service._client.headers
        assert headers.get("CF-Access-Client-Id") == "id-123"
        assert headers.get("CF-Access-Client-Secret") == "secret-456"

    def test_init_without_cf_headers(self) -> None:
        service = EmbeddingService(
            base_url="http://localhost:11434",
            model="embed-model",
        )
        headers = service._client.headers
        assert "CF-Access-Client-Id" not in headers


class TestEmbed:
    """Test suite for embed() method."""

    @pytest.mark.asyncio
    async def test_embed_success(self, service: EmbeddingService) -> None:
        """Successful embedding should return a list of floats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await service.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_embedding_raises(self, service: EmbeddingService) -> None:
        """Empty embedding response should raise LLMError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMError, match="Empty embedding"):
                await service.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self, service: EmbeddingService) -> None:
        """Connection error should raise LLMError."""
        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(LLMError, match="Cannot connect"):
                await service.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_timeout(self, service: EmbeddingService) -> None:
        """Timeout should raise LLMError."""
        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timed out")
            with pytest.raises(LLMError, match="timed out"):
                await service.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_http_error(self, service: EmbeddingService) -> None:
        """HTTP error response should raise LLMError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(LLMError, match="API error"):
                await service.embed("test text")


class TestEmbedBatch:
    """Test suite for embed_batch() method."""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, service: EmbeddingService) -> None:
        """Batch embedding should return list of vectors."""
        with patch.object(service, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.side_effect = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            result = await service.embed_batch(["text1", "text2", "text3"])
            assert len(result) == 3
            assert result[0] == [0.1, 0.2]
            assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, service: EmbeddingService) -> None:
        """Empty batch should return empty list."""
        result = await service.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_generic_error(self, service: EmbeddingService) -> None:
        """Unexpected exceptions should be wrapped in LLMError."""
        with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = RuntimeError("Unexpected failure")
            with pytest.raises(LLMError, match="Embedding generation failed"):
                await service.embed("test text")

    @pytest.mark.asyncio
    async def test_close(self, service: EmbeddingService) -> None:
        """close() should call aclose on the httpx client."""
        with patch.object(service._client, "aclose", new_callable=AsyncMock) as mock_close:
            await service.close()
            mock_close.assert_called_once()
