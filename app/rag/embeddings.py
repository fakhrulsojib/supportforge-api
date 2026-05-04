"""Embedding generation wrapper for Ollama.

Uses Ollama's native ``/api/embeddings`` endpoint (not OpenAI-compatible)
to generate embedding vectors for text.
"""

from __future__ import annotations

import httpx
import structlog

from app.core.exceptions import LLMError

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """Generates embeddings via Ollama's /api/embeddings endpoint.

    Attributes:
        base_url: Ollama API base URL.
        model: Embedding model name.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        cf_client_id: str = "",
        cf_client_secret: str = "",
    ) -> None:
        self.base_url = base_url
        self.model = model

        headers: dict[str, str] = {}
        if cf_client_id and cf_client_secret:
            headers["CF-Access-Client-Id"] = cf_client_id
            headers["CF-Access-Client-Secret"] = cf_client_secret

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            LLMError: If embedding generation fails.
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding", [])
            if not embedding:
                msg = f"Empty embedding returned for text: {text[:50]}..."
                raise LLMError(msg)
            return embedding  # type: ignore[no-any-return]
        except httpx.ConnectError as e:
            msg = f"Cannot connect to Ollama for embeddings: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Embedding request timed out: {e}"
            raise LLMError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Ollama embedding API error: {e.response.status_code}"
            raise LLMError(msg) from e
        except LLMError:
            raise
        except Exception as e:
            msg = f"Embedding generation failed: {e}"
            raise LLMError(msg) from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes sequentially since Ollama doesn't support batch embeddings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings: list[list[float]] = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
