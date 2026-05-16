"""Gemini embedding adapter using Google's REST embedContent API.

Generates text embeddings via the ``embedContent`` endpoint of the
Google Generative AI API.  Each adapter instance is scoped to a single
tenant's API key — there is no shared state between tenants.

Supported models:
    - gemini-embedding-2 (default, recommended)
    - gemini-embedding-001
    - gemini-embedding-2-preview

Security:
    - API keys are NEVER logged.
    - Each adapter instance is per-request and per-tenant.
    - No credential sharing between tenants.
"""

from __future__ import annotations

import httpx
import structlog

from app.core.exceptions import LLMError

logger = structlog.get_logger(__name__)

_GEMINI_EMBED_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Hardcoded embedding model catalog — Gemini embedding models available
# on the Google AI free tier.
_GEMINI_EMBEDDING_MODELS: list[dict[str, object]] = [
    {"id": "gemini-embedding-2", "name": "Gemini Embedding 2", "size_gb": 0},
    {"id": "gemini-embedding-001", "name": "Gemini Embedding 001", "size_gb": 0},
    {"id": "gemini-embedding-2-preview", "name": "Gemini Embedding 2 (Preview)", "size_gb": 0},
]


class GeminiEmbeddingAdapter:
    """Generates embeddings via Google's REST embedContent API.

    Uses ``httpx.AsyncClient`` for HTTP calls.  Authentication is via
    a per-tenant API key injected at construction time.

    This adapter uses the native 3072-dimensional output by default.

    Attributes:
        model: Default embedding model name.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info(
            "gemini_embedding_adapter_created",
            default_model=model,
        )

    async def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: Input text to embed.
            model: Optional model override. Falls back to ``self.model``.
            task_type: Task type hint for the model.
                Use ``RETRIEVAL_QUERY`` for search queries and
                ``RETRIEVAL_DOCUMENT`` for document ingestion.

        Returns:
            Embedding vector as a list of floats (3072 dimensions).

        Raises:
            LLMError: If embedding generation fails.
        """
        resolved_model = model or self.model
        url = f"{_GEMINI_EMBED_BASE}/{resolved_model}:embedContent"

        logger.debug(
            "gemini_embedding_start",
            model=resolved_model,
            task_type=task_type,
            text_length=len(text),
        )

        try:
            response = await self._client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                json={
                    "model": f"models/{resolved_model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                },
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", {}).get("values", [])
            if not embedding:
                msg = f"Empty embedding returned for text: {text[:50]}..."
                raise LLMError(msg)

            logger.debug(
                "gemini_embedding_complete",
                model=resolved_model,
                dimensions=len(embedding),
            )
            return embedding  # type: ignore[no-any-return]

        except httpx.ConnectError as e:
            msg = f"Cannot connect to Gemini embedding API: {e}"
            raise LLMError(msg) from e
        except httpx.TimeoutException as e:
            msg = f"Gemini embedding request timed out: {e}"
            raise LLMError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = f"Gemini embedding API error: {e.response.status_code}"
            logger.error(
                "gemini_embedding_api_error",
                status_code=e.response.status_code,
                response_text=e.response.text[:200],
                model=resolved_model,
            )
            raise LLMError(msg) from e
        except LLMError:
            raise
        except Exception as e:
            msg = f"Gemini embedding generation failed: {e}"
            logger.error(
                "gemini_embedding_error",
                error=str(e),
                error_type=type(e).__name__,
                model=resolved_model,
            )
            raise LLMError(msg) from e

    async def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Processes sequentially.  The default ``task_type`` is
        ``RETRIEVAL_DOCUMENT`` since batch embedding is typically
        used during document ingestion.

        Args:
            texts: List of texts to embed.
            model: Optional model override.
            task_type: Task type hint (default: RETRIEVAL_DOCUMENT).

        Returns:
            List of embedding vectors.
        """
        logger.info(
            "gemini_embedding_batch_start",
            model=model or self.model,
            count=len(texts),
            task_type=task_type,
        )
        embeddings: list[list[float]] = []
        for text in texts:
            embedding = await self.embed(text, model=model, task_type=task_type)
            embeddings.append(embedding)
        logger.info(
            "gemini_embedding_batch_complete",
            model=model or self.model,
            count=len(embeddings),
            dimensions=len(embeddings[0]) if embeddings else 0,
        )
        return embeddings

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        logger.info("gemini_embedding_adapter_closing")
        await self._client.aclose()
