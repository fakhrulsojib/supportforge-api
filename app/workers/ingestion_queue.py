"""Ingestion queue — semaphore-bounded concurrency for document processing.

Limits the number of documents being processed simultaneously to avoid
overwhelming the LLM/embedding server.  The concurrency limit is
configurable via ``INGESTION_MAX_CONCURRENT`` (default: 2).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.domain.interfaces.llm_provider import LLMProvider
    from app.domain.interfaces.vector_store import VectorStore
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class IngestionQueue:
    """Bounded-concurrency queue for ingestion tasks.

    Uses an ``asyncio.Semaphore`` so that at most ``max_concurrent``
    documents are processed at the same time.  Tasks that arrive while
    the semaphore is full wait until a slot becomes available.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        logger.info(
            "ingestion_queue_initialized",
            max_concurrent=max_concurrent,
        )

    @property
    def max_concurrent(self) -> int:
        """Return the configured concurrency limit."""
        return self._max

    async def submit(
        self,
        *,
        document_id: str,
        file_content: bytes,
        tenant_id: str,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Enqueue a document for ingestion.

        Fires an ``asyncio.create_task`` immediately, but the actual
        work blocks on the semaphore so only ``max_concurrent`` tasks
        run at a time.
        """
        logger.info(
            "ingestion_queued",
            document_id=document_id,
            tenant_id=tenant_id,
        )
        asyncio.create_task(
            self._run(
                document_id=document_id,
                file_content=file_content,
                tenant_id=tenant_id,
                embedding_service=embedding_service,
                vector_store=vector_store,
                llm_provider=llm_provider,
            ),
        )

    async def _run(
        self,
        *,
        document_id: str,
        file_content: bytes,
        tenant_id: str,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Acquire the semaphore, run ingestion, then release."""
        from app.workers.ingestion_worker import run_ingestion_task

        async with self._semaphore:
            logger.info(
                "ingestion_slot_acquired",
                document_id=document_id,
            )
            try:
                await run_ingestion_task(
                    document_id=document_id,
                    file_content=file_content,
                    tenant_id=tenant_id,
                    embedding_service=embedding_service,
                    vector_store=vector_store,
                    llm_provider=llm_provider,
                )
            finally:
                logger.info(
                    "ingestion_slot_released",
                    document_id=document_id,
                )

