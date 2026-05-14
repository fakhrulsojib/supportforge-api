"""Async ingestion worker for background document processing.

Runs as a background task triggered by the upload endpoint. Creates
its own database session (independent from the request session) and
processes the document through the full pipeline. All exceptions are
caught and logged — background tasks must never crash the request handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.core.exceptions import IngestionError
from app.domain.services.ingestion_service import IngestionService
from app.infrastructure.database.connection import AsyncSessionLocal
from app.infrastructure.database.repositories.document_repo import SQLDocumentRepository

if TYPE_CHECKING:
    from app.domain.interfaces.llm_provider import LLMProvider
    from app.domain.interfaces.vector_store import VectorStore
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


async def run_ingestion_task(
    document_id: str,
    file_content: bytes,
    tenant_id: str,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    llm_provider: LLMProvider | None = None,
) -> None:
    """Background task to process a document through the ingestion pipeline.

    Creates its own database session so it is fully independent from the
    HTTP request lifecycle.  The upload endpoint commits the document
    record before scheduling this task, so the document is guaranteed
    to be visible in the new session.

    This function is designed to run as a background task (e.g., via
    ``BackgroundTasks`` or ``asyncio.create_task``). It never raises
    exceptions — all errors are logged and the document status is
    updated to FAILED by the IngestionService.

    Args:
        document_id: ID of the document to process.
        file_content: Raw file bytes.
        tenant_id: Tenant ID for isolation verification.
        embedding_service: Service for generating embeddings.
        vector_store: Vector database for storing embeddings.
        llm_provider: Optional LLM provider for chunk contextualisation.
    """
    logger.info(
        "ingestion_task_started",
        document_id=document_id,
        tenant_id=tenant_id,
    )

    try:
        async with AsyncSessionLocal() as session:
            document_repo = SQLDocumentRepository(session)

            # Fetch the document from the repository
            document = await document_repo.get_by_id(document_id)

            if document is None:
                logger.error(
                    "ingestion_task_document_not_found",
                    document_id=document_id,
                )
                return

            # Security check: verify tenant isolation
            if document.tenant_id != tenant_id:
                logger.error(
                    "ingestion_task_tenant_mismatch",
                    document_id=document_id,
                    expected_tenant=tenant_id,
                    actual_tenant=document.tenant_id,
                )
                return

            # Create the ingestion service and process
            service = IngestionService(
                document_repo=document_repo,
                embedding_service=embedding_service,
                vector_store=vector_store,
                llm_provider=llm_provider,
            )

            await service.process_document(
                document=document,
                file_content=file_content,
            )

            # Commit all changes (status updates, chunks, etc.)
            await session.commit()

            logger.info(
                "ingestion_task_completed",
                document_id=document_id,
                tenant_id=tenant_id,
            )

    except IngestionError as e:
        # IngestionService already handled rollback and status update
        logger.error(
            "ingestion_task_failed",
            document_id=document_id,
            error=e.message,
        )

    except Exception:
        # Catch-all for unexpected errors — background tasks must not crash
        logger.error(
            "ingestion_task_unexpected_error",
            document_id=document_id,
            exc_info=True,
        )
