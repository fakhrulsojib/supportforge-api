"""Async ingestion worker for background document processing.

Runs as a background task triggered by the upload endpoint. Creates
its own database session (independent from the request session) and
processes the document through the full pipeline. All exceptions are
caught and logged — background tasks must never crash the request handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from app.core.exceptions import IngestionError
from app.domain.models.enums import DocumentStatus
from app.domain.services.ingestion_service import IngestionService
from app.infrastructure.database.connection import AsyncSessionLocal
from app.infrastructure.database.repositories.document_repo import SQLDocumentRepository
from app.infrastructure.database.repositories.tenant_repo import SQLTenantRepository

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

            # Set status to PROCESSING and commit immediately so the
            # frontend sees this status while the pipeline runs.
            await document_repo.update_status(
                document_id=document_id,
                status=DocumentStatus.PROCESSING,
            )
            await session.commit()

            # Read per-tenant model overrides from config_json
            from app.config import get_settings
            from app.core.tenant_config import resolve_tenant_models
            settings = get_settings()
            tenant_repo = SQLTenantRepository(session)
            tenant = await tenant_repo.get_by_id(tenant_id)

            # Load tenant secrets for API key resolution (secrets > config_json)
            from app.infrastructure.database.repositories.tenant_secret_repo import (
                SQLTenantSecretRepository,
            )
            sec_repo = SQLTenantSecretRepository(session, encryption_key=settings.secret_key)
            try:
                tenant_secrets = await sec_repo.get_all_decrypted(tenant_id)
            except Exception:
                tenant_secrets = {}

            tenant_models = resolve_tenant_models(
                tenant.config_json if tenant else None,
                encryption_key=settings.secret_key,
                secrets=tenant_secrets,
            )

            # Resolve effective embedding service for this tenant
            effective_embed = embedding_service
            embed_disposable = False
            if (
                tenant_models.embedding_provider == "gemini"
                and tenant_models.gemini_embedding_api_key
            ):
                from app.infrastructure.llm.factory import get_gemini_embedding_provider
                resolved_model = tenant_models.embedding_model or "gemini-embedding-2"
                effective_embed = get_gemini_embedding_provider(
                    api_key=tenant_models.gemini_embedding_api_key,
                    model=resolved_model,
                )
                embed_disposable = True
                logger.info(
                    "ingestion_using_gemini_embeddings",
                    document_id=document_id,
                    model=resolved_model,
                )
            elif tenant_models.embedding_provider == "gemini":
                # Provider is gemini but key is missing/corrupted
                logger.warning(
                    "ingestion_gemini_embedding_key_missing",
                    document_id=document_id,
                    hint="Gemini embeddings configured but API key missing — falling back to Ollama",
                )

            # Resolve effective chat LLM for contextualisation
            effective_llm = llm_provider
            llm_disposable = False
            if (
                tenant_models.chat_provider == "gemini"
                and tenant_models.gemini_api_key
            ):
                from app.infrastructure.llm.factory import get_gemini_provider
                resolved_chat_model = tenant_models.chat_model or "gemini-2.5-flash"
                effective_llm = get_gemini_provider(
                    api_key=tenant_models.gemini_api_key,
                    model=resolved_chat_model,
                )
                llm_disposable = True
                logger.info(
                    "ingestion_using_gemini_chat",
                    document_id=document_id,
                    model=resolved_chat_model,
                )
            elif tenant_models.chat_provider == "gemini":
                logger.warning(
                    "ingestion_gemini_chat_key_missing",
                    document_id=document_id,
                    hint="Gemini chat configured but API key missing — falling back to Ollama for contextualisation",
                )

            # Create the ingestion service and process
            try:
                service = IngestionService(
                    document_repo=document_repo,
                    embedding_service=effective_embed,
                    vector_store=vector_store,
                    llm_provider=effective_llm,
                )

                await service.process_document(
                    document=document,
                    file_content=file_content,
                    tenant_chat_model=tenant_models.chat_model,
                    tenant_embedding_model=tenant_models.embedding_model,
                )
            finally:
                if embed_disposable and hasattr(effective_embed, 'close'):
                    try:
                        await effective_embed.close()
                    except Exception:  # noqa: S110
                        logger.debug("embed_adapter_close_failed", exc_info=True)
                if llm_disposable and hasattr(effective_llm, 'close'):
                    try:
                        await effective_llm.close()
                    except Exception:  # noqa: S110
                        logger.debug("llm_adapter_close_failed", exc_info=True)

            # Commit all changes (chunks, READY status, etc.)
            await session.commit()

            # Clean up the cached file on success
            cache_path = Path(f"/tmp/supportforge_docs/{document_id}.bin")
            cache_path.unlink(missing_ok=True)

            logger.info(
                "ingestion_task_completed",
                document_id=document_id,
                tenant_id=tenant_id,
            )

    except IngestionError as e:
        # IngestionService set FAILED via rollback but the session's
        # async-with block has exited — open a fresh session to commit.
        try:
            async with AsyncSessionLocal() as err_session:
                err_repo = SQLDocumentRepository(err_session)
                await err_repo.update_status(
                    document_id=document_id,
                    status=DocumentStatus.FAILED,
                )
                await err_session.commit()
        except Exception:
            logger.error(
                "ingestion_failed_status_commit_error",
                document_id=document_id,
                exc_info=True,
            )
        logger.error(
            "ingestion_task_failed",
            document_id=document_id,
            error=e.message,
        )

    except Exception:
        # Catch-all for unexpected errors — try to mark as FAILED
        try:
            async with AsyncSessionLocal() as err_session:
                err_repo = SQLDocumentRepository(err_session)
                await err_repo.update_status(
                    document_id=document_id,
                    status=DocumentStatus.FAILED,
                )
                await err_session.commit()
        except Exception:
            logger.error(
                "ingestion_failed_status_commit_error",
                document_id=document_id,
                exc_info=True,
            )
        logger.error(
            "ingestion_task_unexpected_error",
            document_id=document_id,
            exc_info=True,
        )
