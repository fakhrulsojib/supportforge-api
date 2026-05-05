"""Domain service for document ingestion pipeline.

Pure business logic — NO framework imports. Orchestrates the full
ingestion workflow: text extraction → chunking → embedding → vector
storage → DB persistence, with proper status tracking and rollback.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from app.core.exceptions import IngestionError
from app.domain.models.document import DocumentChunk
from app.domain.models.enums import DocumentStatus
from app.rag.chunking import RecursiveChunker
from app.workers.text_extractor import TextExtractionError, TextExtractor

if TYPE_CHECKING:
    from app.domain.interfaces.repository import DocumentRepository
    from app.domain.interfaces.vector_store import VectorStore
    from app.domain.models.document import Document
    from app.rag.embeddings import EmbeddingService

logger = structlog.get_logger(__name__)


class IngestionService:
    """Orchestrates the document ingestion pipeline.

    Pipeline steps:
        1. Update status to PROCESSING
        2. Extract text from file content
        3. Chunk text using RecursiveChunker
        4. Generate embeddings via EmbeddingService
        5. Store embeddings in VectorStore (ChromaDB)
        6. Persist chunk records in DocumentRepository
        7. Update status to READY with chunk count

    On failure at any step:
        - Delete any partial chunks from DB
        - Set status to FAILED
        - Raise IngestionError with details

    Attributes:
        _document_repo: Port for document persistence.
        _embedding_service: Service for generating embeddings.
        _vector_store: Port for vector database operations.
        _chunker: Text chunking strategy.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self._document_repo = document_repo
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    async def process_document(
        self,
        document: Document,
        file_content: bytes,
    ) -> None:
        """Process a document through the full ingestion pipeline.

        Args:
            document: Domain document model with metadata.
            file_content: Raw file bytes.

        Raises:
            IngestionError: If any step in the pipeline fails.
        """
        doc_id = document.id
        tenant_id = document.tenant_id

        logger.info(
            "ingestion_started",
            document_id=doc_id,
            filename=document.filename,
            file_type=document.file_type,
            tenant_id=tenant_id,
        )

        # Step 1: Set status to PROCESSING
        await self._document_repo.update_status(
            document_id=doc_id,
            status=DocumentStatus.PROCESSING,
        )

        try:
            # Step 2: Extract text
            text = self._extract_text(file_content, document.file_type)

            # Step 3: Chunk text
            chunks = self._chunker.chunk(
                text,
                metadata={
                    "document_id": doc_id,
                    "filename": document.filename,
                    "file_type": document.file_type,
                },
            )

            if not chunks:
                msg = "Document produced no chunks after text extraction"
                raise IngestionError(msg)

            logger.info(
                "chunking_complete",
                document_id=doc_id,
                chunk_count=len(chunks),
            )

            # Step 4: Generate embeddings
            chunk_texts = [c.content for c in chunks]
            embeddings = await self._embedding_service.embed_batch(chunk_texts)

            logger.info(
                "embeddings_generated",
                document_id=doc_id,
                embedding_count=len(embeddings),
            )

            # Step 5: Generate unique IDs and store in vector DB
            chroma_ids = [f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

            metadatas: list[dict[str, object]] = [
                {
                    "document_id": doc_id,
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "chunk_index": i,
                    "tenant_id": tenant_id,
                }
                for i in range(len(chunks))
            ]

            await self._vector_store.add_documents(
                tenant_id=tenant_id,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=chroma_ids,
            )

            logger.info(
                "vector_store_updated",
                document_id=doc_id,
                stored_count=len(chroma_ids),
            )

            # Step 6: Persist chunk records in DB
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=doc_id,
                    chunk_index=i,
                    content=chunk.content,
                    chroma_id=chroma_ids[i],
                )
                await self._document_repo.create_chunk(db_chunk)

            # Step 7: Update status to READY
            await self._document_repo.update_status(
                document_id=doc_id,
                status=DocumentStatus.READY,
                chunk_count=len(chunks),
            )

            logger.info(
                "ingestion_complete",
                document_id=doc_id,
                chunk_count=len(chunks),
                tenant_id=tenant_id,
            )

        except IngestionError:
            # Already an IngestionError — rollback and re-raise
            await self._rollback(doc_id)
            raise
        except Exception as e:
            # Unexpected error — rollback, wrap, and raise
            await self._rollback(doc_id)
            msg = f"Ingestion failed for document '{doc_id}': {e}"
            raise IngestionError(msg) from e

    def _extract_text(self, content: bytes, file_type: str) -> str:
        """Extract text from file content, wrapping extraction errors.

        Args:
            content: Raw file bytes.
            file_type: File extension.

        Returns:
            Extracted text.

        Raises:
            IngestionError: On extraction failure.
        """
        try:
            return TextExtractor.extract(content, file_type)
        except TextExtractionError as e:
            raise IngestionError(e.message) from e

    async def _rollback(self, document_id: str) -> None:
        """Roll back partial ingestion by cleaning up chunks and setting FAILED.

        Args:
            document_id: The document to roll back.
        """
        logger.warning("ingestion_rollback", document_id=document_id)
        try:
            await self._document_repo.delete_chunks_by_document(document_id)
        except Exception:
            logger.error("rollback_chunk_cleanup_failed", document_id=document_id, exc_info=True)

        try:
            await self._document_repo.update_status(
                document_id=document_id,
                status=DocumentStatus.FAILED,
            )
        except Exception:
            logger.error("rollback_status_update_failed", document_id=document_id, exc_info=True)
