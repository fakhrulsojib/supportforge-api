"""Domain service for document ingestion pipeline.

Pure business logic — NO framework imports. Orchestrates the full
ingestion workflow: text extraction → chunking → contextualisation →
embedding → vector storage → DB persistence, with proper status
tracking and rollback.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog

from app.core.exceptions import IngestionError
from app.domain.models.document import DocumentChunk
from app.domain.models.enums import DocumentStatus
from app.rag.chunking import Chunk, RecursiveChunker
from app.rag.text_extractor import TextExtractionError, TextExtractor

if TYPE_CHECKING:
    from app.domain.interfaces.llm_provider import LLMProvider
    from app.domain.interfaces.repository import DocumentRepository
    from app.domain.interfaces.vector_store import VectorStore
    from app.domain.models.document import Document
    from app.rag.embeddings import EmbeddingService
    from app.rag.markdown_parser import MarkdownSection

logger = structlog.get_logger(__name__)


class IngestionService:
    """Orchestrates the document ingestion pipeline.

    Pipeline steps:
        1. Update status to PROCESSING
        2. Extract text from file content
        3. Chunk text using RecursiveChunker
        4. **Contextualise** chunks via LLM (if llm_provider is given)
        5. Generate embeddings via EmbeddingService
        6. Store embeddings in VectorStore (ChromaDB)
        7. Persist chunk records in DocumentRepository
        8. Update status to READY with chunk count

    On failure at any step:
        - Delete any partial chunks from DB
        - Set status to FAILED
        - Raise IngestionError with details

    Attributes:
        _document_repo: Port for document persistence.
        _embedding_service: Service for generating embeddings.
        _vector_store: Port for vector database operations.
        _llm_provider: Optional LLM provider for chunk contextualisation.
        _chunker: Text chunking strategy.
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_provider: LLMProvider | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        from app.config import get_settings

        settings = get_settings()
        self._document_repo = document_repo
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_provider = llm_provider
        self._chunker = RecursiveChunker(
            chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
            overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        )

    async def process_document(
        self,
        document: Document,
        file_content: bytes,
        *,
        tenant_chat_model: str | None = None,
        tenant_embedding_model: str | None = None,
    ) -> None:
        """Process a document through the full ingestion pipeline.

        Args:
            document: Domain document model with metadata.
            file_content: Raw file bytes.
            tenant_chat_model: Tenant-specific chat model for contextualisation.
            tenant_embedding_model: Tenant-specific embedding model.

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
            #   For markdown files: parse structure first, then chunk each section
            #   For all other files: chunk the raw text directly
            if document.file_type == "md":
                chunks = self._chunk_markdown(text, doc_id, document.filename)
            else:
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

            # Step 4: Contextualise chunks (Anthropic's Contextual Retrieval)
            chunk_texts = [c.content for c in chunks]
            if self._llm_provider is not None:
                chunk_texts = await self._contextualise_chunks(
                    chunk_texts=chunk_texts,
                    full_document_text=text,
                    document_filename=document.filename,
                    chat_model=tenant_chat_model,
                )

            # Step 5: Generate embeddings (on contextualised text)
            embeddings = await self._embedding_service.embed_batch(
                chunk_texts, model=tenant_embedding_model,
            )

            logger.info(
                "embeddings_generated",
                document_id=doc_id,
                embedding_count=len(embeddings),
            )

            # Step 6: Generate unique IDs and store in vector DB
            chroma_ids = [f"{doc_id}_chunk_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]

            metadatas: list[dict[str, object]] = []
            for i, chunk in enumerate(chunks):
                meta: dict[str, object] = {
                    "document_id": doc_id,
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "chunk_index": i,
                    "tenant_id": tenant_id,
                }
                # Include heading_path from markdown parsing if present
                heading_path = chunk.metadata.get("heading_path")
                if heading_path:
                    meta["heading_path"] = heading_path
                metadatas.append(meta)

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

            # Step 7: Persist chunk records in DB
            # Store the contextualised text so DB content matches what's
            # in the vector store (important for source display)
            for i, _chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=doc_id,
                    chunk_index=i,
                    content=chunk_texts[i],
                    chroma_id=chroma_ids[i],
                )
                await self._document_repo.create_chunk(db_chunk)

            # Step 8: Update status to READY
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

    async def _contextualise_chunks(
        self,
        chunk_texts: list[str],
        full_document_text: str,
        document_filename: str,
        *,
        chat_model: str | None = None,
    ) -> list[str]:
        """Add contextual prefixes to each chunk via the LLM.

        Wraps the ``contextualizer`` module. If any individual chunk
        fails, it is returned unchanged (graceful degradation).

        Args:
            chunk_texts: Raw chunk texts.
            full_document_text: Full source document.
            document_filename: Filename for context generation.

        Returns:
            Contextualised chunk texts.
        """
        from app.rag.contextualizer import contextualize_chunks

        assert self._llm_provider is not None  # noqa: S101

        logger.info(
            "contextualizing_chunks",
            filename=document_filename,
            chunk_count=len(chunk_texts),
        )

        contextualised = await contextualize_chunks(
            chunk_texts=chunk_texts,
            full_document_text=full_document_text,
            document_filename=document_filename,
            llm_provider=self._llm_provider,
            chat_model=chat_model,
        )

        logger.info(
            "chunk_contextualization_done",
            filename=document_filename,
            contextualised_count=len(contextualised),
        )

        return contextualised

    def _chunk_markdown(
        self,
        text: str,
        document_id: str,
        filename: str,
    ) -> list[Chunk]:
        """Parse markdown structure, merge small sections, then chunk.

        Strategy:
            1. Split at heading boundaries (markdown parser)
            2. **Merge** adjacent small sections until they approach
               ``chunk_size`` tokens — prevents over-chunking when
               documents have many short headings
            3. Chunk each merged group with RecursiveChunker
            4. Inject heading path from the *first* section in each
               group as metadata for retrieval filtering

        For non-markdown files, this method is never called — the
        RecursiveChunker is used directly.

        Args:
            text: Extracted markdown text.
            document_id: Document ID for metadata.
            filename: Filename for metadata.

        Returns:
            Flat list of Chunk objects with heading_path in metadata.
        """
        from app.rag.markdown_parser import parse_markdown_sections

        sections = parse_markdown_sections(text)

        if not sections:
            # Fallback: no headings found, treat as plain text
            return self._chunker.chunk(
                text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "file_type": "md",
                },
            )

        # ── Merge small sections into chunk-sized groups ──────────
        merged_groups = self._merge_small_sections(sections)

        all_chunks: list[Chunk] = []
        global_index = 0

        for group_text, heading_path in merged_groups:
            section_metadata: dict[str, object] = {
                "document_id": document_id,
                "filename": filename,
                "file_type": "md",
            }
            if heading_path:
                section_metadata["heading_path"] = heading_path

            section_chunks = self._chunker.chunk(
                group_text,
                metadata=section_metadata,
            )

            # Renumber indices to be globally sequential
            for chunk in section_chunks:
                chunk.index = global_index
                chunk.metadata["chunk_index"] = global_index
                global_index += 1

            all_chunks.extend(section_chunks)

        logger.info(
            "markdown_chunking_complete",
            document_id=document_id,
            section_count=len(sections),
            merged_groups=len(merged_groups),
            chunk_count=len(all_chunks),
        )

        return all_chunks

    def _merge_small_sections(
        self,
        sections: list[MarkdownSection],
    ) -> list[tuple[str, str]]:
        """Merge adjacent small sections up to chunk_size tokens.

        Prevents over-chunking when a document has many short headings
        (e.g., 31 sections averaging 88 tokens each).  Adjacent sections
        are combined into a single text block until the accumulated token
        count approaches ``chunk_size``.

        The heading_path of the **first** section in each merged group
        is used as the metadata for the resulting chunk(s).

        Args:
            sections: List of MarkdownSection objects from the parser.

        Returns:
            List of ``(merged_text, heading_path)`` tuples ready for
            the RecursiveChunker.
        """
        if not sections:
            return []

        # Cost of the "\n\n" separator between merged sections (~1 token)
        separator_cost = self._chunker._count_tokens("\n\n")

        groups: list[tuple[str, str]] = []
        current_texts: list[str] = [sections[0].content]
        current_tokens = self._chunker._count_tokens(sections[0].content)
        current_heading = sections[0].heading_path

        for section in sections[1:]:
            section_tokens = self._chunker._count_tokens(section.content)

            # Account for the "\n\n" separator that will join them
            merged_cost = current_tokens + separator_cost + section_tokens

            # If adding this section would exceed chunk_size, flush
            if merged_cost > self._chunker.chunk_size:
                groups.append(("\n\n".join(current_texts), current_heading))
                current_texts = [section.content]
                current_tokens = section_tokens
                current_heading = section.heading_path
            else:
                current_texts.append(section.content)
                current_tokens = merged_cost

        # Flush the last group
        if current_texts:
            groups.append(("\n\n".join(current_texts), current_heading))

        return groups

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
