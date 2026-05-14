"""Contextual Retrieval — chunk contextualization for RAG ingestion.

Implements Anthropic's "Contextual Retrieval" technique: before embedding,
each chunk is prepended with a short LLM-generated summary that situates
it within the full document.  This dramatically improves retrieval accuracy
(up to 49% reduction in retrieval failures per Anthropic's benchmarks)
because every chunk becomes self-explanatory, even out of its original
document context.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.domain.interfaces.llm_provider import LLMProvider

logger = structlog.get_logger(__name__)

# Maximum characters of the full document to include in the context
# generation prompt.  Keeps the prompt within model context limits while
# providing enough surrounding material for the LLM to reason about.
_MAX_DOC_CONTEXT_CHARS = 8000

# System prompt for context generation — highly constrained for determinism
_CONTEXT_SYSTEM_PROMPT = (
    "You generate context snippets for document chunks.\n"
    "\n"
    "RULES:\n"
    "1. Write 1-5 plain English sentences.\n"
    "2. Start with: 'This chunk is from \"<filename>\" and covers ...'\n"
    "3. Mention the document name, the section/topic, and how it "
    "relates to the rest of the document.\n"
    "4. Do NOT summarize the chunk content in detail.\n"
    "5. Do NOT repeat any text from the chunk or the document.\n"
    "6. Do NOT output headings, bullet points, markdown, or labels.\n"
    "7. Do NOT include meta-commentary like 'this context is meant to...'.\n"
    "8. Do NOT fabricate information not present in the document.\n"
    "9. Output ONLY the context sentences. Nothing before, nothing after.\n"
    "\n"
    "EXAMPLE:\n"
    "Input: filename='shipping-policy.md', chunk covers return shipping rates.\n"
    "Output: This chunk is from \"shipping-policy.md\" and covers the "
    "return shipping rate schedule. It appears in the Returns section, "
    "following the standard delivery tiers and before the international "
    "shipping restrictions."
)


async def generate_chunk_context(
    *,
    chunk_text: str,
    full_document_text: str,
    document_filename: str,
    llm_provider: LLMProvider,
) -> str:
    """Generate a contextual prefix for a single chunk.

    Uses the LLM to produce a short context that explains where this
    chunk sits within the overall document.  The result is prepended
    to the chunk text before embedding and vector storage.

    Args:
        chunk_text: The raw chunk content.
        full_document_text: The complete document text (truncated internally).
        document_filename: Original filename for reference in the context.
        llm_provider: LLM adapter for generation.

    Returns:
        The contextualised chunk: ``"<context>\\n\\n<original chunk>"``.
        If context generation fails, the original chunk is returned unchanged.
    """
    # Truncate the full document to fit within prompt budget
    doc_excerpt = full_document_text[:_MAX_DOC_CONTEXT_CHARS]

    user_prompt = (
        f"Filename: \"{document_filename}\"\n"
        f"\n"
        f"<document>\n{doc_excerpt}\n</document>\n"
        f"\n"
        f"<chunk>\n{chunk_text}\n</chunk>\n"
        f"\n"
        f"Write the context snippet for this chunk now."
    )

    try:
        context = await llm_provider.generate(
            messages=[
                {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # fully deterministic
            max_tokens=4096,
        )

        context = context.strip()
        if not context:
            return chunk_text

        return f"{context}\n\n{chunk_text}"

    except Exception:
        logger.warning(
            "chunk_context_generation_failed",
            filename=document_filename,
            chunk_preview=chunk_text[:80],
            exc_info=True,
        )
        # Graceful degradation — return the original chunk unchanged
        return chunk_text


async def contextualize_chunks(
    *,
    chunk_texts: list[str],
    full_document_text: str,
    document_filename: str,
    llm_provider: LLMProvider,
) -> list[str]:
    """Contextualise a batch of chunks from the same document.

    Processes chunks sequentially to avoid overwhelming the LLM
    provider.  Each chunk is prepended with a contextual summary.

    Args:
        chunk_texts: Raw chunk texts to contextualise.
        full_document_text: The complete source document.
        document_filename: Source filename.
        llm_provider: LLM adapter.

    Returns:
        List of contextualised chunk texts (same order as input).
    """
    contextualised: list[str] = []

    for i, chunk_text in enumerate(chunk_texts):
        logger.debug(
            "contextualizing_chunk",
            chunk_index=i,
            total_chunks=len(chunk_texts),
            filename=document_filename,
        )
        result = await generate_chunk_context(
            chunk_text=chunk_text,
            full_document_text=full_document_text,
            document_filename=document_filename,
            llm_provider=llm_provider,
        )
        contextualised.append(result)

    logger.info(
        "chunk_contextualization_complete",
        filename=document_filename,
        total_chunks=len(chunk_texts),
    )

    return contextualised
