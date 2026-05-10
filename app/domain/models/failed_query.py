"""Domain model for failed queries.

Tracks user queries that the RAG pipeline could not answer satisfactorily,
enabling admins to identify knowledge gaps and improve the knowledge base.

Pure Pydantic model — NO framework imports.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.enums import EscalationTrigger, FailureReason


class FailedQuery(BaseModel):
    """A user query that failed to produce a satisfactory answer.

    Created when the RAG pipeline escalates (no relevant docs),
    returns low-confidence answers, encounters LLM errors, or times out.
    """

    id: str = ""
    tenant_id: str = ""
    conversation_id: str = ""
    message_id: str = ""
    query_text: str = Field(..., min_length=1)
    failure_reason: FailureReason
    retrieved_doc_count: int = 0
    max_relevance_score: float = 0.0
    escalation_trigger: EscalationTrigger = EscalationTrigger.NONE
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    resolved_by: str = ""
