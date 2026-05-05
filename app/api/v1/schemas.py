"""Backward-compatibility re-export.

Relocated to ``app.api.schemas.chat``.
"""

from app.api.schemas.chat import ChatRequest, ChatResponse, SourceCitation

__all__ = ["ChatRequest", "ChatResponse", "SourceCitation"]
