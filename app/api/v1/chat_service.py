"""Backward-compatibility re-export.

.. deprecated:: 0.2.0
    Import from ``app.domain.services.chat_service`` instead.
    This shim will be removed in Phase 4.
"""

import warnings

from app.domain.services.chat_service import ChatService

warnings.warn(
    "Importing ChatService from app.api.v1.chat_service is deprecated. Use app.domain.services.chat_service instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ChatService"]
