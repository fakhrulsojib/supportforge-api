"""Factory for creating the configured reranker.

Reads ``reranker_enabled`` and ``reranker_model`` from config
to determine which implementation to instantiate. Falls back
to NoOpReranker if the reranker is disabled or if the required
``sentence-transformers`` library is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.domain.interfaces.reranker import Reranker

logger = structlog.get_logger(__name__)


def get_reranker() -> Reranker:
    """Create and return the configured reranker instance.

    Decision tree:
        1. ``reranker_enabled=False`` → NoOpReranker
        2. ``reranker_enabled=True`` + sentence-transformers installed
           → CrossEncoderReranker(model)
        3. ``reranker_enabled=True`` + library missing
           → NoOpReranker (with warning log)

    Returns:
        A Reranker implementation ready to use.
    """
    from app.config import get_settings

    settings = get_settings()

    if not settings.reranker_enabled:
        from app.infrastructure.reranker.noop_reranker import NoOpReranker

        logger.debug("reranker_disabled", reason="reranker_enabled=False in config")
        return NoOpReranker()

    try:
        from app.infrastructure.reranker.cross_encoder_reranker import (
            CrossEncoderReranker,
        )

        return CrossEncoderReranker(model_name=settings.reranker_model)
    except ImportError:
        from app.infrastructure.reranker.noop_reranker import NoOpReranker

        logger.warning(
            "reranker_fallback_noop",
            reason="sentence-transformers not installed",
            hint='Install via: pip install -e ".[reranker]"',
            configured_model=settings.reranker_model,
        )
        return NoOpReranker()
