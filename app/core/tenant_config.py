"""Tenant configuration helpers — shared readers for per-tenant settings.

Centralised extraction of model selections and other tenant config
values from ``config_json``.  All API entry-points and workers should
use these helpers instead of inline dict reads to avoid duplication
and ensure consistent validation.
"""

from __future__ import annotations

from dataclasses import dataclass


# Keys used in tenant config_json
CONFIG_CHAT_MODEL = "chat_model"
CONFIG_EMBEDDING_MODEL = "embedding_model"


@dataclass(frozen=True, slots=True)
class TenantModelConfig:
    """Resolved per-tenant model selections.

    Attributes:
        chat_model: Tenant's chat model override, or None for server default.
        embedding_model: Tenant's embedding model override, or None for server default.
    """

    chat_model: str | None = None
    embedding_model: str | None = None


def resolve_tenant_models(config_json: dict | None) -> TenantModelConfig:
    """Extract chat and embedding model overrides from tenant config.

    Validates that each value is a non-empty string before accepting it.
    Returns ``None`` for either field if not configured, which signals
    downstream callers to fall back to the server's global default.

    Args:
        config_json: Tenant's ``config_json`` dict (may be None).

    Returns:
        TenantModelConfig with resolved model selections.
    """
    if not config_json:
        return TenantModelConfig()

    chat_model: str | None = None
    raw_chat = config_json.get(CONFIG_CHAT_MODEL)
    if isinstance(raw_chat, str) and raw_chat:
        chat_model = raw_chat

    embedding_model: str | None = None
    raw_embed = config_json.get(CONFIG_EMBEDDING_MODEL)
    if isinstance(raw_embed, str) and raw_embed:
        embedding_model = raw_embed

    return TenantModelConfig(chat_model=chat_model, embedding_model=embedding_model)
