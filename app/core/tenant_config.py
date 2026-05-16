"""Tenant configuration helpers — shared readers for per-tenant settings.

Centralised extraction of model selections and other tenant config
values from ``config_json``.  All API entry-points and workers should
use these helpers instead of inline dict reads to avoid duplication
and ensure consistent validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.crypto import decrypt_value

logger = logging.getLogger(__name__)

# Keys used in tenant config_json
CONFIG_CHAT_MODEL = "chat_model"
CONFIG_EMBEDDING_MODEL = "embedding_model"
CONFIG_CHAT_PROVIDER = "chat_provider"
CONFIG_GEMINI_API_KEY = "gemini_api_key"

# Model-name prefixes that identify the Gemini provider
GEMINI_MODEL_PREFIXES: tuple[str, ...] = ("gemini-",)


@dataclass(frozen=True, slots=True)
class TenantModelConfig:
    """Resolved per-tenant model selections.

    Attributes:
        chat_model: Tenant's chat model override, or None for server default.
        embedding_model: Tenant's embedding model override, or None for server default.
        chat_provider: Provider identifier ("ollama" | "gemini"), or None for default.
        gemini_api_key: Decrypted Gemini API key for runtime use, or None.
    """

    chat_model: str | None = None
    embedding_model: str | None = None
    chat_provider: str | None = None
    gemini_api_key: str | None = None


def _detect_provider(model_name: str | None) -> str | None:
    """Auto-detect provider from model name prefix.

    Args:
        model_name: The chat model identifier.

    Returns:
        ``"gemini"`` if the model name starts with a known Gemini prefix,
        ``None`` otherwise (caller falls back to Ollama).
    """
    if not model_name:
        return None
    for prefix in GEMINI_MODEL_PREFIXES:
        if model_name.startswith(prefix):
            return "gemini"
    return None


def resolve_tenant_models(
    config_json: dict | None,
    *,
    encryption_key: str | None = None,
) -> TenantModelConfig:
    """Extract chat and embedding model overrides from tenant config.

    Validates that each value is a non-empty string before accepting it.
    Returns ``None`` for either field if not configured, which signals
    downstream callers to fall back to the server's global default.

    Args:
        config_json: Tenant's ``config_json`` dict (may be None).
        encryption_key: Application secret key for decrypting the stored
            Gemini API key.  If not provided, encrypted keys cannot be
            decrypted and ``gemini_api_key`` will be ``None``.

    Returns:
        TenantModelConfig with resolved model selections.
    """
    if not config_json:
        return TenantModelConfig()

    # ── Model selections ────────────────────────────────────────
    chat_model: str | None = None
    raw_chat = config_json.get(CONFIG_CHAT_MODEL)
    if isinstance(raw_chat, str) and raw_chat:
        chat_model = raw_chat

    embedding_model: str | None = None
    raw_embed = config_json.get(CONFIG_EMBEDDING_MODEL)
    if isinstance(raw_embed, str) and raw_embed:
        embedding_model = raw_embed

    # ── Provider resolution ─────────────────────────────────────
    chat_provider: str | None = None
    raw_provider = config_json.get(CONFIG_CHAT_PROVIDER)
    if isinstance(raw_provider, str) and raw_provider:
        chat_provider = raw_provider
    elif chat_model:
        # Auto-detect from model name if provider not explicitly set
        chat_provider = _detect_provider(chat_model)

    # ── Gemini API key (encrypted in config_json) ───────────────
    gemini_api_key: str | None = None
    raw_key = config_json.get(CONFIG_GEMINI_API_KEY)
    if isinstance(raw_key, str) and raw_key and encryption_key:
        try:
            gemini_api_key = decrypt_value(raw_key, encryption_key)
        except Exception:
            logger.warning(
                "tenant_gemini_key_decrypt_failed",
                extra={"reason": "invalid_ciphertext"},
            )

    return TenantModelConfig(
        chat_model=chat_model,
        embedding_model=embedding_model,
        chat_provider=chat_provider,
        gemini_api_key=gemini_api_key,
    )
