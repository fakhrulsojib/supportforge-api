"""Tenant configuration helpers — shared readers for per-tenant settings.

Centralised extraction of model selections and other tenant config
values from ``config_json``.  All API entry-points and workers should
use these helpers instead of inline dict reads to avoid duplication
and ensure consistent validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.core.crypto import decrypt_value

logger = structlog.get_logger(__name__)

# Keys used in tenant config_json
CONFIG_CHAT_MODEL = "chat_model"
CONFIG_EMBEDDING_MODEL = "embedding_model"
CONFIG_CHAT_PROVIDER = "chat_provider"
CONFIG_GEMINI_API_KEY = "gemini_api_key"
CONFIG_GEMINI_EMBEDDING_API_KEY = "gemini_embedding_api_key"
CONFIG_EMBEDDING_PROVIDER = "embedding_provider"

# Model-name prefixes that identify the Gemini provider
GEMINI_MODEL_PREFIXES: tuple[str, ...] = ("gemini-",)


@dataclass(frozen=True, slots=True)
class TenantModelConfig:
    """Resolved per-tenant model selections.

    Attributes:
        chat_model: Tenant's chat model override, or None for server default.
        embedding_model: Tenant's embedding model override, or None for server default.
        chat_provider: Provider identifier ("ollama" | "gemini"), or None for default.
        embedding_provider: Embedding provider ("ollama" | "gemini"), or None for default.
        gemini_api_key: Decrypted Gemini API key for chat, or None.
        gemini_embedding_api_key: Decrypted Gemini API key for embeddings, or None.
    """

    chat_model: str | None = None
    embedding_model: str | None = None
    chat_provider: str | None = None
    embedding_provider: str | None = None
    gemini_api_key: str | None = None
    gemini_embedding_api_key: str | None = None


def _detect_provider(model_name: str | None) -> str | None:
    """Auto-detect provider from model name prefix.

    Args:
        model_name: The model identifier (chat or embedding).

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

    # ── Chat provider resolution ─────────────────────────────────
    chat_provider: str | None = None
    raw_provider = config_json.get(CONFIG_CHAT_PROVIDER)
    if isinstance(raw_provider, str) and raw_provider:
        chat_provider = raw_provider
    elif chat_model:
        # Auto-detect from model name if provider not explicitly set
        chat_provider = _detect_provider(chat_model)

    # ── Embedding provider resolution ────────────────────────────
    embedding_provider: str | None = None
    raw_embed_provider = config_json.get(CONFIG_EMBEDDING_PROVIDER)
    if isinstance(raw_embed_provider, str) and raw_embed_provider:
        embedding_provider = raw_embed_provider
    elif embedding_model:
        # Auto-detect from model name
        embedding_provider = _detect_provider(embedding_model)

    # ── Gemini API key (encrypted in config_json) ───────────────
    gemini_api_key: str | None = None
    raw_key = config_json.get(CONFIG_GEMINI_API_KEY)
    if isinstance(raw_key, str) and raw_key and encryption_key:
        try:
            gemini_api_key = decrypt_value(raw_key, encryption_key)
        except Exception:
            logger.warning(
                "tenant_gemini_key_decrypt_failed",
                reason="invalid_ciphertext",
            )

    # ── Gemini Embedding API key (separate, encrypted) ──────────
    gemini_embedding_api_key: str | None = None
    raw_embed_key = config_json.get(CONFIG_GEMINI_EMBEDDING_API_KEY)
    if isinstance(raw_embed_key, str) and raw_embed_key and encryption_key:
        try:
            gemini_embedding_api_key = decrypt_value(raw_embed_key, encryption_key)
        except Exception:
            logger.warning(
                "tenant_gemini_embedding_key_decrypt_failed",
                reason="invalid_ciphertext",
            )

    logger.debug(
        "tenant_models_resolved",
        chat_model=chat_model,
        embedding_model=embedding_model,
        chat_provider=chat_provider,
        embedding_provider=embedding_provider,
        has_gemini_key=gemini_api_key is not None,
        has_gemini_embedding_key=gemini_embedding_api_key is not None,
    )

    return TenantModelConfig(
        chat_model=chat_model,
        embedding_model=embedding_model,
        chat_provider=chat_provider,
        embedding_provider=embedding_provider,
        gemini_api_key=gemini_api_key,
        gemini_embedding_api_key=gemini_embedding_api_key,
    )


# ── Voice configuration keys ────────────────────────────────────
CONFIG_VOICE_ENABLED = "voice_enabled"
CONFIG_STT_PROVIDER = "stt_provider"
CONFIG_STT_API_KEY = "stt_api_key"
CONFIG_TTS_PROVIDER = "tts_provider"
CONFIG_TTS_API_KEY = "tts_api_key"
CONFIG_TTS_VOICE = "tts_voice"
CONFIG_MAX_VOICE_SESSIONS = "max_voice_sessions"


@dataclass(frozen=True, slots=True)
class TenantVoiceConfig:
    """Resolved per-tenant voice settings.

    Attributes:
        voice_enabled: Whether voice is available for this tenant.
        stt_provider: STT provider identifier, or None if disabled.
        stt_api_key: Decrypted STT API key (cloud only), or None.
        tts_provider: TTS provider identifier, or None if disabled.
        tts_api_key: Decrypted TTS API key (cloud only), or None.
        tts_voice: Voice model to use for TTS.
        max_voice_sessions: Concurrent voice session limit.
    """

    voice_enabled: bool = False
    stt_provider: str | None = None
    stt_api_key: str | None = None
    tts_provider: str | None = None
    tts_api_key: str | None = None
    tts_voice: str = "en_US-lessac-medium"
    max_voice_sessions: int = 3


def resolve_tenant_voice_config(
    config_json: dict | None,
    *,
    encryption_key: str | None = None,
    stt_locally_available: bool = False,
    tts_locally_available: bool = False,
) -> TenantVoiceConfig:
    """Resolve per-tenant voice settings with three-tier fallback.

    Resolution tiers:
        1. **Cloud:** Tenant set ``voice_enabled=true`` + cloud API key
           → use the specified cloud STT/TTS provider.
        2. **Local:** Tenant set ``voice_enabled=true``, no API key,
           but local STT+TTS are both available → use whisper/piper.
        3. **Disabled:** Neither cloud nor local → ``voice_enabled=False``.

    This function **never raises**. Invalid config values are silently
    replaced with defaults. Decryption failures log a warning and fall
    through to the next tier.

    Args:
        config_json: Tenant's ``config_json`` dict (may be None).
        encryption_key: Secret for decrypting stored API keys.
        stt_locally_available: Whether local STT (whisper) is loaded.
        tts_locally_available: Whether local TTS (piper) is loaded.

    Returns:
        TenantVoiceConfig — always valid, never raises.
    """
    if not config_json:
        return TenantVoiceConfig()

    try:
        # Check if voice is enabled
        raw_enabled = config_json.get(CONFIG_VOICE_ENABLED)
        if raw_enabled is not True:
            return TenantVoiceConfig()

        # Read optional overrides
        tts_voice = config_json.get(CONFIG_TTS_VOICE, "en_US-lessac-medium")
        if not isinstance(tts_voice, str) or not tts_voice:
            tts_voice = "en_US-lessac-medium"

        raw_max = config_json.get(CONFIG_MAX_VOICE_SESSIONS, 3)
        try:
            max_sessions = max(1, int(raw_max))
        except (ValueError, TypeError):
            max_sessions = 3

        # ── Tier 1: Cloud providers with API keys ────────────────
        stt_provider = config_json.get(CONFIG_STT_PROVIDER)
        stt_api_key = None
        if isinstance(stt_provider, str) and stt_provider:
            raw_stt_key = config_json.get(CONFIG_STT_API_KEY)
            if isinstance(raw_stt_key, str) and raw_stt_key:
                if encryption_key:
                    try:
                        stt_api_key = decrypt_value(raw_stt_key, encryption_key)
                    except Exception:
                        logger.warning("tenant_stt_key_decrypt_failed")
                        stt_api_key = None
                else:
                    # Plaintext key (dev mode)
                    stt_api_key = raw_stt_key

        tts_provider = config_json.get(CONFIG_TTS_PROVIDER)
        tts_api_key = None
        if isinstance(tts_provider, str) and tts_provider:
            raw_tts_key = config_json.get(CONFIG_TTS_API_KEY)
            if isinstance(raw_tts_key, str) and raw_tts_key:
                if encryption_key:
                    try:
                        tts_api_key = decrypt_value(raw_tts_key, encryption_key)
                    except Exception:
                        logger.warning("tenant_tts_key_decrypt_failed")
                        tts_api_key = None
                else:
                    tts_api_key = raw_tts_key

        # Cloud STT+TTS both have keys → Tier 1
        if stt_api_key and tts_api_key:
            return TenantVoiceConfig(
                voice_enabled=True,
                stt_provider=stt_provider,
                stt_api_key=stt_api_key,
                tts_provider=tts_provider,
                tts_api_key=tts_api_key,
                tts_voice=tts_voice,
                max_voice_sessions=max_sessions,
            )

        # ── Tier 2: Local providers available ────────────────────
        if stt_locally_available and tts_locally_available:
            return TenantVoiceConfig(
                voice_enabled=True,
                stt_provider="whisper",
                tts_provider="piper",
                tts_voice=tts_voice,
                max_voice_sessions=max_sessions,
            )

        # ── Tier 3: Disabled ─────────────────────────────────────
        return TenantVoiceConfig(
            tts_voice=tts_voice,
            max_voice_sessions=max_sessions,
        )

    except Exception:
        logger.warning("tenant_voice_config_resolution_failed", exc_info=True)
        return TenantVoiceConfig()

