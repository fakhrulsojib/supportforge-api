"""Application configuration using Pydantic Settings.

All environment variables are validated at startup. Missing required
variables will raise a clear error before the server starts.
"""

from __future__ import annotations

from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SupportForge API configuration.

    Loads values from environment variables and `.env` file.
    Every field is typed and validated at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "SupportForge"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"  # noqa: S104
    app_port: int = 8000
    app_log_level: str = "DEBUG"
    secret_key: str = "change-me-to-a-random-secret-key"  # noqa: S105

    # ── Ollama (Self-Hosted LLM) ─────────────────────────────────
    ollama_base_url: str = "https://localhost:11434"
    ollama_chat_model: str = "gemma3:4b"
    ollama_embedding_model: str = "nomic-embed-text"
    cf_ollama_id: str = ""
    cf_ollama_secret: str = ""

    # ── PostgreSQL ───────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "supportforge"
    postgres_user: str = "supportforge"
    postgres_password: str = "change-me"  # noqa: S105
    database_url: str = ""

    # ── Redis ────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_url: str = ""

    # ── ChromaDB ─────────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_prefix: str = "tenant_"

    # ── Ingestion Pipeline ───────────────────────────────────────
    ingestion_max_concurrent: int = 1
    chunk_size: int = 500          # Target chunk size in tokens
    chunk_overlap: int = 75        # Overlap between chunks in tokens (15%)

    # ── Retrieval Pipeline ────────────────────────────────────────
    # Toggle components on/off
    bm25_enabled: bool = True              # Enable BM25 keyword search alongside vector
    reranker_enabled: bool = False         # Enable cross-encoder reranking (needs sentence-transformers)

    # Candidate counts
    retrieval_k_per_method: int = 20       # Candidates retrieved per method (vector, BM25)
    retrieval_final_k: int = 5             # Final chunks sent to LLM prompt

    # Weighted RRF fusion tuning
    retrieval_rrf_k: int = 60              # RRF smoothing constant (standard=60)
    retrieval_vector_weight: float = 0.5   # Vector search priority in fusion (0.0–1.0)
    retrieval_bm25_weight: float = 0.5     # BM25 search priority in fusion (0.0–1.0)

    # Reranker configuration
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Conversation History ─────────────────────────────────────
    # After this many user-assistant pairs, older history is summarized
    history_summarize_threshold: int = 10

    # ── JWT Authentication ───────────────────────────────────────
    jwt_secret_key: str = "change-me-to-another-random-secret"  # noqa: S105
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_per_user: int = 60
    rate_limit_per_tenant: int = 1000

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Superadmin Bootstrap (optional) ──────────────────────────
    # If both are set, the app will auto-create a "management" tenant
    # and superadmin user on startup (idempotent — skips if exists).
    superadmin_email: str = ""
    superadmin_password: str = ""  # noqa: S105

    @field_validator("app_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return upper

    @field_validator("ingestion_max_concurrent")
    @classmethod
    def validate_ingestion_concurrency(cls, v: int) -> int:
        """Ensure ingestion concurrency is at least 1."""
        if v < 1:
            msg = f"INGESTION_MAX_CONCURRENT must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Ensure app environment is valid."""
        allowed = {"development", "staging", "production", "test", "testing"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"Invalid app_env '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return lower

    @field_validator("retrieval_k_per_method", "retrieval_final_k")
    @classmethod
    def validate_retrieval_k_values(cls, v: int, info: Any) -> int:
        """Ensure retrieval k values are at least 1."""
        if v < 1:
            msg = f"{info.field_name} must be >= 1, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("retrieval_rrf_k")
    @classmethod
    def validate_rrf_k(cls, v: int) -> int:
        """Ensure RRF smoothing constant is positive (prevents division by zero)."""
        if v < 1:
            msg = f"retrieval_rrf_k must be >= 1, got {v} (k=0 causes division by zero)"
            raise ValueError(msg)
        return v

    @field_validator("retrieval_vector_weight", "retrieval_bm25_weight")
    @classmethod
    def validate_retrieval_weights(cls, v: float, info: Any) -> float:
        """Ensure retrieval weights are non-negative."""
        if v < 0:
            msg = f"{info.field_name} must be >= 0.0, got {v}"
            raise ValueError(msg)
        return v

    @property
    def computed_database_url(self) -> str:
        """Build database URL from components if not explicitly set."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def computed_redis_url(self) -> str:
        """Build Redis URL from components if not explicitly set."""
        if self.redis_url:
            return self.redis_url
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# ── Singleton settings cache ────────────────────────────────────

_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """Return the cached application settings singleton.

    The Settings object is created once on first call and reused
    for all subsequent calls, avoiding repeated ``.env`` file
    parsing and validation on every request.

    Use :func:`clear_settings_cache` in tests that need a fresh
    instance, and :func:`create_settings` to build ad-hoc instances
    with overrides.
    """
    global _settings_cache  # noqa: PLW0603
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def clear_settings_cache() -> None:
    """Reset the cached settings (for test isolation)."""
    global _settings_cache  # noqa: PLW0603
    _settings_cache = None


def create_settings(**overrides: Any) -> Settings:
    """Create a fresh Settings instance with optional overrides.

    Unlike :func:`get_settings`, this always creates a **new**
    instance and does NOT update the singleton cache. Useful in
    tests that need specific configuration.
    """
    return Settings(**overrides)
