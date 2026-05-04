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
    ollama_chat_model: str = "qwen3:4b"
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

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Ensure app environment is valid."""
        allowed = {"development", "staging", "production"}
        lower = v.lower()
        if lower not in allowed:
            msg = f"Invalid app_env '{v}'. Must be one of: {', '.join(sorted(allowed))}"
            raise ValueError(msg)
        return lower

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


def get_settings(**overrides: Any) -> Settings:
    """Create a Settings instance with optional overrides for testing."""
    return Settings(**overrides)
