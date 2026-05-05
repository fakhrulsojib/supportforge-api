"""Tests for application configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, create_settings, get_settings


class TestSettings:
    """Test suite for Settings configuration class."""

    def test_default_settings_load(self) -> None:
        """Settings should load with defaults when no env vars are set."""
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.app_name == "SupportForge"
        assert settings.app_env == "development"
        assert settings.app_debug is True
        assert settings.app_port == 8000

    def test_app_env_validation_accepts_valid(self) -> None:
        """Valid app_env values should be accepted."""
        for env in ("development", "staging", "production"):
            settings = Settings(app_env=env, _env_file=None)  # type: ignore[call-arg]
            assert settings.app_env == env

    def test_app_env_validation_rejects_invalid(self) -> None:
        """Invalid app_env should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid app_env"):
            Settings(app_env="invalid", _env_file=None)  # type: ignore[call-arg]

    def test_log_level_validation_accepts_valid(self) -> None:
        """Valid log levels should be accepted and uppercased."""
        for level in ("debug", "INFO", "Warning", "ERROR", "critical"):
            settings = Settings(app_log_level=level, _env_file=None)  # type: ignore[call-arg]
            assert settings.app_log_level == level.upper()

    def test_log_level_validation_rejects_invalid(self) -> None:
        """Invalid log level should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid log level"):
            Settings(app_log_level="TRACE", _env_file=None)  # type: ignore[call-arg]

    def test_computed_database_url_from_components(self) -> None:
        """computed_database_url should build URL from components when database_url is empty."""
        settings = Settings(
            database_url="",
            postgres_user="testuser",
            postgres_password="testpass",
            postgres_host="dbhost",
            postgres_port=5433,
            postgres_db="testdb",
            _env_file=None,  # type: ignore[call-arg]
        )
        expected = "postgresql+asyncpg://testuser:testpass@dbhost:5433/testdb"
        assert settings.computed_database_url == expected

    def test_computed_database_url_prefers_explicit(self) -> None:
        """computed_database_url should return explicit database_url if set."""
        explicit = "postgresql+asyncpg://explicit:url@host/db"
        settings = Settings(database_url=explicit, _env_file=None)  # type: ignore[call-arg]
        assert settings.computed_database_url == explicit

    def test_computed_redis_url_without_password(self) -> None:
        """Redis URL should omit password segment when not set."""
        settings = Settings(
            redis_url="",
            redis_host="redishost",
            redis_port=6380,
            redis_password="",
            redis_db=1,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.computed_redis_url == "redis://redishost:6380/1"

    def test_computed_redis_url_with_password(self) -> None:
        """Redis URL should include password segment when set."""
        settings = Settings(
            redis_url="",
            redis_host="redishost",
            redis_port=6380,
            redis_password="secret",
            redis_db=2,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.computed_redis_url == "redis://:secret@redishost:6380/2"

    def test_computed_redis_url_prefers_explicit(self) -> None:
        """Redis URL should return explicit value if set."""
        explicit = "redis://explicit:6379/0"
        settings = Settings(redis_url=explicit, _env_file=None)  # type: ignore[call-arg]
        assert settings.computed_redis_url == explicit

    def test_cors_origin_list_parsing(self) -> None:
        """CORS origins string should be parsed into a list."""
        settings = Settings(
            cors_origins="http://a.com, http://b.com , http://c.com",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.cors_origin_list == ["http://a.com", "http://b.com", "http://c.com"]

    def test_cors_origin_list_empty_string(self) -> None:
        """Empty CORS origins should produce an empty list."""
        settings = Settings(cors_origins="", _env_file=None)  # type: ignore[call-arg]
        assert settings.cors_origin_list == []

    def test_create_settings_factory(self) -> None:
        """create_settings should return a fresh Settings with overrides."""
        settings = create_settings(app_name="TestApp", _env_file=None)
        assert settings.app_name == "TestApp"

    def test_get_settings_returns_singleton(self) -> None:
        """get_settings should return the same instance on repeated calls."""
        from app.config import clear_settings_cache

        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        clear_settings_cache()  # cleanup

    def test_port_type_coercion(self) -> None:
        """String port values should be coerced to int."""
        settings = Settings(app_port="9000", _env_file=None)  # type: ignore[call-arg]
        assert settings.app_port == 9000
        assert isinstance(settings.app_port, int)

