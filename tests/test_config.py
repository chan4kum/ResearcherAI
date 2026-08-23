import os
from unittest.mock import patch

from app.config import Settings, clear_settings_cache, get_settings


def test_settings_defaults() -> None:
    """Verify default settings instantiation."""
    settings = Settings()
    assert settings.app_name == "Enterprise Agentic Research & Knowledge Platform"
    assert settings.app_version == "0.1.0"
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.port == 8000


def test_settings_environment_override() -> None:
    """Verify settings loaded from environment variables."""
    with patch.dict(
        os.environ,
        {
            "APP_NAME": "Overridden Platform",
            "APP_ENV": "production",
            "PORT": "9000",
            "DEBUG": "false",
        },
    ):
        clear_settings_cache()
        settings = get_settings()
        assert settings.app_name == "Overridden Platform"
        assert settings.app_env == "production"
        assert settings.port == 9000
        assert settings.debug is False
        assert settings.is_production is True
        assert settings.is_development is False
    clear_settings_cache()
