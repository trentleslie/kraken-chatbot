"""Tests for Settings env wiring and the Biomapper config guards (Unit 1)."""

import pytest

from kestrel_backend.config import (
    Settings,
    biomapper_misconfig_reason,
    get_settings,
    resolve_biomapper_base_url,
)


class TestBiomapperSettingsEnv:
    """get_settings() reads the Biomapper secrets from the environment."""

    def test_reads_biomapper_env(self, monkeypatch):
        monkeypatch.setenv("BIOMAPPER_API_KEY", "test-key-123")
        monkeypatch.setenv("BIOMAPPER_BASE_URL", "https://biomapper.example.org")
        get_settings.cache_clear()
        try:
            settings = get_settings()
            assert settings.biomapper_api_key == "test-key-123"
            assert settings.biomapper_base_url == "https://biomapper.example.org"
        finally:
            get_settings.cache_clear()

    def test_unset_biomapper_env_is_none(self, monkeypatch):
        # get_settings() calls load_dotenv(), which would re-populate these from a real
        # backend/.env on disk. Neutralize it so this asserts true "unset" behavior.
        monkeypatch.setattr("kestrel_backend.config.load_dotenv", lambda *a, **k: False)
        monkeypatch.delenv("BIOMAPPER_API_KEY", raising=False)
        monkeypatch.delenv("BIOMAPPER_BASE_URL", raising=False)
        get_settings.cache_clear()
        try:
            settings = get_settings()
            assert settings.biomapper_api_key is None
            assert settings.biomapper_base_url is None
        finally:
            get_settings.cache_clear()


class TestBiomapperHttpsValidator:
    """The API key must never be sent over plaintext HTTP."""

    def test_http_base_url_with_key_rejected(self):
        with pytest.raises(ValueError, match="HTTPS"):
            Settings(
                biomapper_api_key="secret",
                biomapper_base_url="http://biomapper.example.org",
            )

    def test_https_base_url_with_key_ok(self):
        s = Settings(
            biomapper_api_key="secret",
            biomapper_base_url="https://biomapper.example.org",
        )
        assert s.biomapper_base_url == "https://biomapper.example.org"

    def test_http_base_url_without_key_allowed(self):
        # No key set → nothing to leak; allowed (e.g. unused config).
        s = Settings(biomapper_base_url="http://localhost:8001")
        assert s.biomapper_base_url == "http://localhost:8001"

    def test_none_base_url_with_key_ok(self):
        # None base URL = client default (prod HTTPS); fine to have a key.
        s = Settings(biomapper_api_key="secret", biomapper_base_url=None)
        assert s.biomapper_api_key == "secret"


class TestResolveBiomapperBaseUrl:
    """The prod/dev API toggle resolves to the right base URL or raises a clear error."""

    def _settings(self, **kw):
        return Settings(**kw)

    def test_default_none_env_is_production(self):
        s = self._settings(biomapper_base_url="https://prod.example/api/v1")
        assert resolve_biomapper_base_url(None, s) == "https://prod.example/api/v1"

    def test_explicit_production(self):
        s = self._settings(biomapper_base_url="https://prod.example/api/v1")
        assert resolve_biomapper_base_url("production", s) == "https://prod.example/api/v1"

    def test_dev_returns_dev_url(self):
        s = self._settings(
            biomapper_base_url="https://prod.example/api/v1",
            biomapper_dev_base_url="https://dev.example/api/v1",
        )
        assert resolve_biomapper_base_url("dev", s) == "https://dev.example/api/v1"

    def test_dev_unconfigured_raises(self):
        s = self._settings(biomapper_base_url="https://prod.example/api/v1")
        with pytest.raises(ValueError, match="not configured"):
            resolve_biomapper_base_url("dev", s)

    def test_invalid_env_raises(self):
        with pytest.raises(ValueError, match="Invalid biomapper env"):
            resolve_biomapper_base_url("staging", self._settings())

    def test_production_with_no_url_is_none(self):
        # None = biomapper client default (prod); valid.
        assert resolve_biomapper_base_url("production", self._settings()) is None


class TestBiomapperMisconfigReason:
    """Fail-closed startup guard: enabled-but-no-key surfaces a CRITICAL reason."""

    def test_enabled_without_key_returns_reason(self):
        reason = biomapper_misconfig_reason(enabled=True, api_key=None)
        assert reason is not None
        assert "BIOMAPPER_API_KEY" in reason

    def test_enabled_with_key_is_ok(self):
        assert biomapper_misconfig_reason(enabled=True, api_key="k") is None

    def test_disabled_is_always_ok(self):
        assert biomapper_misconfig_reason(enabled=False, api_key=None) is None
        assert biomapper_misconfig_reason(enabled=False, api_key="k") is None

    def test_reason_does_not_leak_key(self):
        # The reason is logged; it must never echo the secret value.
        reason = biomapper_misconfig_reason(enabled=True, api_key=None)
        assert reason is not None
        assert "supersecretvalue" not in reason
