import shutil

import pytest
from kestrel_backend import byok
from kestrel_backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    s = Settings(byok_trusted_email_domains=["phenomehealth.org"],
                 server_anthropic_api_key="sk-server")
    monkeypatch.setattr(byok, "get_settings", lambda: s)


def test_byok_key_wins_even_for_trusted_user():
    key, source = byok.resolve_effective_key("sk-user", "trent@phenomehealth.org")
    assert (key, source) == ("sk-user", "byok")


def test_trusted_user_no_byok_uses_server_key():
    key, source = byok.resolve_effective_key(None, "trent@phenomehealth.org")
    assert (key, source) == ("sk-server", "server")


def test_untrusted_no_byok_raises():
    with pytest.raises(byok.NeedsKeyError):
        byok.resolve_effective_key(None, "someone@gmail.com")


def test_no_verified_email_is_untrusted():
    with pytest.raises(byok.NeedsKeyError):
        byok.resolve_effective_key(None, None)


def test_domain_match_is_case_insensitive_exact_suffix():
    assert byok.is_trusted_email("A@Phenomehealth.org") is True
    assert byok.is_trusted_email("x@evil-phenomehealth.org") is False
    assert byok.is_trusted_email("x@phenomehealth.org.evil.com") is False


def test_resolve_returns_provider_anthropic():
    key, provider, source = byok.resolve_effective_key_and_provider("sk-user", "t@phenomehealth.org")
    assert (key, provider, source) == ("sk-user", "anthropic", "byok")


def test_build_agent_env_empty_when_no_key():
    byok.current_api_key.set(None)
    assert byok.build_agent_env() == {}


def test_build_agent_env_maps_keys(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "litellm_master_key", "sk-proxy", raising=False)
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    byok.current_api_key.set("sk-user")
    env = byok.build_agent_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-user"          # user key → x-api-key (forwarded)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-proxy"      # proxy-auth → Authorization (stripped)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"


def test_build_agent_env_raises_when_proxy_configured_without_master_key(monkeypatch):
    """Proxy mode with a missing master key must fail fast, not silently omit
    ANTHROPIC_AUTH_TOKEN (which would make every request 401 against the proxy)."""
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(s, "litellm_master_key", "", raising=False)
    byok.current_api_key.set("sk-user")
    with pytest.raises(RuntimeError):
        byok.build_agent_env()


def test_build_agent_env_direct_mode_when_no_base_url(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "", raising=False)
    byok.current_api_key.set("sk-user")
    env = byok.build_agent_env()
    assert env == {"ANTHROPIC_API_KEY": "sk-user"}        # no proxy → legacy direct behavior


def test_system_cli_path_is_string_or_none():
    assert byok.system_cli_path() in (None, shutil.which("claude"))


def test_agent_cli_path_raises_when_proxy_configured_and_cli_missing(monkeypatch):
    """Proxy mode with no system `claude` binary must fail fast — the bundled SDK binary
    ignores ANTHROPIC_BASE_URL and would silently bypass the proxy straight to Anthropic."""
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(byok, "system_cli_path", lambda: None)
    with pytest.raises(RuntimeError):
        byok.agent_cli_path()


def test_agent_cli_path_returns_path_when_proxy_configured_and_cli_present(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "http://127.0.0.1:4000", raising=False)
    monkeypatch.setattr(byok, "system_cli_path", lambda: "/usr/local/bin/claude")
    assert byok.agent_cli_path() == "/usr/local/bin/claude"


def test_agent_cli_path_returns_none_when_no_proxy_even_if_cli_missing(monkeypatch):
    s = byok.get_settings()
    monkeypatch.setattr(s, "kraken_llm_base_url", "", raising=False)
    monkeypatch.setattr(byok, "system_cli_path", lambda: None)
    assert byok.agent_cli_path() is None
