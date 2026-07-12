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
