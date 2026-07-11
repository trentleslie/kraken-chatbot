import pytest
from kestrel_backend import clerk_identity as ci
from kestrel_backend.config import Settings


class _Resp:
    def __init__(self, payload): self._p = payload; self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self._p


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(ci, "get_settings", lambda: Settings(clerk_secret_key="sk_test"))
    ci._email_cache.clear()


@pytest.mark.asyncio
async def test_returns_verified_primary_email(monkeypatch):
    payload = {
        "primary_email_address_id": "idb- 1",
        "email_addresses": [
            {"id": "idb- 1", "email_address": "trent@phenomehealth.org",
             "verification": {"status": "verified"}},
        ],
    }
    async def fake_get(*a, **k): return _Resp(payload)
    monkeypatch.setattr(ci, "_clerk_get", fake_get)
    assert await ci.get_verified_email({"sub": "user_1"}) == "trent@phenomehealth.org"


@pytest.mark.asyncio
async def test_unverified_primary_returns_none(monkeypatch):
    payload = {
        "primary_email_address_id": "e1",
        "email_addresses": [
            {"id": "e1", "email_address": "x@phenomehealth.org",
             "verification": {"status": "unverified"}},
        ],
    }
    async def fake_get(*a, **k): return _Resp(payload)
    monkeypatch.setattr(ci, "_clerk_get", fake_get)
    assert await ci.get_verified_email({"sub": "user_2"}) is None


@pytest.mark.asyncio
async def test_missing_sub_returns_none():
    assert await ci.get_verified_email({}) is None
