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


@pytest.mark.asyncio
async def test_cache_hit_avoids_refetch_within_ttl(monkeypatch):
    """A second lookup within the TTL is served from cache (no second Clerk call)."""
    calls = {"n": 0}
    payload = {
        "primary_email_address_id": "e1",
        "email_addresses": [
            {"id": "e1", "email_address": "a@phenomehealth.org",
             "verification": {"status": "verified"}},
        ],
    }
    async def fake_get(*a, **k):
        calls["n"] += 1
        return _Resp(payload)
    monkeypatch.setattr(ci, "_clerk_get", fake_get)
    assert await ci.get_verified_email({"sub": "u"}) == "a@phenomehealth.org"
    assert await ci.get_verified_email({"sub": "u"}) == "a@phenomehealth.org"
    assert calls["n"] == 1  # cached, not re-fetched


@pytest.mark.asyncio
async def test_cache_expires_and_refetches_revoked_email(monkeypatch):
    """After the TTL lapses, a revoked trusted email is re-fetched and drops to None."""
    verified = {
        "primary_email_address_id": "e1",
        "email_addresses": [
            {"id": "e1", "email_address": "a@phenomehealth.org",
             "verification": {"status": "verified"}},
        ],
    }
    revoked = {
        "primary_email_address_id": "e1",
        "email_addresses": [
            {"id": "e1", "email_address": "a@phenomehealth.org",
             "verification": {"status": "unverified"}},
        ],
    }
    state = {"payload": verified}
    async def fake_get(*a, **k): return _Resp(state["payload"])
    monkeypatch.setattr(ci, "_clerk_get", fake_get)

    clock = {"t": 1000.0}
    monkeypatch.setattr(ci.time, "monotonic", lambda: clock["t"])

    assert await ci.get_verified_email({"sub": "u"}) == "a@phenomehealth.org"
    # Clerk revokes verification; advance past the TTL so the cache entry expires.
    state["payload"] = revoked
    clock["t"] += ci._CACHE_TTL_SECONDS + 1
    assert await ci.get_verified_email({"sub": "u"}) is None
