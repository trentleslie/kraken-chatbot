"""S2 429 handling: exponential backoff before giving up (S2 API-key terms require it)."""

import importlib.util
from pathlib import Path

import httpx
import pytest
import respx

# Load the real module under a private name: test_literature_grounding.py replaces
# kestrel_backend.semantic_scholar in sys.modules with a mock, so a normal import is order-dependent.
_SRC = Path(__file__).resolve().parents[1] / "src" / "kestrel_backend" / "semantic_scholar.py"
_spec = importlib.util.spec_from_file_location("_s2_under_test", _SRC)
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

OK_BODY = {"data": [{"paperId": "p1", "title": "BCAA and insulin resistance"}]}


@pytest.fixture(autouse=True)
def reset_cooldown(monkeypatch):
    monkeypatch.setattr(s2, "_cooldown_until", 0.0)


@pytest.fixture
def sleeps(monkeypatch):
    """Record every asyncio.sleep the client makes instead of waiting."""
    calls: list[float] = []

    async def fake_sleep(seconds):
        calls.append(seconds)

    monkeypatch.setattr(s2.asyncio, "sleep", fake_sleep)
    return calls


@respx.mock
async def test_429_then_success_backs_off_and_returns_papers(sleeps):
    route = respx.get(s2.S2_SEARCH_ENDPOINT).mock(
        side_effect=[httpx.Response(429), httpx.Response(429), httpx.Response(200, json=OK_BODY)]
    )
    papers = await s2.search_papers("leucine adiponectin")
    assert [p["paperId"] for p in papers] == ["p1"]
    assert route.call_count == 3
    assert sleeps[1:] == [2.0, 4.0]  # sleeps[0] is the normal pacing delay


@respx.mock
async def test_retry_after_header_is_honored(sleeps):
    respx.get(s2.S2_SEARCH_ENDPOINT).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=OK_BODY)]
    )
    await s2.search_papers("leucine adiponectin")
    assert sleeps[1] == 7


@respx.mock
async def test_gives_up_after_max_retries_with_rate_limit_error(sleeps):
    route = respx.get(s2.S2_SEARCH_ENDPOINT).mock(return_value=httpx.Response(429))
    with pytest.raises(s2.S2RateLimitError):
        await s2.search_papers("leucine adiponectin")
    assert route.call_count == s2.S2_MAX_RETRIES + 1
    assert sleeps[1:] == [2.0, 4.0, 8.0]


@respx.mock
async def test_retry_after_is_capped(sleeps):
    respx.get(s2.S2_SEARCH_ENDPOINT).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "600"}), httpx.Response(200, json=OK_BODY)]
    )
    await s2.search_papers("leucine adiponectin")
    assert sleeps[1] == s2.S2_BACKOFF_CAP


@respx.mock
async def test_exhausted_retries_fail_fast_during_cooldown(sleeps):
    route = respx.get(s2.S2_SEARCH_ENDPOINT).mock(return_value=httpx.Response(429))
    with pytest.raises(s2.S2RateLimitError):
        await s2.search_papers("first")
    calls_after_first = route.call_count
    with pytest.raises(s2.S2RateLimitError):
        await s2.search_papers("second")
    assert route.call_count == calls_after_first  # no request, no backoff while cooling down


@respx.mock
async def test_requests_resume_after_cooldown(sleeps, monkeypatch):
    respx.get(s2.S2_SEARCH_ENDPOINT).mock(return_value=httpx.Response(200, json=OK_BODY))
    monkeypatch.setattr(s2, "_cooldown_until", s2.time.monotonic() - 1)
    assert [p["paperId"] for p in await s2.search_papers("x")] == ["p1"]


@respx.mock
async def test_non_429_error_is_not_retried(sleeps):
    route = respx.get(s2.S2_SEARCH_ENDPOINT).mock(return_value=httpx.Response(403))
    assert await s2.search_papers("leucine adiponectin") == []
    assert route.call_count == 1
