"""OpenAlex: send OPENALEX_API_KEY when set; a 429 (budget exhausted) is logged, not silent."""

import importlib.util
import logging
from pathlib import Path

import httpx
import respx

# Load the real module under a private name (other tests replace kestrel_backend modules in sys.modules).
_SRC = Path(__file__).resolve().parents[1] / "src" / "kestrel_backend" / "openalex.py"
_spec = importlib.util.spec_from_file_location("_openalex_under_test", _SRC)
oa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oa)

URL = f"{oa.OPENALEX_API}/works"


@respx.mock
async def test_api_key_sent_when_set(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "k123")
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"results": [{"id": "W1"}]}))
    assert await oa.search_works("insulin", limit=1) == [{"id": "W1"}]
    assert route.calls[0].request.url.params["api_key"] == "k123"


@respx.mock
async def test_no_api_key_param_when_unset(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"results": []}))
    await oa.search_works("insulin", limit=1)
    assert "api_key" not in route.calls[0].request.url.params


@respx.mock
async def test_budget_429_is_logged_as_rate_limit(monkeypatch, caplog):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    respx.get(URL).mock(return_value=httpx.Response(
        429, headers={"x-ratelimit-remaining-usd": "0", "retry-after": "7788"}))
    with caplog.at_level(logging.WARNING):
        assert await oa.search_works("insulin", limit=1) == []
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "rate limited (429)" in msg and "7788" in msg and "NOT SET" in msg
