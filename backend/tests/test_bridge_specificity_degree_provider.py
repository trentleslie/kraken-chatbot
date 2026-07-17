"""Axis C, Unit 2 — hybrid, bounded, per-run cached degree provider.

Resolves a CURIE's KG degree: opportunistic inline read first, else a semaphore-bounded,
per-run dedup-cached ``one_hop_query`` count fetch (mode="preview", ``results_count`` — as
triage does). Never raises; returns ``int | None``. Mirrors ``bridge_grounding.cached_leg_fetcher``.

Run with: uv run python -m pytest tests/test_bridge_specificity_degree_provider.py -v
"""

import asyncio

import pytest

from kestrel_backend.graph.nodes import bridge_specificity as bs


def _preview_response(count: int) -> dict:
    return {"isError": False, "content": [{"text": f'{{"results_count": {count}}}'}]}


def _stub_kestrel(monkeypatch, handler, calls=None):
    async def fake(tool_name, params):
        if calls is not None:
            calls.append((tool_name, params))
        return await handler(tool_name, params) if asyncio.iscoroutinefunction(handler) else handler(tool_name, params)
    monkeypatch.setattr(bs, "call_kestrel_tool", fake)


# --- happy path ----------------------------------------------------------------------------

async def test_cold_curie_one_preview_fetch(monkeypatch):
    calls = []
    _stub_kestrel(monkeypatch, lambda t, p: _preview_response(42), calls)
    get = bs.make_degree_provider(concurrency=4)
    assert await get("HGNC:1") == 42
    assert len(calls) == 1
    tool, params = calls[0]
    assert tool == "one_hop_query"
    assert params["start_node_ids"] == "HGNC:1"
    assert params["mode"] == "preview"


# --- opportunistic inline read -------------------------------------------------------------

async def test_inline_degree_used_without_fetch(monkeypatch):
    calls = []
    _stub_kestrel(monkeypatch, lambda t, p: _preview_response(999), calls)
    # Only meaningful if a live envelope is confirmed to carry per-node degree; the provider
    # treats it as an opportunistic bonus.
    get = bs.make_degree_provider(inline_nodes={"HGNC:1": {"name": "X", "degree": 7}})
    assert await get("HGNC:1") == 7
    assert calls == []  # no fetch when inline degree present


async def test_inline_without_degree_falls_back_to_fetch(monkeypatch):
    calls = []
    _stub_kestrel(monkeypatch, lambda t, p: _preview_response(55), calls)
    # Documented nodes entry is {name, categories} -> NO degree; must fall back to the fetch.
    get = bs.make_degree_provider(inline_nodes={"HGNC:1": {"name": "X", "categories": ["biolink:Gene"]}})
    assert await get("HGNC:1") == 55
    assert len(calls) == 1


# --- per-run cache (dedup) -----------------------------------------------------------------

async def test_same_curie_fetched_once(monkeypatch):
    calls = []
    _stub_kestrel(monkeypatch, lambda t, p: _preview_response(10), calls)
    get = bs.make_degree_provider()
    assert await get("HGNC:1") == 10
    assert await get("HGNC:1") == 10
    assert len(calls) == 1  # cache hit second time


async def test_none_result_is_cached(monkeypatch):
    calls = []
    _stub_kestrel(monkeypatch, lambda t, p: {"isError": True, "content": []}, calls)
    get = bs.make_degree_provider()
    assert await get("HGNC:1") is None
    assert await get("HGNC:1") is None
    assert len(calls) == 1  # a None result is cached too (no re-fetch)


# --- error paths (never raise) -------------------------------------------------------------

async def test_isError_returns_none(monkeypatch):
    _stub_kestrel(monkeypatch, lambda t, p: {"isError": True, "content": [{"text": "boom"}]})
    get = bs.make_degree_provider()
    assert await get("HGNC:1") is None


async def test_exception_returns_none(monkeypatch):
    def boom(t, p):
        raise RuntimeError("kestrel down")
    _stub_kestrel(monkeypatch, boom)
    get = bs.make_degree_provider()
    assert await get("HGNC:1") is None  # no exception escapes


async def test_unparseable_response_returns_none(monkeypatch):
    _stub_kestrel(monkeypatch, lambda t, p: {"isError": False, "content": [{"text": "not json"}]})
    get = bs.make_degree_provider()
    assert await get("HGNC:1") is None


async def test_missing_results_count_returns_none(monkeypatch):
    _stub_kestrel(monkeypatch, lambda t, p: {"isError": False, "content": [{"text": "{}"}]})
    get = bs.make_degree_provider()
    assert await get("HGNC:1") is None


# --- concurrency bound ---------------------------------------------------------------------

async def test_concurrency_never_exceeds_bound(monkeypatch):
    bound = 3
    state = {"in_flight": 0, "max": 0}

    async def handler(t, p):
        state["in_flight"] += 1
        state["max"] = max(state["max"], state["in_flight"])
        await asyncio.sleep(0.01)  # hold the slot so overlap is observable
        state["in_flight"] -= 1
        return _preview_response(1)

    _stub_kestrel(monkeypatch, handler)
    get = bs.make_degree_provider(concurrency=bound)
    curies = [f"HGNC:{i}" for i in range(12)]  # all distinct -> no cache dedup
    results = await asyncio.gather(*[get(c) for c in curies])
    assert all(r == 1 for r in results)
    assert state["max"] <= bound
