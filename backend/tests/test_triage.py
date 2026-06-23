"""Triage reliability at module scale (plan 2026-06-23-001).

Core invariant: a genuine 0-edge count → cold_start (biology); a measurement *failure*
(isError / empty / unparseable / exhausted retries) → moderate (direct-KG path) + a visible
marker, never silent cold_start. Edge-count concurrency is bounded by config.
"""

import asyncio
import json

import pytest

from kestrel_backend.graph.nodes import triage
from kestrel_backend.graph.pipeline_config import PipelineConfig, TriageConfig
from kestrel_backend.graph.state import EntityResolution


def _entity(curie, name=None, method="biomapper"):
    return EntityResolution(
        raw_name=name or curie, curie=curie, resolved_name=name or curie,
        category="biolink:Gene", confidence=0.9, method=method,
    )


def _use_cfg(monkeypatch, **triage_kwargs):
    cfg = PipelineConfig(triage=TriageConfig(**triage_kwargs))
    monkeypatch.setattr(triage, "get_pipeline_config", lambda: cfg)
    monkeypatch.setattr(triage, "_RETRY_BACKOFF_S", 0.0)  # no real sleeps in tests (fail-fast if renamed)
    return cfg


def _ok(n):
    return {"isError": False, "content": [{"text": json.dumps({"results_count": n})}]}


def _err():
    return {"isError": True, "content": []}


def _empty():
    return {"isError": False, "content": []}


class FakeKestrel:
    """Async stand-in for call_kestrel_tool. `script` maps curie -> list of response-builders
    consumed per attempt; `default` is used when a curie has no script. Tracks max concurrency."""

    def __init__(self, script=None, default=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default or (lambda: _ok(500))
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(self, tool, args):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)  # yield so concurrent calls actually overlap
            curie = args["start_node_ids"]
            seq = self.script.get(curie)
            builder = seq.pop(0) if seq else self.default
            return builder()
        finally:
            self.concurrent -= 1


async def _run(monkeypatch, entities, fake):
    monkeypatch.setattr(triage, "call_kestrel_tool", fake)
    return await triage.run({"resolved_entities": entities})


# --- classification happy paths -------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("count,bucket", [(500, "well_characterized_curies"), (50, "moderate_curies"),
                                          (5, "sparse_curies"), (0, "cold_start_curies")])
async def test_classification_by_count(monkeypatch, count, bucket):
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    fake = FakeKestrel(default=lambda: _ok(count))
    out = await _run(monkeypatch, [_entity("NCBIGene:1")], fake)
    assert out[bucket] == ["NCBIGene:1"]


@pytest.mark.asyncio
async def test_genuine_zero_is_cold_start(monkeypatch):
    """A successful query returning results_count=0 is real biology → cold_start, no marker."""
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    out = await _run(monkeypatch, [_entity("CHEBI:1")], FakeKestrel(default=lambda: _ok(0)))
    assert out["cold_start_curies"] == ["CHEBI:1"]
    assert not [e for e in out["errors"] if "edge-count" in str(e)]


# --- the core fix: measurement failure must NOT masquerade as cold_start ---------------


@pytest.mark.asyncio
async def test_measurement_failure_routes_to_moderate_with_marker(monkeypatch):
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    out = await _run(monkeypatch, [_entity("NCBIGene:3569", "IL6")], FakeKestrel(default=_err))
    assert out["moderate_curies"] == ["NCBIGene:3569"]      # routed to direct-KG path
    assert "NCBIGene:3569" not in out["cold_start_curies"]  # NOT silently cold-started
    score = next(s for s in out["novelty_scores"] if s.curie == "NCBIGene:3569")
    assert score.classification == "moderate"
    marker = [e for e in out["errors"] if "edge-count" in str(e) and "NCBIGene:3569" in str(e)]
    assert marker, "a visible degraded marker must be emitted"


@pytest.mark.asyncio
async def test_empty_content_is_measurement_failure(monkeypatch):
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    out = await _run(monkeypatch, [_entity("NCBIGene:7422", "VEGFA")], FakeKestrel(default=_empty))
    assert "NCBIGene:7422" in out["moderate_curies"]
    assert "NCBIGene:7422" not in out["cold_start_curies"]


@pytest.mark.asyncio
async def test_retry_recovers_transient_failure(monkeypatch):
    """isError on attempt 1, success on attempt 2 → classified by the real count, no marker."""
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    fake = FakeKestrel(script={"NCBIGene:3569": [_err, lambda: _ok(9911)]})
    out = await _run(monkeypatch, [_entity("NCBIGene:3569", "IL6")], fake)
    assert out["well_characterized_curies"] == ["NCBIGene:3569"]
    assert not [e for e in out["errors"] if "edge-count" in str(e)]


@pytest.mark.asyncio
async def test_failed_resolution_still_cold_start(monkeypatch):
    """An entity that failed resolution (no usable curie) is genuinely cold_start — unchanged."""
    _use_cfg(monkeypatch, kestrel_concurrency=4)
    out = await _run(monkeypatch, [_entity("x", "mystery", method="failed")], FakeKestrel())
    assert "mystery" in out["cold_start_curies"]


# --- concurrency bound (R1) -----------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_count_concurrency_is_bounded(monkeypatch):
    _use_cfg(monkeypatch, kestrel_concurrency=3)
    fake = FakeKestrel(default=lambda: _ok(500))
    entities = [_entity(f"NCBIGene:{i}") for i in range(20)]
    out = await _run(monkeypatch, entities, fake)
    assert fake.max_concurrent <= 3, f"expected <=3 concurrent, saw {fake.max_concurrent}"
    assert len(out["well_characterized_curies"]) == 20  # all still counted
