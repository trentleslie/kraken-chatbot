"""Tests for pathway_enrichment Phase B degradation guard + HTTP migration (issue #44)."""

import types
from unittest.mock import AsyncMock

import pytest

from kestrel_backend.graph.nodes import pathway_enrichment as pe
from kestrel_backend.graph.state import (
    BiologicalTheme,
    ModelUsageRecord,
)


# --- Fixtures / factories ---

def _record(mcp_tool_calls: int, available_tools=None) -> ModelUsageRecord:
    return ModelUsageRecord(
        model_name="m",
        node_name="pathway_enrichment",
        mcp_tool_calls=mcp_tool_calls,
        available_tools=available_tools,
    )


def _entity(curie: str, name: str = "N"):
    return types.SimpleNamespace(curie=curie, method="exact", resolved_name=name, raw_name=name)


def _novelty(curie: str, edge_count: int):
    return types.SimpleNamespace(curie=curie, edge_count=edge_count)


def _state():
    return {
        "resolved_entities": [_entity("CHEBI:1", "fructose"), _entity("CHEBI:2", "glucose")],
        "novelty_scores": [_novelty("CHEBI:1", 50), _novelty("CHEBI:2", 60)],
    }


def _theme():
    return BiologicalTheme(
        category="biolink:BiologicalProcess",
        members=["NCBIGene:9"],
        member_names=["GENE9"],
        input_coverage=2,
    )


def _populated_prefetch():
    edge = {"object": {"id": "NCBIGene:9"}, "predicate": "biolink:related_to"}
    return (
        {
            "CHEBI:1": {"summary": "related_to: 1", "edges": [edge], "errored": False},
            "CHEBI:2": {"summary": "related_to: 1", "edges": [edge], "errored": False},
        },
        False,  # no_data
    )


@pytest.fixture
def patched(monkeypatch):
    """Patch the always-present collaborators; individual tests patch query_with_usage/parse."""
    monkeypatch.setattr(pe, "HAS_SDK", True)
    monkeypatch.setattr(
        pe, "find_two_hop_shared_neighbors",
        AsyncMock(return_value=({"NCBIGene:9": 2}, [])),
    )
    # Stage 2: Phase B prefetches one-hop data via HTTP; default to a populated result.
    monkeypatch.setattr(
        pe, "prefetch_one_hop_neighbors",
        AsyncMock(return_value=_populated_prefetch()),
    )
    return monkeypatch


# --- Pure helper tests ---

class TestBuildTwoHopFindings:
    def test_empty(self):
        assert pe._build_two_hop_findings({}) == []

    def test_nonempty_is_tagged_two_hop(self):
        out = pe._build_two_hop_findings({"NCBIGene:9": 2})
        assert len(out) == 1
        assert out[0].source == "pathway_enrichment_two_hop"


class TestDegradedResultBuilder:
    def test_drops_sdk_keeps_two_hop_and_flags(self):
        two_hop = pe._build_two_hop_findings({"NCBIGene:9": 2})
        rec = _record(0)
        out = pe._degraded_phase_b_result(two_hop, ["base err"], rec, "mcp_zero_mcp_tool_calls")
        assert out["shared_neighbors"] == []
        assert out["biological_themes"] == []
        assert out["direct_findings"] == two_hop
        assert out["pathway_enrichment_degraded"] is True
        assert any("degraded" in e for e in out["errors"])
        assert out["model_usages"] == [rec]


# --- run() integration tests ---

def _conns(mapping):
    """Build an async get_entity_connections stub returning per-curie results."""
    async def _f(curie):
        result = mapping[curie]
        if isinstance(result, Exception):
            raise result
        return result
    return _f


def _ok(n_edges: int):
    """A successful one_hop result (includes total_count, the non-errored marker)."""
    return {"edges": [{"predicate": "biolink:related_to"}] * n_edges,
            "summary": f"{n_edges} edges", "total_count": n_edges}


def _errored():
    """A failed one_hop result (no total_count key, like get_entity_connections failures)."""
    return {"edges": [], "summary": "Query failed"}


class TestPrefetchOneHopNeighbors:
    async def test_two_populated_not_no_data(self, monkeypatch):
        monkeypatch.setattr(pe, "get_entity_connections",
                            _conns({"CHEBI:1": _ok(3), "CHEBI:2": _ok(2)}))
        per_entity, no_data = await pe.prefetch_one_hop_neighbors(
            [_entity("CHEBI:1"), _entity("CHEBI:2")])
        assert no_data is False
        assert per_entity["CHEBI:1"]["errored"] is False
        assert per_entity["CHEBI:2"]["errored"] is False

    async def test_one_errored_triggers_no_data(self, monkeypatch):
        monkeypatch.setattr(pe, "get_entity_connections",
                            _conns({"CHEBI:1": _ok(3), "CHEBI:2": _errored()}))
        per_entity, no_data = await pe.prefetch_one_hop_neighbors(
            [_entity("CHEBI:1"), _entity("CHEBI:2")])
        assert no_data is True  # only 1 of 2 populated
        assert per_entity["CHEBI:1"]["errored"] is False
        assert per_entity["CHEBI:2"]["errored"] is True

    async def test_genuine_empty_counts_as_unpopulated(self, monkeypatch):
        # Non-errored but zero edges → not "real neighbor data" → no_data.
        monkeypatch.setattr(pe, "get_entity_connections",
                            _conns({"CHEBI:1": _ok(0), "CHEBI:2": _ok(0)}))
        per_entity, no_data = await pe.prefetch_one_hop_neighbors(
            [_entity("CHEBI:1"), _entity("CHEBI:2")])
        assert no_data is True
        assert per_entity["CHEBI:1"]["errored"] is False  # not an error, just empty

    async def test_exception_marks_errored(self, monkeypatch):
        monkeypatch.setattr(pe, "get_entity_connections",
                            _conns({"CHEBI:1": _ok(3), "CHEBI:2": RuntimeError("boom")}))
        per_entity, no_data = await pe.prefetch_one_hop_neighbors(
            [_entity("CHEBI:1"), _entity("CHEBI:2")])
        assert per_entity["CHEBI:2"]["errored"] is True
        assert no_data is True


class TestRunMigratedPhaseB:
    """Stage 2 (issue #44): Phase B prefetches via HTTP and reasons in-prompt (no MCP)."""

    async def test_no_data_degrades_and_keeps_two_hop(self, patched):
        # Prefetch found <2 populated entities → degrade without running inference.
        patched.setattr(pe, "prefetch_one_hop_neighbors", AsyncMock(return_value=({}, True)))
        sdk = AsyncMock()
        patched.setattr(pe, "query_with_usage", sdk)
        out = await pe.run(_state())

        assert out["pathway_enrichment_degraded"] is True
        assert out["shared_neighbors"] == []
        assert out["biological_themes"] == []
        assert [f.source for f in out["direct_findings"]] == ["pathway_enrichment_two_hop"]
        assert any("prefetch_no_data" in e for e in out["errors"])
        sdk.assert_not_awaited()  # inference is skipped on the no-data path

    async def test_healthy_inference_keeps_sdk_findings(self, patched):
        # Populated prefetch (default) + parseable inference output → findings kept.
        patched.setattr(
            pe, "query_with_usage",
            AsyncMock(return_value=('{"shared_neighbors": [], "themes": []}', _record(0))),
        )
        patched.setattr(pe, "parse_enrichment_result", lambda text: ([], [_theme()], []))
        out = await pe.run(_state())

        assert out["pathway_enrichment_degraded"] is False
        assert len(out["biological_themes"]) == 1
        sources = {f.source for f in out["direct_findings"]}
        assert "pathway_enrichment" in sources           # SDK theme finding kept
        assert "pathway_enrichment_two_hop" in sources    # two-hop also present

    async def test_inert_mcp_classifier_does_not_flag_migrated_path(self, patched):
        # Even if the inference text echoes the fallback phrase, allowed_tools=[] means
        # the MCP classifier is inert — the run must NOT be flagged degraded.
        patched.setattr(
            pe, "query_with_usage",
            AsyncMock(return_value=("those tools are not available in my current tool set",
                                    _record(0))),
        )
        patched.setattr(pe, "parse_enrichment_result", lambda text: ([], [_theme()], []))
        out = await pe.run(_state())

        assert out["pathway_enrichment_degraded"] is False

    async def test_exception_preserves_two_hop_findings(self, patched):
        # A Phase B exception must not lose the staged Phase A two-hop findings (R5).
        patched.setattr(pe, "query_with_usage", AsyncMock(side_effect=ValueError("boom")))
        out = await pe.run(_state())

        assert out["shared_neighbors"] == []
        assert [f.source for f in out["direct_findings"]] == ["pathway_enrichment_two_hop"]
