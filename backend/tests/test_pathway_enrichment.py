"""Tests for pathway_enrichment Phase B degradation guard + HTTP migration (issue #44)."""

import types
from unittest.mock import AsyncMock

import pytest

from kestrel_backend.graph.nodes import pathway_enrichment as pe
from kestrel_backend.graph.state import (
    BiologicalTheme,
    ModelUsageRecord,
    SharedNeighbor,
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


@pytest.fixture
def patched(monkeypatch):
    """Patch the always-present collaborators; individual tests patch query_with_usage/parse."""
    monkeypatch.setattr(pe, "HAS_SDK", True)
    monkeypatch.setattr(
        pe, "find_two_hop_shared_neighbors",
        AsyncMock(return_value=({"NCBIGene:9": 2}, [])),
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

class TestRunDegradation:
    async def test_degraded_drops_sdk_keeps_two_hop(self, patched):
        patched.setattr(
            pe, "query_with_usage",
            AsyncMock(return_value=(
                "The Kestrel MCP tools are not available in my current tool set.",
                _record(0),  # zero mcp tool calls
            )),
        )
        # parse would yield SDK neighbors/themes, but degradation must discard them
        patched.setattr(
            pe, "parse_enrichment_result",
            lambda text: (
                [SharedNeighbor(curie="NCBIGene:9", name="G", category="biolink:Gene",
                                degree=10, connected_inputs=["CHEBI:1", "CHEBI:2"])],
                [_theme()],
                [],
            ),
        )
        out = await pe.run(_state())

        assert out["pathway_enrichment_degraded"] is True
        assert out["shared_neighbors"] == []
        assert out["biological_themes"] == []
        # only the HTTP-derived two-hop finding survives
        assert [f.source for f in out["direct_findings"]] == ["pathway_enrichment_two_hop"]

    async def test_healthy_keeps_sdk_findings(self, patched):
        patched.setattr(
            pe, "query_with_usage",
            AsyncMock(return_value=('{"ok": true}', _record(2))),  # tools were used
        )
        patched.setattr(pe, "parse_enrichment_result", lambda text: ([], [_theme()], []))
        out = await pe.run(_state())

        assert out["pathway_enrichment_degraded"] is False
        assert len(out["biological_themes"]) == 1
        sources = {f.source for f in out["direct_findings"]}
        assert "pathway_enrichment" in sources           # SDK theme finding kept
        assert "pathway_enrichment_two_hop" in sources    # two-hop also present

    async def test_drop_disabled_keeps_output_but_flags_degraded(self, patched):
        patched.setattr(pe._config, "drop_findings_on_degraded", False)
        patched.setattr(
            pe, "query_with_usage",
            AsyncMock(return_value=("not available in my current tool set", _record(0))),
        )
        patched.setattr(pe, "parse_enrichment_result", lambda text: ([], [_theme()], []))
        out = await pe.run(_state())

        # output retained ...
        assert len(out["biological_themes"]) == 1
        # ... but the run is still flagged degraded for disclosure
        assert out["pathway_enrichment_degraded"] is True

    async def test_exception_preserves_two_hop_findings(self, patched):
        # A Phase B exception must not lose the staged Phase A two-hop findings (R5).
        patched.setattr(pe, "query_with_usage", AsyncMock(side_effect=ValueError("boom")))
        out = await pe.run(_state())

        assert out["shared_neighbors"] == []
        assert [f.source for f in out["direct_findings"]] == ["pathway_enrichment_two_hop"]
