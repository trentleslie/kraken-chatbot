"""Tests for the hypothesis_extraction node's run() wrapper and its R13 failure boundary.

The relocated functions validate_bridge_hypotheses / extract_hypotheses are exercised verbatim
by their original suites (test_multi_hop_integration.TestValidateBridgeHypotheses and
test_langgraph_prototype.TestExtractHypotheses, retargeted to this module). This file covers what
those don't: the new node-level run() that wires them together, and the failure boundary that keeps
an upstream crash from aborting the run before synthesis (R13).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.kestrel_backend.graph.nodes import hypothesis_extraction
from src.kestrel_backend.graph.nodes.hypothesis_extraction import run
from src.kestrel_backend.graph.state import Bridge, Finding
from src.kestrel_backend.graph.state_contracts import (
    HypothesisExtractionOutput,
    StateValidationError,
)


def _path_found_response() -> dict:
    """Real Kestrel multi_hop_query envelope with a reachable path (drives a Tier-3 upgrade)."""
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "results": [{"end_node_id": "HGNC:6081",
                             "paths": [["CHEBI:17234", "HGNC:6081"]]}],
                "nodes": {"CHEBI:17234": {"name": "glucose"},
                          "HGNC:6081": {"name": "INS"}},
            }),
        }],
        "isError": False,
    }


def _no_path_response() -> dict:
    """Real empty Kestrel envelope — no path, so a Tier-3 bridge must stay Tier 3."""
    return {
        "content": [{"type": "text", "text": json.dumps({"results": []})}],
        "isError": False,
    }


def _tier3_bridge() -> Bridge:
    return Bridge(
        path_description="glucose -> INS",
        entities=["CHEBI:17234", "HGNC:6081"],
        entity_names=["glucose", "INS"],
        predicates=["biolink:affects"],
        tier=3,
        novelty="inferred",
        significance="Speculative glucose-insulin bridge",
    )


def _gate_finding() -> Finding:
    """A direct finding only to satisfy HypothesisExtractionInput's OR-gate; extract_hypotheses
    does not read direct_findings, so it contributes no hypotheses (isolates bridge-derived ones)."""
    return Finding(entity="CHEBI:17234", claim="glucose is associated with X", tier=1,
                   source="direct_kg", confidence="high")


class TestHypothesisExtractionRun:
    @pytest.mark.asyncio
    async def test_tier3_bridge_upgraded_yields_tier2_hypothesis(self):
        """Happy path: a KG-validated Tier-3 bridge upgrades to Tier 2, and the hypothesis built
        from the validated list carries tier=2 (proves hypotheses use the validated bridges)."""
        state = {"direct_findings": [_gate_finding()], "bridges": [_tier3_bridge()]}
        with patch(
            "src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query",
            new_callable=AsyncMock,
        ) as mock_mhq:
            mock_mhq.return_value = _path_found_response()
            result = await run(state)

        assert len(result["bridges"]) == 1
        assert result["bridges"][0].tier == 2
        assert "[KG-validated]" in result["bridges"][0].significance
        bridge_hyps = [h for h in result["hypotheses"] if h.title.startswith("Bridge:")]
        assert len(bridge_hyps) == 1
        assert bridge_hyps[0].tier == 2

    @pytest.mark.asyncio
    async def test_no_path_keeps_tier3(self):
        """A no-path multi_hop result leaves the bridge Tier 3 — guards the old always-upgrade bug."""
        state = {"direct_findings": [_gate_finding()], "bridges": [_tier3_bridge()]}
        with patch(
            "src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query",
            new_callable=AsyncMock,
        ) as mock_mhq:
            mock_mhq.return_value = _no_path_response()
            result = await run(state)

        assert result["bridges"][0].tier == 3
        bridge_hyps = [h for h in result["hypotheses"] if h.title.startswith("Bridge:")]
        assert bridge_hyps[0].tier == 3

    @pytest.mark.asyncio
    async def test_cold_start_only_empty_bridges(self):
        """No bridges, only a Tier-3 cold-start finding → hypotheses from cold-start, bridges []."""
        state = {
            "cold_start_findings": [
                Finding(entity="TEST:1", claim="Speculative role of compound", tier=3,
                        source="cold_start", confidence="low", logic_chain="analogue inference"),
            ],
            "bridges": [],
        }
        result = await run(state)
        assert result["bridges"] == []
        assert len(result["hypotheses"]) == 1
        assert result["hypotheses"][0].title.startswith("Inferred role of")

    @pytest.mark.asyncio
    async def test_empty_hypotheses_still_valid_output(self):
        """A well-characterized-only run (direct findings, no Tier-3 findings, no bridges) passes the
        input gate yet yields hypotheses: [] — a valid output, not a contract failure."""
        state = {"direct_findings": [_gate_finding()], "bridges": []}
        result = await run(state)  # @validate_state would raise if the output were invalid
        assert result["hypotheses"] == []
        assert result["bridges"] == []
        # Sanity: the returned payload genuinely satisfies the output contract.
        HypothesisExtractionOutput.model_validate(result)

    @pytest.mark.asyncio
    async def test_kg_call_failure_keeps_bridge_and_returns(self):
        """A raising multi_hop_query is caught per-bridge (Tier 3 kept); the node still returns."""
        state = {"direct_findings": [_gate_finding()], "bridges": [_tier3_bridge()]}
        with patch(
            "src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query",
            new_callable=AsyncMock,
        ) as mock_mhq:
            mock_mhq.side_effect = RuntimeError("Kestrel down")
            result = await run(state)
        assert result["bridges"][0].tier == 3

    @pytest.mark.asyncio
    async def test_r13_degrade_on_unguarded_exception(self, monkeypatch):
        """R13: if extraction raises an unguarded exception, run() degrades to a contract-valid
        payload ({bridges: <upstream>, hypotheses: []}) instead of propagating — so main.py never
        converts it to PIPELINE_ERROR and the run still reaches synthesis with a report."""
        original = [_tier3_bridge()]
        state = {"direct_findings": [_gate_finding()], "bridges": original}

        def boom(_state):
            raise RuntimeError("malformed finding")

        # Force the (otherwise unguarded) extract step to raise.
        monkeypatch.setattr(hypothesis_extraction, "extract_hypotheses", boom)
        with patch(
            "src.kestrel_backend.graph.nodes.hypothesis_extraction.multi_hop_query",
            new_callable=AsyncMock,
        ) as mock_mhq:
            mock_mhq.return_value = _no_path_response()
            result = await run(state)  # must NOT raise

        assert result["hypotheses"] == []
        assert result["bridges"] == original
        # The degrade payload must itself pass output validation (bridges is required).
        HypothesisExtractionOutput.model_validate(result)

    @pytest.mark.asyncio
    async def test_input_gate_rejects_empty_pipeline(self):
        """The OR-gate rejects a genuinely empty pipeline (no direct and no cold-start findings)."""
        with pytest.raises(StateValidationError, match="input"):
            await run({"direct_findings": [], "cold_start_findings": [], "bridges": []})

    @pytest.mark.asyncio
    async def test_validate_timeout_degrades_to_failure_payload(self, monkeypatch):
        """R13 latency ceiling: a slow validate loop hits asyncio.timeout (TimeoutError, an Exception)
        and degrades to the failure-boundary payload {bridges: upstream, hypotheses: []} — the run is
        bounded, not stalled, and still reaches synthesis."""
        import asyncio
        from src.kestrel_backend.graph.pipeline_config import get_pipeline_config

        monkeypatch.setattr(
            get_pipeline_config().hypothesis_extraction, "validate_timeout_seconds", 0.05
        )

        async def slow_validate(bridges):
            await asyncio.sleep(1.0)  # exceeds the 0.05s ceiling
            return bridges

        monkeypatch.setattr(hypothesis_extraction, "validate_bridge_hypotheses", slow_validate)

        original = [_tier3_bridge()]
        result = await run({"direct_findings": [_gate_finding()], "bridges": original})
        assert result["hypotheses"] == []
        assert result["bridges"] == original  # upstream passthrough
        HypothesisExtractionOutput.model_validate(result)
