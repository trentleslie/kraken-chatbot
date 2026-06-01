"""Tests for the reasoning-depth demo slice (Unit 1: multi-hop in direct_kg).

Plan: docs/plans/2026-05-30-001-feat-discovery-depth-demo-slice-plan.md
Focus: the hub-suppression guard (D3, the load-bearing correctness invariant) plus the
multi-hop parser, the in-query degree constraint, and the default-off flag.
"""
import pytest

from kestrel_backend.graph.nodes import direct_kg
from kestrel_backend.graph.state import Finding


# Representative live-API result shapes (captured 2026-06-01, MONDO:0005148):
_RESULT_OK = {
    "start_node_ids": ["MONDO:0005148"],
    "end_node_id": "MONDO:0002909",
    "score": 0.777,
    "degree": 3582,  # < hub_threshold (5000)
    "path_count": 166,
    "paths": [
        ["MONDO:0005148", "MONDO:0002909"],
        ["MONDO:0005148", "CHEBI:28077", "MONDO:0002909"],
    ],
}
_RESULT_HUB = {**_RESULT_OK, "end_node_id": "NCBIGene:HUB", "degree": 9999}  # > hub_threshold


def test_multi_hop_constraints_guard_on_degree():
    """The in-query guard must constrain node degree below the hub threshold."""
    cons = direct_kg._multi_hop_constraints(hub_threshold=5000)
    assert isinstance(cons, list) and cons, "expected a non-empty constraints list"
    degree_c = next((c for c in cons if c.get("field") == "degree"), None)
    assert degree_c is not None, "a degree constraint must be present"
    assert degree_c["operator"] in {"lt", "lte"}
    assert degree_c["value"] == 5000


def test_parse_multi_hop_findings_builds_finding():
    """A low-degree result becomes a multi-hop Finding with a mechanistic chain."""
    findings = direct_kg._parse_multi_hop_findings(
        "MONDO:0005148", "type 2 diabetes", {"results": [_RESULT_OK]}, hub_threshold=5000
    )
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.entity == "MONDO:0005148"
    assert f.source == "direct_kg_multi_hop"
    assert f.logic_chain and "CHEBI:28077" in f.logic_chain  # the multi-hop path, not the direct edge
    assert "MONDO:0002909" in f.logic_chain


def test_parse_multi_hop_findings_suppresses_hub():
    """LOAD-BEARING (D3): a result whose node degree exceeds the hub threshold is dropped."""
    findings = direct_kg._parse_multi_hop_findings(
        "MONDO:0005148", "type 2 diabetes", {"results": [_RESULT_HUB]}, hub_threshold=5000
    )
    assert findings == [], "hub-degree results must be suppressed before becoming findings"


def test_parse_multi_hop_findings_mixed_keeps_only_non_hub():
    findings = direct_kg._parse_multi_hop_findings(
        "MONDO:0005148", "type 2 diabetes",
        {"results": [_RESULT_OK, _RESULT_HUB]}, hub_threshold=5000,
    )
    assert len(findings) == 1
    assert findings[0].logic_chain and "NCBIGene:HUB" not in findings[0].logic_chain


async def test_analyze_multi_hop_disabled_is_inert(monkeypatch):
    """Default-off: when the flag is False, no API call is made and no findings are produced."""
    monkeypatch.setattr(direct_kg._config, "multi_hop_enabled", False)

    async def _boom(*a, **k):  # must never be called
        raise AssertionError("multi_hop_query called while disabled")

    monkeypatch.setattr(direct_kg, "multi_hop_query", _boom)
    assert await direct_kg.analyze_multi_hop("MONDO:0005148", "type 2 diabetes") == []


async def test_analyze_multi_hop_enabled_passes_degree_constraint(monkeypatch):
    """When enabled, the degree guard is sent in-query and findings are parsed back."""
    import json

    monkeypatch.setattr(direct_kg._config, "multi_hop_enabled", True)
    captured = {}

    async def _fake_mhq(**kwargs):
        captured.update(kwargs)
        return {"content": [{"type": "text", "text": json.dumps({"results": [_RESULT_OK]})}],
                "isError": False}

    monkeypatch.setattr(direct_kg, "multi_hop_query", _fake_mhq)
    findings = await direct_kg.analyze_multi_hop("MONDO:0005148", "type 2 diabetes")

    assert captured["start_node_ids"] == ["MONDO:0005148"]
    assert any(c.get("field") == "degree" for c in captured.get("constraints", [])), \
        "the degree hub-guard constraint must be passed in-query"
    assert len(findings) == 1 and findings[0].source == "direct_kg_multi_hop"


async def test_analyze_multi_hop_handles_api_error(monkeypatch):
    """A failed/empty API response degrades to no findings (never raises)."""
    monkeypatch.setattr(direct_kg._config, "multi_hop_enabled", True)

    async def _err(**kwargs):
        return {"content": [{"type": "text", "text": "Error: boom"}], "isError": True}

    monkeypatch.setattr(direct_kg, "multi_hop_query", _err)
    assert await direct_kg.analyze_multi_hop("MONDO:0005148", "type 2 diabetes") == []
