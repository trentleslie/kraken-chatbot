"""Unit 7: synthesis emits a run-level marker when it silently falls back.

The motivating degradation (context overflow) returns HTTP-success with empty
text, so without this marker the run reads as status=complete, errors=0 and the
failure is invisible to the performance report.
"""

import asyncio

import pytest

from kestrel_backend.graph.nodes import synthesis
from kestrel_backend.graph.state import Finding


def _state():
    # SynthesisInput requires at least one direct/cold_start finding.
    return {
        "raw_query": "what connects X and Y?",
        "resolved_entities": [],
        "direct_findings": [
            Finding(entity="CHEBI:17234", claim="glucose assoc", tier=1, source="direct_kg")
        ],
        "cold_start_findings": [],
        "hypotheses": [],
        "bridges": [],
    }


def _run(state):
    return asyncio.run(synthesis.run(state))


def test_no_marker_on_successful_synthesis(monkeypatch):
    async def ok(*_a, **_k):
        return "A full synthesis report.", None

    monkeypatch.setattr(synthesis, "HAS_SDK", True)
    monkeypatch.setattr(synthesis, "query_with_usage", ok)
    result = _run(_state())
    assert result["synthesis_report"].startswith("A full synthesis report.")
    assert not result.get("errors")  # no degradation -> no marker


def test_marker_on_empty_llm_output(monkeypatch):
    # The overflow case: HTTP-success but empty text.
    async def empty(*_a, **_k):
        return "   ", None

    monkeypatch.setattr(synthesis, "HAS_SDK", True)
    monkeypatch.setattr(synthesis, "query_with_usage", empty)
    result = _run(_state())
    assert "errors" in result
    assert any("empty output" in e for e in result["errors"])
    # report still produced via the deterministic fallback
    assert result["synthesis_report"]


def test_marker_on_sdk_exception(monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(synthesis, "HAS_SDK", True)
    monkeypatch.setattr(synthesis, "query_with_usage", boom)
    result = _run(_state())
    assert "errors" in result
    assert any("RuntimeError" in e and "LLM call failed" in e for e in result["errors"])
    assert result["synthesis_report"]


def test_no_marker_when_sdk_unavailable(monkeypatch):
    # SDK-unavailable is an environment condition, not a degradation -> no marker.
    monkeypatch.setattr(synthesis, "HAS_SDK", False)
    result = _run(_state())
    assert not result.get("errors")
    assert result["synthesis_report"]
