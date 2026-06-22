"""Unit 5: reporting terminal node + graph wiring (fail-safe)."""

import asyncio

from kestrel_backend import run_reports_io
from kestrel_backend.graph.builder import build_discovery_graph
from kestrel_backend.graph.nodes import reporting


def _state():
    return {
        "raw_query": "what connects X and Y?",
        "biomapper_env": "production",
        "node_timings": {"intake": 1.0, "synthesis": 2.0},
        "model_usages": [],
        "errors": [],
    }


def test_reporting_writes_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(run_reports_io, "DEFAULT_DIR", tmp_path)
    result = asyncio.run(reporting.run(_state()))
    assert "report_path" in result
    assert list(tmp_path.glob("*.json")) and list(tmp_path.glob("*.md"))


def test_reporting_omits_raw_query_from_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(run_reports_io, "DEFAULT_DIR", tmp_path)
    state = _state()
    state["raw_query"] = "SECRET PATIENT QUERY"
    asyncio.run(reporting.run(state))
    md = next(tmp_path.glob("*.md")).read_text()
    js = next(tmp_path.glob("*.json")).read_text()
    assert "SECRET PATIENT QUERY" not in md  # markdown is Outline-bound
    assert "SECRET PATIENT QUERY" in js  # full query retained in JSON


def test_failsafe_on_write_error(tmp_path, monkeypatch):
    # R4: a write failure is swallowed; run() returns {} and does not raise.
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(reporting.run_reports_io, "write_report", boom)
    result = asyncio.run(reporting.run(_state()))
    assert result == {}


def test_failsafe_on_build_error(monkeypatch):
    # R4: a build failure (malformed state) is swallowed; no partial artifact, no raise.
    def boom(*_a, **_k):
        raise ValueError("malformed state")

    monkeypatch.setattr(reporting, "build_report", boom)
    result = asyncio.run(reporting.run(_state()))
    assert result == {}


def test_graph_wires_reporting_after_synthesis():
    g = build_discovery_graph().get_graph()
    node_ids = set(g.nodes)
    assert "reporting" in node_ids
    edges = {(e.source, e.target) for e in g.edges}
    assert ("synthesis", "reporting") in edges
    # reporting flows to END (LangGraph's terminal node id is "__end__")
    assert any(src == "reporting" and tgt in ("__end__", "END") for src, tgt in edges)
    # synthesis no longer goes directly to END
    assert not any(
        src == "synthesis" and tgt in ("__end__", "END") for src, tgt in edges
    )


def test_reporting_not_in_node_status_messages():
    # Intentionally invisible to the WS stream in v1.
    from kestrel_backend.protocol import NODE_STATUS_MESSAGES

    assert "reporting" not in NODE_STATUS_MESSAGES


def test_pipeline_nodes_matches_graph():
    """Drift guard: every graph node (minus __start__/__end__/reporting) must be in
    PIPELINE_NODES, else build_report silently marks a real node as an 'extra'."""
    from kestrel_backend.graph.performance_report import PIPELINE_NODES

    g = build_discovery_graph().get_graph()
    graph_nodes = {n for n in g.nodes if n not in ("__start__", "__end__", "reporting")}
    assert graph_nodes == set(PIPELINE_NODES)
