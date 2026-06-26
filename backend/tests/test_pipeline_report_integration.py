"""Unit 6: WS-path node_timings regression guard + integration.

Drives the real ``handle_pipeline_mode`` with a mocked ``stream_discovery`` that
yields per-node ``node_timings`` deltas (as the timed_node wrapper produces), and
asserts main.py's accumulator merges them into the full dict — the regression
this unit prevents (a dict reducer field is not in CONCAT_LIST_FIELDS, so without
an explicit dict-merge branch it collapses to last-write-wins).
"""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.kestrel_backend import main
from src.kestrel_backend.graph import runner


def _sent(ws):
    return [json.loads(c.args[0]) for c in ws.send_text.call_args_list]


def _multi_node_stream(query, conversation_history=None, config=None, biomapper_env=None):
    """Yield three nodes, each carrying its own node_timings delta (wrapper shape)."""

    async def _gen():
        yield {
            "type": "node_update",
            "node": "intake",
            "node_output": {"raw_entities": [], "node_timings": {"intake": 1.0}},
        }
        yield {
            "type": "node_update",
            "node": "triage",
            "node_output": {"node_timings": {"triage": 2.0}},
        }
        yield {
            "type": "node_update",
            "node": "synthesis",
            "node_output": {
                "synthesis_report": "FINAL REPORT",
                "node_timings": {"synthesis": 3.0},
            },
        }

    return _gen()


@pytest.fixture
def captured_turn(monkeypatch):
    """Disable langfuse, capture add_turn metrics, wire a conversation id."""
    monkeypatch.setattr(main, "_get_pipeline_langfuse", lambda: None)
    monkeypatch.setattr(runner, "stream_discovery", _multi_node_stream)

    captured = {}

    async def fake_add_turn(**kwargs):
        captured.update(kwargs)
        return uuid4()

    monkeypatch.setattr(main, "add_turn", fake_add_turn)

    conn = "conn-report-int"
    main.conversation_ids[conn] = uuid4()
    main.turn_counters[conn] = 0
    yield conn, captured
    main.conversation_ids.pop(conn, None)


@pytest.mark.asyncio
async def test_node_timings_merged_across_nodes(captured_turn):
    conn, captured = captured_turn
    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "metabolite X disease Y", conn)

    metrics = captured.get("metrics", {})
    timings = metrics.get("node_timings", {})
    # Regression guard: ALL three nodes present, not collapsed to the last one.
    assert timings == {"intake": 1.0, "triage": 2.0, "synthesis": 3.0}


@pytest.mark.asyncio
async def test_authoritative_duration_in_detail_messages(captured_turn):
    conn, _ = captured_turn
    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "q", conn)

    details = {m["node"]: m for m in _sent(ws) if m.get("type") == "pipeline_node_detail"}
    # duration_ms is sourced from the wrapper's authoritative per-node timing.
    assert details["synthesis"]["duration_ms"] == 3000
    assert details["intake"]["duration_ms"] == 1000


@pytest.mark.asyncio
async def test_ws_run_streams_normally(captured_turn):
    conn, _ = captured_turn
    ws = AsyncMock()
    await main.handle_pipeline_mode(ws, "q", conn)
    types = [m.get("type") for m in _sent(ws)]
    assert "pipeline_complete" in types
    assert "done" in types
    assert "error" not in types


@pytest.mark.asyncio
async def test_reporting_node_in_stream_does_not_break_ws(captured_turn):
    """A `reporting` node update in the stream (not in NODE_STATUS_MESSAGES) is skipped
    by the WS accumulator and does not disrupt completion — fail-safe behavior of the
    reporting node itself is covered in test_reporting_node.py."""
    conn, _ = captured_turn

    def stream_with_reporting(query, conversation_history=None, config=None, biomapper_env=None):
        async def _gen():
            yield {
                "type": "node_update",
                "node": "synthesis",
                "node_output": {"synthesis_report": "R", "node_timings": {"synthesis": 1.0}},
            }
            yield {
                "type": "node_update",
                "node": "reporting",
                "node_output": {"report_path": "/tmp/x.json", "node_timings": {"reporting": 0.1}},
            }

        return _gen()

    import pytest as _pytest  # local to avoid confusion

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(runner, "stream_discovery", stream_with_reporting)
    try:
        ws = AsyncMock()
        await main.handle_pipeline_mode(ws, "q", conn)
        types = [m.get("type") for m in _sent(ws)]
        assert "pipeline_complete" in types and "done" in types
        # reporting is invisible to the WS stream (not in NODE_STATUS_MESSAGES)
        nodes = {m.get("node") for m in _sent(ws) if m.get("type") == "pipeline_progress"}
        assert "reporting" not in nodes
    finally:
        monkeypatch.undo()
