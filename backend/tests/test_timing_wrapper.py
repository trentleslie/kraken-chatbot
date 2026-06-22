"""Unit 1: per-node timing wrapper + node_timings merge reducer.

Covers the plan's Unit 1 test scenarios: dict/None/non-dict returns, the
defensive pass-through when the wrapper's own logic fails, the merge reducer
under parallel branches, and an end-to-end check that a two-branch graph writes
node_timings without raising LangGraph's InvalidUpdateError.
"""

import asyncio

import pytest

from kestrel_backend.graph import timing
from kestrel_backend.graph.timing import timed_node
from kestrel_backend.graph.state import merge_node_timings


# --- timed_node wrapper ------------------------------------------------------

def test_wraps_dict_return_adds_timing():
    async def node(state):
        return {"errors": ["x"]}

    result = asyncio.run(timed_node("intake", node)({}))
    assert result["errors"] == ["x"]
    assert "node_timings" in result
    assert set(result["node_timings"]) == {"intake"}
    assert result["node_timings"]["intake"] >= 0.0


def test_wraps_none_return_still_carries_timing():
    async def node(state):
        return None

    result = asyncio.run(timed_node("triage", node)({}))
    assert result == {"node_timings": {"triage": result["node_timings"]["triage"]}}
    assert result["node_timings"]["triage"] >= 0.0


def test_wraps_empty_dict_return():
    async def node(state):
        return {}

    result = asyncio.run(timed_node("integration", node)({}))
    assert list(result.keys()) == ["node_timings"]
    assert "integration" in result["node_timings"]


def test_wraps_nondict_return_without_raising(caplog):
    import logging

    async def node(state):
        return ["not", "a", "dict"]  # invalid state update, must not crash the wrapper

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(timed_node("weird", node)({}))
    assert result["node_timings"]["weird"] >= 0.0
    # A dropped non-dict state update must not be silent.
    assert any("non-dict" in rec.message for rec in caplog.records)


def test_shallow_copy_preserves_reducer_values_identity():
    # Reducer-bound list objects must pass through unchanged (not deep-copied / rebuilt).
    sentinel = [object()]

    async def node(state):
        return {"model_usages": sentinel}

    result = asyncio.run(timed_node("synthesis", node)({}))
    assert result["model_usages"] is sentinel


def test_duration_at_least_sleep():
    async def node(state):
        await asyncio.sleep(0.02)
        return {}

    result = asyncio.run(timed_node("slow", node)({}))
    assert result["node_timings"]["slow"] >= 0.02


def test_node_exception_propagates_unchanged():
    # A node raising is the node's own error and must surface (wrapper does not guard fn()).
    async def node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(timed_node("intake", node)({}))


def test_wrapper_defensive_when_timing_logic_raises(monkeypatch):
    # If the wrapper's own timing/merge logic raises, the node's result returns unchanged.
    original = {"errors": ["keep me"]}

    async def node(state):
        return original

    def boom(*_args, **_kwargs):
        raise RuntimeError("timing exploded")

    monkeypatch.setattr(timing, "_attach_timing", boom)
    result = asyncio.run(timed_node("intake", node)({}))
    assert result is original  # unchanged; pipeline unaffected


# --- merge_node_timings reducer ----------------------------------------------

def test_merge_disjoint_keys():
    assert merge_node_timings({"direct_kg": 1.0}, {"cold_start": 2.0}) == {
        "direct_kg": 1.0,
        "cold_start": 2.0,
    }


def test_merge_handles_none_operands():
    assert merge_node_timings(None, {"a": 1.0}) == {"a": 1.0}
    assert merge_node_timings({"a": 1.0}, None) == {"a": 1.0}
    assert merge_node_timings(None, None) == {}


def test_merge_key_collision_keeps_right():
    assert merge_node_timings({"a": 1.0}, {"a": 2.0}) == {"a": 2.0}


# --- integration: reducer prevents InvalidUpdateError on parallel writes ------

def test_two_parallel_branches_write_node_timings_without_error():
    """A minimal graph fanning out to two nodes that both write node_timings must
    merge (not raise InvalidUpdateError) and retain both keys."""
    from typing import Annotated, TypedDict

    from langgraph.graph import StateGraph, END

    class MiniState(TypedDict, total=False):
        node_timings: Annotated[dict, merge_node_timings]

    async def left(state):
        return {}

    async def right(state):
        return {}

    g = StateGraph(MiniState)
    g.add_node("left", timed_node("left", left))
    g.add_node("right", timed_node("right", right))
    # Fan out from entry to both, then both to END (same superstep convergence).
    g.set_conditional_entry_point(lambda s: ["left", "right"], {"left": "left", "right": "right"})
    g.add_edge("left", END)
    g.add_edge("right", END)
    compiled = g.compile()

    result = asyncio.run(compiled.ainvoke({}))
    assert set(result["node_timings"]) == {"left", "right"}
