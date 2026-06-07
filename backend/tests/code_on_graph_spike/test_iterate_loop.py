"""Unit 0.4 — iterate-loop executor (mocked LLM + REST, no SDK cost)."""
import json

from tests.code_on_graph_spike.config import CONFIG
from tests.code_on_graph_spike.iterate_loop import (
    run_iterate_loop, run_iterate_item_k, _extract_spec, _spec_curies,
)

ITEM = {"trial_id": "x", "start_curie": "CHEBI:1", "gold_target_curie": "MONDO:1",
        "gold_bridge_curies": ["NCBIGene:5"], "stratum": "random"}


class FakeRest:
    def __init__(self, paths_by_pair=None, raise_on_multi=False):
        self.paths_by_pair = paths_by_pair or {}
        self.raise_on_multi = raise_on_multi
        self.kestrel_calls = 0

    async def multi_hop(self, start, end, **kw):
        self.kestrel_calls += 1
        if self.raise_on_multi:
            raise RuntimeError("transport down")
        key = (start[0] if start else None, end[0] if end else None)
        return {"results": [{"paths": self.paths_by_pair.get(key, [])}]}

    async def equivalent_ids(self, curie):
        return set()

    async def _post(self, path, body):
        self.kestrel_calls += 1
        return {"results": []}

    async def hybrid_search(self, text, limit=3):
        self.kestrel_calls += 1
        return []


def _query_then_done(spec):
    """Stateless LLM: emit `spec` until a refining query has run (prompt reports
    'new paths'), then 'done'."""
    async def fn(prompt, system):
        if "new paths" in prompt:
            return json.dumps({"action": "done"}), None
        return json.dumps(spec), None
    return fn


def _always(spec):
    async def fn(prompt, system):
        return json.dumps(spec), None
    return fn


# --- pure helpers ---

def test_extract_spec_tolerates_prose():
    assert _extract_spec('sure! {"action":"done"} ok') == {"action": "done"}
    assert _extract_spec("no json here") is None


def test_spec_curies_collects_endpoints():
    spec = {"start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1", "MONDO:2"]}
    assert _spec_curies(spec) == ["CHEBI:1", "MONDO:1", "MONDO:2"]


# --- loop behavior ---

async def test_loop_hit_when_query_recovers_gold():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["hit"] is True and rec["terminal_state"] == "found"
    assert rec["grounding_violations"] == 0
    assert rec["finding_level_hallucination"] == 0  # win via the grounded seed


async def test_turn_cap_hit_is_miss():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    gold_absent = {**ITEM, "gold_bridge_curies": ["NCBIGene:9999"]}
    llm = _always({"action": "query", "verb": "multi_hop",
                   "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    rec = await run_iterate_loop(rest, gold_absent, llm)
    assert rec["terminal_state"] == "turn-cap-hit" and rec["hit"] is False
    assert rec["turns"] == CONFIG.turn_cap


async def test_invalid_verb_is_reprompted_then_recovers():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})

    async def llm(prompt, system):
        if "verb must be one of" in prompt:
            return json.dumps({"action": "query", "verb": "multi_hop",
                               "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3}), None
        if "new paths" in prompt:
            return json.dumps({"action": "done"}), None
        return json.dumps({"action": "query", "verb": "BOGUS"}), None  # initial -> invalid verb

    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["hit"] is True  # baseline seed recovers the gold; invalid verb is re-prompted


async def test_grounding_violation_on_invented_curie():
    rest = FakeRest({})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:9999"],  # never returned/grounded
                            "end_node_ids": ["MONDO:1"], "max_path_length": 2})
    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["grounding_violations"] >= 1


async def test_grounding_casing_is_not_a_violation():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): []})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["chebi:1"],   # casing of the grounded start
                            "end_node_ids": ["mondo:1"], "max_path_length": 2})
    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["grounding_violations"] == 0


async def test_loop_finding_level_hallucination_when_win_needs_ungrounded_query():
    # Seed recovers nothing; the win comes ONLY from a query with an ungrounded start CURIE.
    rest = FakeRest({
        ("CHEBI:1", "MONDO:1"): [],                                          # grounded seed: no win
        ("CHEBI:9999", "MONDO:1"): [["CHEBI:9999", "NCBIGene:5", "MONDO:1"]],  # ungrounded query: win
    })
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:9999"],  # never returned -> ungrounded
                            "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["hit"] is True
    assert rec["grounding_violations"] >= 1
    assert rec["finding_level_hallucination"] == 1  # the win depended on the ungrounded query


async def test_transport_error_mid_loop_does_not_crash():
    rest = FakeRest(raise_on_multi=True)
    llm = _always({"action": "query", "verb": "multi_hop",
                   "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 2})
    rec = await run_iterate_loop(rest, ITEM, llm)
    assert rec["hit"] is False and "errors" in rec


async def test_item_k_majority_and_variance_band():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    res = await run_iterate_item_k(rest, ITEM, llm, k=3)
    assert res["hit"] is True
    assert res["variance"] == "stable"
    assert len(res["hit_runs"]) == 3


async def test_loop_multi_bridge_any_one_diverges_from_strict():
    # Two gold interior nodes; the loop's accumulated paths recover only one.
    item = {**ITEM, "gold_bridge_curies": ["NCBIGene:5", "NCBIGene:6"]}
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    rec = await run_iterate_loop(rest, item, llm)
    assert rec["hit_strict"] is False and rec["hit_any"] is True and rec["hit"] is True


async def test_item_k_reports_both_bridge_units():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    llm = _query_then_done({"action": "query", "verb": "multi_hop",
                            "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"], "max_path_length": 3})
    res = await run_iterate_item_k(rest, ITEM, llm, k=3)
    assert res["hit_strict"] is True and res["hit_any"] is True
    assert len(res["hit_strict_runs"]) == 3 and len(res["hit_any_runs"]) == 3
