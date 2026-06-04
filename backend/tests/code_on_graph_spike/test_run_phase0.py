"""Unit 0.5 — Phase-0 orchestrator plumbing (mocked rest + llm, no live calls)."""
import json

from tests.code_on_graph_spike.run_phase0 import run_phase0

ITEMS = [{"trial_id": f"i{i}", "start_curie": "CHEBI:1", "gold_target_curie": "MONDO:1",
          "gold_bridge_curies": ["NCBIGene:5"], "stratum": "random"} for i in range(4)]


class FakeRest:
    def __init__(self):
        self.kestrel_calls = 0

    async def multi_hop(self, start, end, **kw):
        self.kestrel_calls += 1
        return {"results": [{"paths": [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]}]}

    async def equivalent_ids(self, curie):
        return set()

    async def _post(self, path, body):
        self.kestrel_calls += 1
        return {"results": []}

    async def hybrid_search(self, text, limit=3):
        self.kestrel_calls += 1
        return []


async def _llm(prompt, system):
    if prompt.startswith("Task:"):
        return json.dumps({"action": "query", "verb": "multi_hop",
                           "start_node_ids": ["CHEBI:1"], "end_node_ids": ["MONDO:1"],
                           "max_path_length": 3}), None
    return json.dumps({"action": "done"}), None


async def test_orchestrator_chains_pilot_arms_score_gate():
    res = await run_phase0(FakeRest(), _llm, ITEMS, k=1, n_pilot=2)
    assert set(res) == {"pilot", "gate", "baseline_records", "iterate_records"}
    assert res["gate"]["verdict"] in {"PROCEED-TO-PHASE-1", "NO-GO", "INCONCLUSIVE"}
    assert res["gate"]["n"] == 4
    assert len(res["iterate_records"]) == 4
    # both arms recover the gold here -> concordant, no lift
    assert res["gate"]["baseline_recall"] == 1.0
    assert res["gate"]["iterate_recall"] == 1.0
