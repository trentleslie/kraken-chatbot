"""Unit 0.3 — static baseline arm (mocked REST)."""
from tests.code_on_graph_spike.baseline import run_baseline

ITEM = {"trial_id": "x", "start_curie": "CHEBI:1", "gold_target_curie": "MONDO:1",
        "gold_bridge_curies": ["NCBIGene:5"], "stratum": "t2d"}


class FakeRest:
    def __init__(self, paths_by_pair):
        self.paths_by_pair = paths_by_pair
        self.kestrel_calls = 0

    async def multi_hop(self, start, end, **kw):
        self.kestrel_calls += 1
        return {"results": [{"paths": self.paths_by_pair.get((start[0], end[0]), [])}]}


async def test_hit_when_path_contains_all_gold():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    rec = await run_baseline(rest, ITEM)
    assert rec["hit"] is True and rec["terminal_state"] == "found"
    assert "NCBIGene:5" in rec["intermediates"]
    assert rec["method"] == "static"


async def test_miss_when_gold_bridge_absent():
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:9", "MONDO:1"]]})
    rec = await run_baseline(rest, ITEM)
    assert rec["hit"] is False and rec["terminal_state"] == "found"


async def test_empty_result_is_miss():
    rest = FakeRest({})
    rec = await run_baseline(rest, ITEM)
    assert rec["hit"] is False and rec["terminal_state"] == "empty"


async def test_transport_failure_is_not_a_miss():
    class Boom:
        kestrel_calls = 0
        async def multi_hop(self, *a, **k):
            raise RuntimeError("kestrel down")

    rec = await run_baseline(Boom(), ITEM)
    assert rec["terminal_state"] == "transport-failed" and rec["hit"] is False
    assert rec["hit_strict"] is False and rec["hit_any"] is False


async def test_multi_bridge_any_one_diverges_from_strict():
    # Two gold interior nodes; the path recovers only one -> strict miss, any-one hit.
    item = {**ITEM, "gold_bridge_curies": ["NCBIGene:5", "NCBIGene:6"]}
    rest = FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    rec = await run_baseline(rest, item)
    assert rec["hit_strict"] is False  # not ALL interior in one path
    assert rec["hit_any"] is True      # but ANY interior recovered
    assert rec["hit"] is True          # primary defaults to any_one
