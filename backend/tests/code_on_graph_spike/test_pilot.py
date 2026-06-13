"""Unit 0.3 — pilot: R0 per stratum + powered-N (Monte-Carlo McNemar)."""
import numpy as np

from tests.code_on_graph_spike.pilot import (
    compute_r0, mcnemar_power, powered_n, run_pilot,
)


def test_compute_r0_overall_and_per_stratum():
    recs = [
        {"hit": True, "stratum": "t2d"}, {"hit": False, "stratum": "t2d"},
        {"hit": True, "stratum": "random"},
    ]
    r0 = compute_r0(recs)
    assert abs(r0["overall"] - 2 / 3) < 1e-9
    assert r0["by_stratum"]["t2d"] == 0.5
    assert r0["by_stratum"]["random"] == 1.0


def test_mcnemar_power_increases_with_n():
    rng = np.random.default_rng(0)
    low = mcnemar_power(15, 0.30, 0.15, 0.05, 200, rng)
    high = mcnemar_power(150, 0.30, 0.15, 0.05, 200, rng)
    assert high > low


def test_powered_n_smaller_for_larger_effect():
    # fixed discordance, bigger marginal effect -> fewer items needed (robust monotonic)
    n_small_effect = powered_n(0.30, 0.10, reps=200, n_max=400)
    n_large_effect = powered_n(0.30, 0.25, reps=200, n_max=400)
    assert n_large_effect < n_small_effect


def test_powered_n_is_plausible_for_config_params():
    # pi_d=0.25 prior, 15pp effect: expect a usable-but-nontrivial N
    n = powered_n(0.25, 0.15, reps=300, n_max=400)
    assert 20 <= n <= 400


class _FakeRest:
    def __init__(self, paths_by_pair):
        self.paths_by_pair = paths_by_pair
        self.kestrel_calls = 0

    async def multi_hop(self, start, end, **kw):
        self.kestrel_calls += 1
        return {"results": [{"paths": self.paths_by_pair.get((start[0], end[0]), [])}]}


async def test_run_pilot_inconclusive_when_n_below_floor():
    items = [{"trial_id": f"i{i}", "start_curie": "CHEBI:1", "gold_target_curie": "MONDO:1",
              "gold_bridge_curies": ["NCBIGene:5"], "stratum": "random"} for i in range(5)]
    rest = _FakeRest({("CHEBI:1", "MONDO:1"): [["CHEBI:1", "NCBIGene:5", "MONDO:1"]]})
    res = await run_pilot(rest, items, n_subset=3)
    assert res["verdict"] == "INCONCLUSIVE"   # 5 items < n_floor (30)
    assert res["r0"]["overall"] == 1.0
    assert res["gate_form"] == "relative"     # R0=1.0 > r0_relative_switch
