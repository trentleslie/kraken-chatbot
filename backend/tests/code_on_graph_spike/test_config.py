"""Unit 0.1 — frozen pre-registration config."""
import pytest

from tests.code_on_graph_spike.config import CONFIG, SpikeConfig


def test_config_loads_with_typed_thresholds():
    assert CONFIG.recall_lift_abs == 0.15
    assert CONFIG.alpha == 0.05
    assert CONFIG.mcnemar_exact_primary is True
    assert CONFIG.turn_cap == 5
    assert CONFIG.k_reruns >= 1
    assert CONFIG.n_floor == 30
    assert isinstance(CONFIG.drugmechdb_sample_seed, int)


def test_config_is_frozen():
    with pytest.raises(Exception):
        CONFIG.recall_lift_abs = 0.99  # type: ignore[misc]


def test_multi_hop_limit_is_a_single_frozen_knob_for_both_arms():
    # finding #2: the per-query limit determines whether the gold path surfaces; it must be
    # one value applied identically to baseline and iterate arms.
    assert isinstance(CONFIG.multi_hop_limit, int)
    # the loop may accumulate beyond one query's worth (its mechanism); bounded by turn cap.
    assert CONFIG.aggregate_path_budget == (CONFIG.turn_cap + 1) * CONFIG.multi_hop_limit


def test_a_fresh_instance_uses_the_same_defaults():
    assert SpikeConfig().drugmechdb_sample_seed == CONFIG.drugmechdb_sample_seed


def test_corrected_pre_registration_values():
    # 2026-06-07 corrections: any-one primary bridge unit, N=100, pinned DrugMechDB source.
    assert CONFIG.primary_bridge_unit == "any_one"
    assert CONFIG.n_target == 100
    assert CONFIG.drugmechdb_commit_sha == "aef224217071216748740c10faeb6db8e3f15901"
