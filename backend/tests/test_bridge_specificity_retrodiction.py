"""Axis C, Unit 4 — 1-MNA retrodiction + distribution measurement hook.

The 1-MNA hypothesis ("TNFRSF10A -> 1-MNA via blood, placenta, cancer") INVERTED. Its
intermediates are near-universal, extremely high-degree KG nodes — "both found in blood" carries
near-zero evidential weight. This test proves the DWPC scorer would have FLAGGED that scaffold as
`generic` (and a low-degree control as `specific`) using the recorded real degrees of those nodes,
NOT hand-picked round numbers. Degrees come from the calibration fixture the live probe produces
(assessment_data/bridge_specificity_calibration.py); the cut-point constants are tuned on an
independent distribution, so the retrodiction is not fitted to this same fixture.

Run with: uv run python -m pytest tests/test_bridge_specificity_retrodiction.py -v
"""

import json
from pathlib import Path

import pytest

from kestrel_backend.graph.nodes.bridge_specificity import (
    GENERIC_CUTOFF,
    MODERATE_CUT,
    bridge_specificity,
    summarize_specificity,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "bridge_specificity_1mna_degrees.json"


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(_FIXTURE.read_text())


# --- retrodiction: the 1-MNA scaffold must be condemned as generic ------------------------

def test_1mna_scaffold_labels_generic(recorded):
    scaffold = recorded["one_mna_scaffold"]
    curies = list(scaffold)
    degrees = [scaffold[c]["degree"] for c in curies]

    spec = bridge_specificity(curies, degrees)

    assert spec.label == "generic"
    # all three near-universal hubs are flagged as the offending generic intermediates
    assert set(spec.generic_intermediates) == set(curies)
    # aggregate score sits in the generic band (well below the specific/moderate reward buckets)
    assert spec.score is not None and spec.score < MODERATE_CUT
    # and every recorded degree genuinely exceeds the per-node generic cutoff
    assert all(d > GENERIC_CUTOFF for d in degrees)


# --- retrodiction control: a low-degree intermediate must reach specific -------------------

def test_low_degree_control_labels_specific(recorded):
    control = recorded["control_specific"]
    curies = list(control)
    degrees = [control[c]["degree"] for c in curies]

    spec = bridge_specificity(curies, degrees)

    assert spec.label == "specific"
    assert spec.generic_intermediates == []


# --- calibration guard: the reward bucket is reachable on the recorded population ----------

def test_specific_is_reachable_on_recorded_population(recorded):
    # Under the calibrated constants a nontrivial fraction of the real intermediate-degree
    # population must land `specific` — else the reward axis is decorative (only the penalty
    # side ever fires). Each population degree is a single-intermediate scaffold here.
    population = recorded["recorded_intermediate_population"]
    labels = [bridge_specificity([f"N:{i}"], [d]).label for i, d in enumerate(population)]
    n_specific = labels.count("specific")
    assert n_specific >= 2, f"reward bucket collapsed: labels={labels}"
    # and the generic penalty side also fires (the hubs)
    assert labels.count("generic") >= 1


# --- distribution measurement hook --------------------------------------------------------

def test_summarize_specificity_counts_labels():
    smap = {
        ("a",): bridge_specificity(["X:1"], [5]),        # specific
        ("b",): bridge_specificity(["X:2"], [GENERIC_CUTOFF * 5]),  # generic
        ("c",): bridge_specificity(["X:3"], [None]),     # unknown
    }
    summary = summarize_specificity(smap)

    assert summary["scored"] == 3
    assert summary["counts"]["specific"] == 1
    assert summary["counts"]["generic"] == 1
    assert summary["counts"]["unknown"] == 1
    # score summary computed only over non-None scores (specific + generic here)
    assert "score_min" in summary and "score_median" in summary and "score_max" in summary
    assert summary["score_max"] >= summary["score_min"]


def test_summarize_specificity_empty_map():
    summary = summarize_specificity({})
    assert summary["scored"] == 0
    assert summary["counts"] == {"specific": 0, "moderate": 0, "generic": 0, "unknown": 0}
    assert "score_min" not in summary  # no scores -> no score summary keys
