"""Axis C, Unit 4 — 1-MNA retrodiction + scorer-math + distribution measurement hook.

The 1-MNA hypothesis ("TNFRSF10A -> 1-MNA via blood, placenta, cancer") INVERTED. Its
intermediates are near-universal, extremely high-degree KG nodes — "both found in blood" carries
near-zero evidential weight. This module proves the DWPC scorer would have FLAGGED that scaffold as
`generic` using the RECORDED REAL degrees of those nodes.

Two distinct kinds of test live here, deliberately NOT conflated (a reviewer flagged an earlier
version whose "live retrodiction" actually asserted against fabricated placeholder degrees):

  * REAL-degree retrodiction — reads tests/fixtures/bridge_specificity_1mna_degrees.json, which is
    produced ONLY by the live calibration probe (assessment_data/bridge_specificity_calibration.py)
    against Kestrel. These assertions are meaningful only because the degrees are measured, not
    hand-picked. Regenerate the fixture by running the probe; do NOT hand-edit degrees into it.
  * SCORER MATH — uses SYNTHETIC degrees, clearly labelled, to prove the pure scorer transform (that
    the `specific` reward band is reachable in principle). No claim about real KG data.

The real degrees drove the cut-point calibration: the population splits into a genuine-specific
cluster {115, 143} and a hub cluster {1270 .. 10000} with a wide gap, so the constants are set to
SPECIFIC_CUT=0.12 (single-intermediate degree <= ~200 -> specific) and MODERATE_CUT=0.063
(degree <= ~1000, coinciding with GENERIC_CUTOFF). Saturated degrees (== the one_hop query cap) are
treated as a FLOOR and force-generic rather than raising the query cap — a mega-hub is generic
regardless of its exact degree, so paying for a higher cap buys nothing.

Run with: uv run python -m pytest tests/test_bridge_specificity_retrodiction.py -v
"""

import json
from pathlib import Path

import pytest

from kestrel_backend.graph.nodes.bridge_specificity import (
    GENERIC_CUTOFF,
    MODERATE_CUT,
    SPECIFIC_CUT,
    bridge_specificity,
    summarize_specificity,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "bridge_specificity_1mna_degrees.json"


@pytest.fixture(scope="module")
def recorded() -> dict:
    data = json.loads(_FIXTURE.read_text())
    # Guard against a hand-faked fixture: the retrodiction is only load-bearing if these degrees
    # came from the live probe. A PLACEHOLDER marker means nobody has recorded real degrees yet.
    if "PLACEHOLDER" in json.dumps(data.get("provenance", {})):
        pytest.skip(
            "bridge_specificity_1mna_degrees.json holds PLACEHOLDER degrees — run the live probe "
            "(assessment_data/bridge_specificity_calibration.py) to record real Kestrel degrees "
            "before the retrodiction can assert anything."
        )
    return data


# --- REAL-degree retrodiction: the 1-MNA scaffold must be condemned as generic ------------------

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


# --- REAL-degree retrodiction: a low-degree metabolite is rewarded relative to the hubs ---------

def test_low_degree_control_scores_above_generic_scaffold(recorded):
    # The recorded control (1-methylnicotinamide) is a genuinely low-degree metabolite. The
    # meaningful retrodiction contrast is that it is NOT condemned as a generic hub and scores
    # strictly above the all-hub scaffold — i.e. the DWPC signal separates them on real degrees.
    # With the calibrated constants the control's real degree (~143) now lands `specific` — a
    # genuinely low-degree metabolite is rewarded, not merely "not condemned".
    control = recorded["control_specific"]
    curies = list(control)
    degrees = [control[c]["degree"] for c in curies]
    control_spec = bridge_specificity(curies, degrees)

    scaffold = recorded["one_mna_scaffold"]
    scaffold_curies = list(scaffold)
    scaffold_spec = bridge_specificity(
        scaffold_curies, [scaffold[c]["degree"] for c in scaffold_curies]
    )

    assert control_spec.label == "specific"
    assert control_spec.generic_intermediates == []
    assert control_spec.score is not None and scaffold_spec.score is not None
    assert control_spec.score > scaffold_spec.score


# --- SCORER MATH (synthetic degrees): the `specific` reward band is reachable in principle -------

def test_scorer_specific_band_reachable_synthetic():
    # Pure scorer transform on a SYNTHETIC low degree — no real-KG claim. Proves the reward axis is
    # not dead code: a sufficiently specific (low-degree) intermediate does clear SPECIFIC_CUT.
    spec = bridge_specificity(["SYNTHETIC:1"], [5])
    assert spec.label == "specific"
    assert spec.score is not None and spec.score >= SPECIFIC_CUT
    assert spec.generic_intermediates == []


# --- REAL-population reward reachability: closed by the Unit-4 recalibration ---------------------

def test_specific_reachable_on_recorded_population(recorded):
    # Both label bands must be reachable on the REAL intermediate-degree population — else the
    # reward axis is decorative (only the penalty side ever fires). Calibrated cut points
    # (SPECIFIC_CUT=0.12 / MODERATE_CUT=0.063) reward the genuine-specific cluster {115, 143} while
    # the hubs (>= 1270, incl. saturated) stay generic. This previously xfail'd under the
    # placeholder constants; the recalibration closes the gap.
    population = recorded["recorded_intermediate_population"]
    labels = [bridge_specificity([f"N:{i}"], [d]).label for i, d in enumerate(population)]
    n_specific = labels.count("specific")
    assert n_specific >= 2, f"reward bucket collapsed: labels={labels}"
    # the generic penalty side must also fire (the hubs)
    assert labels.count("generic") >= 1


# --- saturation is treated as a FLOOR -> force-generic -------------------------------------------

def test_saturated_degree_is_forced_generic():
    from kestrel_backend.graph.nodes.bridge_specificity import SATURATION_DEGREE

    # A degree measured AT the one_hop query cap is a lower bound on a mega-hub, not an exact count,
    # so the intermediate must be condemned generic regardless of the score-band arithmetic.
    spec = bridge_specificity(["SAT:1"], [SATURATION_DEGREE])
    assert spec.label == "generic"
    assert spec.generic_intermediates == ["SAT:1"]


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
