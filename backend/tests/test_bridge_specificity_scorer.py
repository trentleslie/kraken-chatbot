"""Axis C, Unit 1 — pure DWPC bridge-specificity scorer + BridgeSpecificity model.

The scorer maps a bridge's intermediate-node CURIEs + KG degrees to a frozen
``BridgeSpecificity`` (structural genericity, NOT mechanism confidence). Length-normalized
geometric-mean DWPC (Himmelstein 2015, w=0.4) so 1-, 2-, 3-intermediate scaffolds are
comparable under one cut point. No I/O — deterministic and pure.

Run with: uv run python -m pytest tests/test_bridge_specificity_scorer.py -v
"""

import math

import pytest
from pydantic import ValidationError

from kestrel_backend.graph.nodes.bridge_specificity import (
    DAMPING_W,
    GENERIC_CUTOFF,
    MODERATE_CUT,
    SPECIFIC_CUT,
    bridge_specificity,
)
from kestrel_backend.graph.state import BridgeSpecificity


# --- model ---------------------------------------------------------------------------------

def test_model_field_names_are_pinned():
    # Contract L12/L17: axis E reads these exact names. Renaming requires a new ledger entry.
    assert set(BridgeSpecificity.model_fields) == {
        "score",
        "label",
        "intermediate_curies",
        "intermediate_degrees",
        "generic_intermediates",
    }


def test_model_is_frozen():
    spec = bridge_specificity(["X:1"], [1])
    with pytest.raises(ValidationError):
        spec.score = 0.0  # type: ignore[misc]


# --- happy paths ---------------------------------------------------------------------------

def test_single_degree_one_intermediate_is_maximally_specific():
    spec = bridge_specificity(["X:1"], [1])
    assert spec.score == pytest.approx(1.0)
    assert spec.label == "specific"
    assert spec.generic_intermediates == []
    assert spec.intermediate_curies == ["X:1"]
    assert spec.intermediate_degrees == [1]


def test_single_high_degree_intermediate_is_generic():
    d = GENERIC_CUTOFF * 5
    spec = bridge_specificity(["HUB:blood"], [d])
    assert spec.label == "generic"
    assert "HUB:blood" in spec.generic_intermediates
    assert 0.0 < spec.score < SPECIFIC_CUT


def test_two_mid_degree_intermediates_geometric_mean_in_range():
    degrees = [30, 60]
    spec = bridge_specificity(["A:1", "B:2"], degrees)
    expected = (30 * 60) ** (-DAMPING_W / 2)
    assert spec.score == pytest.approx(expected)
    assert 0.0 < spec.score <= 1.0
    assert spec.label in {"specific", "moderate", "generic"}


# --- length normalization (the reward axis must fire) --------------------------------------

def test_length_normalization_two_low_degree_nodes_stay_specific():
    # Two genuinely low-degree nodes MUST NOT be condemned purely for scaffold length.
    # Under a RAW product ((d1*d2)**-0.4) this would score much lower; length-normalization
    # (geometric mean) keeps it in the reward bucket.
    spec = bridge_specificity(["A:1", "B:2"], [10, 10])
    raw_product_score = (10 * 10) ** (-DAMPING_W)  # what a non-normalized DWPC would give
    assert spec.score > raw_product_score
    assert spec.label == "specific"
    assert spec.generic_intermediates == []


def test_specific_is_reachable_for_realistic_low_degree_scaffold():
    # Calibration guard (unit-level): the reward bucket is not decorative — a plausible
    # low-degree intermediate (tens of edges) reaches `specific`.
    spec = bridge_specificity(["A:1"], [20])
    assert spec.label == "specific"


# --- condemn-on-known (partial-unknown) ----------------------------------------------------

def test_condemn_on_known_high_degree_with_missing_degree():
    spec = bridge_specificity(["HUB:blood", "X:unknown"], [GENERIC_CUTOFF * 10, None])
    assert spec.label == "generic"
    assert "HUB:blood" in spec.generic_intermediates
    assert "X:unknown" not in spec.generic_intermediates
    # score computed over the known subset only
    assert spec.score == pytest.approx((GENERIC_CUTOFF * 10) ** (-DAMPING_W))
    assert spec.intermediate_degrees == [GENERIC_CUTOFF * 10, None]


# --- unknown reserved for all-missing ------------------------------------------------------

def test_all_degrees_missing_is_unknown():
    spec = bridge_specificity(["A:1", "B:2"], [None, None])
    assert spec.label == "unknown"
    assert spec.score is None
    assert spec.generic_intermediates == []
    assert spec.intermediate_degrees == [None, None]


# --- zero intermediates (2-node bridge) ----------------------------------------------------

def test_zero_intermediates_is_maximally_specific():
    spec = bridge_specificity([], [])
    assert spec.score == pytest.approx(1.0)
    assert spec.label == "specific"
    assert spec.intermediate_curies == []
    assert spec.intermediate_degrees == []
    assert spec.generic_intermediates == []


# --- boundaries ----------------------------------------------------------------------------

def test_generic_cutoff_boundary_is_strict_greater_than():
    # A node whose degree EXCEEDS the cutoff is generic; degree == cutoff is NOT flagged.
    at = bridge_specificity(["A:1"], [GENERIC_CUTOFF])
    over = bridge_specificity(["A:1"], [GENERIC_CUTOFF + 1])
    assert "A:1" not in at.generic_intermediates
    assert "A:1" in over.generic_intermediates


def test_score_cut_points_inclusive_on_upper_bucket():
    # Construct a single-intermediate degree whose score lands exactly on SPECIFIC_CUT.
    # score = d**(-W) == SPECIFIC_CUT  =>  d = SPECIFIC_CUT ** (-1/W)
    d = round(SPECIFIC_CUT ** (-1.0 / DAMPING_W))
    spec = bridge_specificity(["A:1"], [d])
    # score >= SPECIFIC_CUT is inclusive -> specific (unless it is a generic hub, which it is not here)
    assert spec.score == pytest.approx(d ** (-DAMPING_W))
    if d <= GENERIC_CUTOFF and spec.score >= SPECIFIC_CUT:
        assert spec.label == "specific"


# --- degree-zero guard ---------------------------------------------------------------------

def test_degree_zero_is_clamped_not_crash():
    # An isolated node (results_count == 0) must not raise (0 ** -0.4 = inf); clamp to 1.
    spec = bridge_specificity(["ISO:1"], [0])
    assert spec.score == pytest.approx(1.0)
    assert spec.label == "specific"
    assert spec.intermediate_degrees == [0]  # raw degree preserved


# --- determinism ---------------------------------------------------------------------------

def test_determinism_same_inputs_same_output():
    a = bridge_specificity(["A:1", "B:2"], [30, 900])
    b = bridge_specificity(["A:1", "B:2"], [30, 900])
    assert a == b
    assert a.model_dump() == b.model_dump()


def test_constants_are_ordered_and_sane():
    assert 0.0 < DAMPING_W < 1.0
    assert 0.0 < MODERATE_CUT < SPECIFIC_CUT <= 1.0
    assert GENERIC_CUTOFF >= 1
