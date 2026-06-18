"""U4 — v1 scoring (ratio + counts, min-leg gating, binding-leg floor).

Run with: uv run python -m pytest tests/test_bridge_grounding_scoring.py -v
"""

import pytest

from src.kestrel_backend.bridge_grounding.scoring import (
    DECISION_GROUNDED,
    DECISION_INSUFFICIENT,
    DECISION_UNGROUNDED,
    score_chain,
    score_leg,
)


# --- score_leg ------------------------------------------------------------------------

def test_leg_support_fraction_basic():
    ls = score_leg(2, 0, 1)  # 2 support / (2+0+1)
    assert ls.total_labeled == 3
    assert ls.support_fraction == pytest.approx(2 / 3)


def test_leg_neither_inflates_denominator_reads_low():
    ls = score_leg(2, 1, 47)  # 2 / 50
    assert ls.support_fraction == pytest.approx(0.04)


def test_leg_empty_is_none():
    ls = score_leg(0, 0, 0)
    assert ls.total_labeled == 0 and ls.support_fraction is None


# --- score_chain: min-leg gating ------------------------------------------------------

def test_chain_headline_is_weaker_leg_strong_leg_retained():
    # leg0 strong (8/10=0.8), leg1 weak (5/10=0.5)
    cs = score_chain([(8, 0, 2), (5, 0, 5)])
    assert cs.support_fraction == pytest.approx(0.5)        # headline = weaker
    assert cs.strong_leg_fraction == pytest.approx(0.8)     # strong leg retained
    assert cs.strong_leg_n == 10
    assert cs.weak_leg_index == 1
    assert cs.decision == DECISION_GROUNDED                 # 0.5 >= threshold


def test_chain_lossy_min_distinguished_by_strong_leg():
    # (0.95, 0.55) and (0.56, 0.55) share the headline but differ in the strong-leg key.
    a = score_chain([(19, 0, 1), (11, 0, 9)])   # 0.95, 0.55
    b = score_chain([(14, 0, 11), (11, 0, 9)])  # 0.56, 0.55
    assert a.support_fraction == pytest.approx(b.support_fraction, abs=0.02)
    assert a.strong_leg_fraction == pytest.approx(0.95)
    assert b.strong_leg_fraction == pytest.approx(0.56, abs=0.01)


def test_chain_equal_legs_still_report_strong_leg():
    # Tied legs (0.5, 0.5): the secondary key must still be reported (the OTHER leg), not None,
    # so a downstream ranker sees the strong-leg signal even when both fractions are equal.
    cs = score_chain([(5, 0, 5), (5, 0, 5)])  # both 5/10 = 0.5
    assert cs.support_fraction == pytest.approx(0.5)
    assert cs.strong_leg_fraction == pytest.approx(0.5)  # reported, not None
    assert cs.strong_leg_n == 10


def test_chain_ungrounded_when_headline_below_threshold():
    cs = score_chain([(8, 0, 2), (2, 3, 5)])  # weak leg 2/10 = 0.2
    assert cs.support_fraction == pytest.approx(0.2)
    assert cs.decision == DECISION_UNGROUNDED


# --- binding-leg floor / insufficiency ------------------------------------------------

def test_binding_leg_floor_blocks_thin_leg():
    # leg1 has only 1 labeled abstract (< k=2) -> insufficient, not a one-abstract score.
    cs = score_chain([(5, 0, 1), (1, 0, 0)])
    assert cs.decision == DECISION_INSUFFICIENT
    assert cs.support_fraction == 0.0 and cs.weak_leg_index is None


def test_empty_leg_is_insufficient():
    assert score_chain([(0, 0, 0), (5, 0, 1)]).decision == DECISION_INSUFFICIENT


def test_both_empty_is_insufficient():
    assert score_chain([(0, 0, 0), (0, 0, 0)]).decision == DECISION_INSUFFICIENT


def test_floor_is_configurable():
    # With floor=3, a 2-abstract leg is insufficient; with floor=2 it scores.
    assert score_chain([(5, 0, 1), (1, 1, 0)], binding_leg_floor=3).decision == DECISION_INSUFFICIENT
    assert score_chain([(5, 0, 1), (1, 1, 0)], binding_leg_floor=2).decision != DECISION_INSUFFICIENT


# --- keep-first stability (weak-leg identity) -----------------------------------------

def test_weak_leg_identity_stable_under_co_mention_reassignment():
    # A shared abstract counted on leg0 vs leg1: the weaker leg (and headline) stays the same
    # when the strong leg dominates -> stable, no flip flag needed.
    assign_a = score_chain([(9, 0, 1), (3, 0, 7)])  # leg0 0.9, leg1 0.3
    assign_b = score_chain([(8, 0, 1), (3, 0, 8)])  # shared abstract moved off leg0
    assert assign_a.weak_leg_index == assign_b.weak_leg_index == 1


def test_weak_leg_flip_is_detectable():
    # Near-tie legs: moving one abstract flips which leg is weaker -> weak_leg_index differs,
    # so a caller can flag the chain lower-confidence.
    assign_a = score_chain([(5, 0, 5), (6, 0, 4)])   # 0.5 vs 0.6 -> weak=0
    assign_b = score_chain([(6, 0, 4), (5, 0, 5)])   # swapped -> weak=1
    assert assign_a.weak_leg_index != assign_b.weak_leg_index


# --- never raises ---------------------------------------------------------------------

def test_never_raises_on_degenerate_inputs():
    for legs in ([], [(0, 0, 0)], [(1, 0, 0), (0, 0, 0)]):
        cs = score_chain(legs)  # must not raise
        assert cs.decision in (DECISION_GROUNDED, DECISION_UNGROUNDED, DECISION_INSUFFICIENT)
