"""U8 Tier A — panel fixtures + the pre-registered gate evaluator (pure; no SDK/network).

Run with: uv run python -m pytest tests/test_bridge_grounding_panel.py -v
"""

from src.kestrel_backend.bridge_grounding.panel import (
    PANEL,
    PRE_REGISTERED,
    ChainResult,
    build_bridge,
    evaluate_tier_a,
)


def _result(name, polarity, frac, *, decision="grounded", min_n=10, off_topic=0.1):
    return ChainResult(
        name=name, polarity=polarity, decision=decision,
        support_fraction=frac, min_labeled_per_leg=min_n, max_off_topic_fraction=off_topic)


# --- panel + fixtures -----------------------------------------------------------------

def test_panel_has_positives_negatives_and_hard_negative():
    pols = {c.polarity for c in PANEL}
    assert {"positive", "negative", "hard_negative"} <= pols
    assert sum(c.polarity == "positive" for c in PANEL) >= 2


def test_build_bridge_produces_valid_3node_bridge():
    b = build_bridge(PANEL[0])
    assert len(b.entities) == 3 and len(b.entity_names) == 3
    assert len(b.predicates) == 2 and len(b.predicate_directions) == 2
    assert b.tier == 2 and b.grounding is None


# --- gate evaluator -------------------------------------------------------------------

def _passing_set():
    return [
        _result("p1", "positive", 0.80),
        _result("p2", "positive", 0.70),
        _result("n1", "negative", 0.20),
        _result("n2", "negative", 0.30),
        _result("h1", "hard_negative", 0.25),
    ]


def test_gate_passes_on_clean_separation():
    v = evaluate_tier_a(_passing_set())
    assert v["passed"] is True
    assert all(v["checks"].values())
    assert v["margin"] >= PRE_REGISTERED.margin_min


def test_gate_fails_on_thin_margin_only():
    # positive 0.60 (>= floor), negative 0.35 (<= ceiling), but margin 0.25 < 0.30.
    res = [_result("p1", "positive", 0.60), _result("p2", "positive", 0.65),
           _result("n1", "negative", 0.35), _result("h1", "hard_negative", 0.30)]
    v = evaluate_tier_a(res)
    assert v["passed"] is False
    assert v["checks"]["margin_ok"] is False
    assert v["checks"]["positives_above_floor"] is True
    assert v["checks"]["negatives_below_ceiling"] is True


def test_gate_fails_when_negative_scores_high():
    res = _passing_set()
    res[2] = _result("n1", "negative", 0.50)  # above the 0.35 ceiling
    v = evaluate_tier_a(res)
    assert v["passed"] is False
    assert v["checks"]["negatives_below_ceiling"] is False


def test_gate_fails_on_off_topic_pool():
    res = _passing_set()
    res[0] = _result("p1", "positive", 0.80, off_topic=0.60)  # > 0.50 ceiling
    v = evaluate_tier_a(res)
    assert v["passed"] is False
    assert v["checks"]["pools_on_topic"] is False


def test_gate_not_evaluable_on_insufficient_chain():
    res = _passing_set()
    res[1] = _result("p2", "positive", None, decision="insufficient_literature")
    v = evaluate_tier_a(res)
    assert v["passed"] is None and v["reason"] == "not_evaluable"
    assert "p2" in v["not_evaluable"]


def test_gate_not_evaluable_when_labeled_n_below_floor():
    res = _passing_set()
    res[0] = _result("p1", "positive", 0.80, min_n=3)  # < min_labeled_per_leg (5)
    v = evaluate_tier_a(res)
    assert v["passed"] is None and v["reason"] == "not_evaluable"
    assert "p1" in v["not_evaluable"]


def test_gate_not_evaluable_when_polarity_missing():
    res = [_result("p1", "positive", 0.8), _result("p2", "positive", 0.7)]  # no negatives
    v = evaluate_tier_a(res)
    assert v["passed"] is None and v["reason"] == "panel_missing_polarity"


def test_empty_results_not_evaluable():
    assert evaluate_tier_a([])["passed"] is None
