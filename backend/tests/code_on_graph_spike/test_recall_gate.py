"""Unit 0.5 — recall scorer + Phase-0 gate (pure, no network)."""
from tests.code_on_graph_spike.recall_scorer import build_table, score
from tests.code_on_graph_spike.gate_recall import evaluate_gate

PILOT_OK = {"gate_form": "absolute", "powered_n": 15}


def _make(a, b, c, d, flap_discordant=False, stratum="random"):
    """Build paired baseline/iterate records with a:both-hit b:base-only c:iter-only d:both-miss."""
    bl, it, i = [], [], 0

    def add(bl_hit, it_hit, flap=False):
        nonlocal i
        tid = f"t{i}"; i += 1
        bl.append({"trial_id": tid, "hit": bl_hit, "hit_strict": bl_hit, "hit_any": bl_hit,
                   "stratum": stratum})
        it.append({"trial_id": tid, "hit": it_hit, "hit_strict": it_hit, "hit_any": it_hit,
                   "stratum": stratum, "variance": "flapping" if flap else "stable",
                   "grounding_violations": 0, "llm_calls": 2, "runs": [{"kestrel_calls": 3}]})

    for _ in range(a): add(True, True)
    for _ in range(b): add(True, False, flap_discordant)
    for _ in range(c): add(False, True, flap_discordant)
    for _ in range(d): add(False, False)
    return bl, it


def test_build_table_cells_and_recalls():
    bl, it = _make(5, 1, 12, 2)
    t = build_table(bl, it)
    assert (t["a"], t["b"], t["c"], t["d"]) == (5, 1, 12, 2)
    assert t["baseline_recall"] == 6 / 20
    assert t["iterate_recall"] == 17 / 20
    assert t["concordant_miss"] == 2


def test_score_includes_per_stratum():
    bl, it = _make(3, 0, 4, 0)
    s = score(bl, it)
    assert s["overall"]["n"] == 7
    assert "random" in s["by_stratum"]


def test_gate_proceeds_on_significant_lift():
    bl, it = _make(5, 1, 12, 2)
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "PROCEED-TO-PHASE-1"
    assert g["significant"] and g["lift_meets_threshold"]


def test_gate_query_arg_leakage_is_caveat_not_nogo():
    # Query-argument leakage that did NOT drive a win is reported, not a kill (corrected R9).
    bl, it = _make(5, 1, 12, 2)
    it[0]["grounding_violations"] = 3  # leakage on a non-winning-dependent item
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "PROCEED-TO-PHASE-1"
    assert g["query_arg_leakage"] == 3 and g["finding_level_hallucinations"] == 0


def test_gate_finding_level_hallucination_is_nogo_override():
    # A win that exists only via an ungrounded query is a hard NO-GO.
    bl, it = _make(5, 1, 12, 2)
    it[0]["finding_level_hallucinations"] = 1
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "NO-GO" and g["finding_level_hallucinations"] == 1


def test_gate_inconclusive_below_powered_n():
    bl, it = _make(5, 1, 12, 2)
    g = evaluate_gate(score(bl, it), {"gate_form": "absolute", "powered_n": 100}, it)
    assert g["verdict"] == "INCONCLUSIVE"


def test_gate_inconclusive_when_discordant_flaps():
    bl, it = _make(5, 1, 12, 2, flap_discordant=True)
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "INCONCLUSIVE"


def test_gate_nogo_when_no_lift():
    bl, it = _make(10, 2, 2, 6)  # iterate ties baseline -> not significant, no lift
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "NO-GO"


def test_absolute_gate_does_not_fall_through_to_relative():
    # Greptile P1: under the "absolute" form, a sub-threshold abs_lift must NOT pass via the
    # relative recover_frac. a=75,b=0,c=13,d=12 -> abs_lift=0.13 (<0.15) but recover_frac
    # =13/25=0.52 (>=0.50). The old fall-through would have returned a spurious PROCEED.
    bl, it = _make(75, 0, 13, 12)
    g = evaluate_gate(score(bl, it), {"gate_form": "absolute", "powered_n": 90}, it)
    assert g["lift_meets_threshold"] is False
    assert g["verdict"] == "NO-GO"


def test_score_primary_any_one_with_strict_sensitivity():
    # Baseline misses all 4; iterate recovers ANY interior on all 4 but ALL interior on
    # only 1. Primary (any-one) -> iterate recall 4/4; strict sensitivity -> 1/4.
    bl = [{"trial_id": f"t{i}", "hit": False, "hit_strict": False, "hit_any": False,
           "stratum": "random"} for i in range(4)]
    it = [{"trial_id": f"t{i}", "hit": True, "hit_strict": (i == 0), "hit_any": True,
           "stratum": "random", "variance": "stable", "grounding_violations": 0,
           "llm_calls": 1, "runs": [{"kestrel_calls": 1}]} for i in range(4)]
    s = score(bl, it)
    assert s["primary_metric"] == "any_one"
    assert s["overall"]["iterate_recall"] == 1.0
    assert s["sensitivity"]["metric"] == "strict"
    assert s["sensitivity"]["overall"]["iterate_recall"] == 0.25
