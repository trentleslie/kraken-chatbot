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
        bl.append({"trial_id": tid, "hit": bl_hit, "stratum": stratum})
        it.append({"trial_id": tid, "hit": it_hit, "stratum": stratum,
                   "variance": "flapping" if flap else "stable",
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


def test_gate_hallucinated_curie_is_nogo_override():
    bl, it = _make(5, 1, 12, 2)
    it[0]["grounding_violations"] = 1
    g = evaluate_gate(score(bl, it), PILOT_OK, it)
    assert g["verdict"] == "NO-GO" and g["hallucinated_curies"] == 1


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
