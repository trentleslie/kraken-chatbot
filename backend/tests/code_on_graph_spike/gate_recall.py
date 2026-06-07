"""Phase-0 recall gate: the kill decision.

Verdict lattice (recall is the kill axis; cost is advisory, not a hard gate — a
single-call baseline vs a multi-turn loop makes a hard cost ratio meaningless):
  hallucinated-CURIE > 0                      -> NO-GO (override; the contract is non-negotiable)
  N < powered-N                               -> INCONCLUSIVE (never a kill below power)
  discordant items flapped across K reruns    -> INCONCLUSIVE (McNemar verdict unstable)
  recall lift below threshold OR McNemar n.s. -> NO-GO (the idea is killed)
  else                                        -> PROCEED-TO-PHASE-1
"""
from __future__ import annotations

from statsmodels.stats.contingency_tables import mcnemar

from .config import CONFIG


def _lift_meets_threshold(tbl: dict, gate_form: str) -> bool:
    abs_lift = tbl["iterate_recall"] - tbl["baseline_recall"]
    if gate_form == "absolute":
        if abs_lift >= CONFIG.recall_lift_abs:
            return True
    # relative form: fraction of the baseline's MISSES that iterate recovered
    static_misses = tbl["c"] + tbl["d"]
    recover_frac = (tbl["c"] / static_misses) if static_misses else 0.0
    return recover_frac >= CONFIG.recall_lift_recover_frac


def evaluate_gate(score_result: dict, pilot_result: dict, iterate_records: list[dict]) -> dict:
    tbl = score_result["overall"]
    a, b, c, d = tbl["a"], tbl["b"], tbl["c"], tbl["d"]
    gate_form = pilot_result.get("gate_form", "absolute")

    mcn = mcnemar(tbl["table"], exact=True)
    significant = (mcn.pvalue < CONFIG.alpha) and (c > b)  # iterate must be the winning direction
    lift_ok = _lift_meets_threshold(tbl, gate_form)

    hallucinated = sum(int(r.get("grounding_violations", 0)) for r in iterate_records)
    powered_n = pilot_result.get("powered_n", CONFIG.n_floor)

    # advisory cost (not a kill): actual loop calls vs baseline (1 call/item)
    loop_kestrel = sum(int(run.get("kestrel_calls", 0)) for r in iterate_records for run in r.get("runs", []))
    loop_llm = sum(int(r.get("llm_calls", 0)) for r in iterate_records)

    if hallucinated > 0:
        verdict = "NO-GO"
    elif tbl["n"] < powered_n:
        verdict = "INCONCLUSIVE"
    elif tbl["discordant_flapped"]:
        verdict = "INCONCLUSIVE"
    elif lift_ok and significant:
        verdict = "PROCEED-TO-PHASE-1"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "gate_form": gate_form,
        "primary_metric": score_result.get("primary_metric"),
        "sensitivity": score_result.get("sensitivity"),
        "baseline_recall": tbl["baseline_recall"],
        "iterate_recall": tbl["iterate_recall"],
        "abs_lift": tbl["iterate_recall"] - tbl["baseline_recall"],
        "mcnemar_p": float(mcn.pvalue),
        "mcnemar_statistic": float(mcn.statistic),
        "significant": significant,
        "lift_meets_threshold": lift_ok,
        "hallucinated_curies": hallucinated,
        "n": tbl["n"],
        "powered_n": powered_n,
        "discordant": {"b_baseline_only": b, "c_iterate_only": c, "concordant_both_hit": a, "concordant_both_miss": d},
        "per_stratum": {s: {"baseline_recall": t["baseline_recall"], "iterate_recall": t["iterate_recall"], "n": t["n"]}
                        for s, t in score_result["by_stratum"].items()},
        "cost_advisory": {"loop_kestrel_calls": loop_kestrel, "loop_llm_calls": loop_llm, "baseline_kestrel_calls": tbl["n"]},
    }
