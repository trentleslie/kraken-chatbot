"""Baseline-only pilot: measure R0 (per stratum) and compute the powered N.

Resolves the powered-N circularity (we don't yet have the iterate arm) with a
conservative discordance prior (config.pi_d_prior). Power is estimated by Monte-Carlo
around the EXACT McNemar test we will actually run (statsmodels), not a normal
approximation — honest at small N. The gate FORM (absolute vs relative recall-lift)
is frozen here from R0 (finding: if R0 > r0_relative_switch the absolute form is
unattainable).
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

from .baseline import run_baseline_all
from .config import CONFIG
from .kestrel_rest import KestrelREST


def compute_r0(baseline_records: list[dict]) -> dict:
    """Overall + per-stratum baseline recall (R0)."""
    def rate(recs):
        return (sum(1 for r in recs if r["hit"]) / len(recs)) if recs else 0.0
    strata = sorted({str(r["stratum"]) for r in baseline_records if r.get("stratum")})
    return {
        "overall": rate(baseline_records),
        "by_stratum": {s: rate([r for r in baseline_records if r.get("stratum") == s]) for s in strata},
        "n": len(baseline_records),
    }


def mcnemar_power(n: int, pi_d: float, effect: float, alpha: float, reps: int,
                  rng: np.random.Generator) -> float:
    """Monte-Carlo power of the exact McNemar test to detect a marginal recall
    difference `effect` at discordance rate `pi_d`, for sample size `n`."""
    p10 = (pi_d + effect) / 2.0
    p01 = (pi_d - effect) / 2.0
    if p01 < 0:
        p01, p10 = 0.0, pi_d
    sig = 0
    for _ in range(reps):
        b = int(rng.binomial(n, p01))
        c = int(rng.binomial(n, p10))
        if mcnemar([[0, b], [c, 0]], exact=True).pvalue < alpha:
            sig += 1
    return sig / reps


def powered_n(pi_d: float, effect: float, alpha: float = 0.05, power: float = 0.8,
              reps: int = 400, n_max: int = 400, seed: int = 0) -> int:
    """Smallest N (in steps of 5) reaching `power`; n_max if unreachable."""
    rng = np.random.default_rng(seed)
    for n in range(10, n_max + 1, 5):
        if mcnemar_power(n, pi_d, effect, alpha, reps, rng) >= power:
            return n
    return n_max


def _stratified_subset(items: list[dict], k: int) -> list[dict]:
    strata: dict[str, list[dict]] = {}
    for it in items:
        strata.setdefault(it.get("stratum", "?"), []).append(it)
    out: list[dict] = []
    per = max(1, k // max(1, len(strata)))
    for group in strata.values():
        out.extend(group[:per])
    return out[:k]


async def run_pilot(rest: KestrelREST, items: list[dict], n_subset: int = 15) -> dict:
    subset = _stratified_subset(items, n_subset)
    recs = await run_baseline_all(rest, subset)
    r0 = compute_r0(recs)
    pn = powered_n(CONFIG.pi_d_prior, CONFIG.recall_lift_abs, CONFIG.alpha)
    gate_form = "relative" if r0["overall"] > CONFIG.r0_relative_switch else "absolute"
    n_available = len(items)
    return {
        "r0": r0,
        "powered_n": pn,
        "n_floor": max(CONFIG.n_floor, pn),
        "n_available": n_available,
        "gate_form": gate_form,
        "verdict": "INCONCLUSIVE" if n_available < max(CONFIG.n_floor, pn) else "OK",
        "pilot_kestrel_calls": rest.kestrel_calls,
    }
