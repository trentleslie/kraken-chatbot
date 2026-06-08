"""Score the two arms into the paired McNemar table (overall + per stratum).

Cell convention for table [[a, b], [c, d]] (rows = baseline, cols = iterate):
  a = both hit, b = baseline hit & iterate miss, c = baseline miss & iterate hit, d = both miss.
Discordant pairs (b, c) carry all McNemar signal. Recall is under OWA (un-found is not
a false positive). Reported as pooled recall — relative comparison is robust, absolute
is biased up (TREC pooling caveat).
"""
from __future__ import annotations

from .config import CONFIG

_HIT_FIELD = {"any_one": "hit_any", "strict": "hit_strict"}


def build_table(baseline_records: list[dict], iterate_records: list[dict], field: str = "hit") -> dict:
    bl = {r["trial_id"]: bool(r[field]) for r in baseline_records}
    it = {r["trial_id"]: bool(r[field]) for r in iterate_records}
    flap = {r["trial_id"]: (r.get("variance") == "flapping") for r in iterate_records}
    ids = sorted(set(bl) & set(it))
    a = b = c = d = 0
    discordant_flapped = False
    for i in ids:
        if bl[i] and it[i]:
            a += 1
        elif bl[i] and not it[i]:
            b += 1
            discordant_flapped = discordant_flapped or flap.get(i, False)
        elif not bl[i] and it[i]:
            c += 1
            discordant_flapped = discordant_flapped or flap.get(i, False)
        else:
            d += 1
    n = a + b + c + d
    return {
        "table": [[a, b], [c, d]], "n": n, "a": a, "b": b, "c": c, "d": d,
        "baseline_recall": (a + b) / n if n else 0.0,
        "iterate_recall": (a + c) / n if n else 0.0,
        "concordant_miss": d,
        "discordant_flapped": discordant_flapped,
    }


def _tables(baseline_records: list[dict], iterate_records: list[dict], field: str) -> dict:
    overall = build_table(baseline_records, iterate_records, field)
    strata = sorted({str(r["stratum"]) for r in baseline_records if r.get("stratum")})
    by_stratum = {}
    for s in strata:
        bl_s = [r for r in baseline_records if r.get("stratum") == s]
        it_s = [r for r in iterate_records if r.get("stratum") == s]
        by_stratum[str(s)] = build_table(bl_s, it_s, field)
    return {"overall": overall, "by_stratum": by_stratum}


def score(baseline_records: list[dict], iterate_records: list[dict]) -> dict:
    """Score on the configured PRIMARY bridge unit (overall + per stratum), and attach
    the other unit as a reported sensitivity. The verdict (gate_recall) reads `overall`/
    `by_stratum`; `sensitivity` is informational."""
    primary_unit = CONFIG.primary_bridge_unit
    sensitivity_unit = "strict" if primary_unit == "any_one" else "any_one"
    primary = _tables(baseline_records, iterate_records, _HIT_FIELD[primary_unit])
    sens = _tables(baseline_records, iterate_records, _HIT_FIELD[sensitivity_unit])
    return {
        "overall": primary["overall"],
        "by_stratum": primary["by_stratum"],
        "primary_metric": primary_unit,
        "sensitivity": {"metric": sensitivity_unit, "overall": sens["overall"],
                        "by_stratum": sens["by_stratum"]},
    }
