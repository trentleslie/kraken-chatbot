"""Rehydration, mix replay, P6 replay fidelity, diagnostics and the decision rule (Unit 3).

The kept set is what users see, so it is the headline: the production merge (KG-first
short-circuit, ``merge_literature`` priority dedup, relevance sort, cutoff) is replayed over the
one snapshot for each pre-fixed mix. ``harness_merge`` mirrors ``merge_literature`` so mixes can
carry a Consensus source the production ``LiteratureSupport`` model does not admit; its parity
with production is unit-tested, and P6 checks the current mix against the REAL ``run()`` under
cassette replay.
"""

from __future__ import annotations

import random
import statistics
from math import sqrt
from typing import Any, Callable

import httpx
import respx

from ...assessment.cassette import CassetteReplayer
from ...graph.nodes import literature_grounding as lg
from ...graph.pipeline_config import get_pipeline_config
from ...graph.state import DiseaseAssociation, EntityResolution, Finding, Hypothesis
from . import sources

PRIORITY: dict[str, float] = dict(lg.SOURCE_PRIORITY)
PRODUCTION_ORDER = ["openalex", "exa", "pubmed", "s2"]
TIER1 = ["pubmed", "openalex", "exa", "s2"]
MIXES = ["current", "a_fix_query", "b_add_consensus", "c_reweight"]
REAL = "real"


class GateFailure(Exception):
    """A planted-truth or fidelity gate failed: halt before any headline number exists."""


class ReplayMismatch(GateFailure):
    pass


# ---------------------------------------------------------------------------------------------
# Rehydration (captures serialize with model_dump; production code uses attribute access)
# ---------------------------------------------------------------------------------------------

_MODEL_KEYS = {
    "hypotheses": Hypothesis,
    "direct_findings": Finding,
    "cold_start_findings": Finding,
    "disease_associations": DiseaseAssociation,
    "resolved_entities": EntityResolution,
}


def rehydrate_state(serialized: dict[str, Any]) -> dict[str, Any]:
    state = dict(serialized)
    for key, model in _MODEL_KEYS.items():
        if isinstance(state.get(key), list):
            state[key] = [model.model_validate(v) if isinstance(v, dict) else v for v in state[key]]
    return state


def kg_pmid_total(state: dict[str, Any]) -> int:
    return sum(len(v) for v in lg.collect_pmids_from_state(state).values())


def check_kg_gate(live_count: int, state: dict[str, Any]) -> None:
    got = kg_pmid_total(state)
    if got != live_count:
        raise GateFailure(f"KG PMID count from rehydrated state ({got}) != live count at capture "
                          f"({live_count}); the harness would fabricate 'KG yields 0'")


# ---------------------------------------------------------------------------------------------
# Merge replay
# ---------------------------------------------------------------------------------------------


def unique_key(rec: dict[str, Any]) -> str:
    """Mirror of ``literature_grounding.get_unique_key`` over a record dict."""
    if rec.get("doi"):
        return f"doi:{rec['doi'].lower()}"
    pid = str(rec.get("paper_id") or "")
    if pid.startswith("PMID:"):
        return f"pmid:{pid.replace('PMID:', '')}"
    return f"title:{(rec.get('title') or '').lower()[:100].strip()}:{rec.get('year')}"


def harness_merge(records: list[dict[str, Any]], priority: dict[str, float], limit: int) -> list[dict[str, Any]]:
    """Mirror of ``merge_literature`` (group by key in first-seen order, best priority wins with a
    stable sort, stable relevance sort descending, cutoff). Complementary-field merging does not
    change keys or order, so it is not reproduced."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_key.setdefault(unique_key(r), []).append(r)
    merged = [sorted(group, key=lambda p: priority.get(p["source"], 99))[0] for group in by_key.values()]
    merged.sort(key=lambda p: p["relevance_score"], reverse=True)
    return merged[:limit]


def _papers(snaps, hyp_id, source, arm="arm1") -> list[dict[str, Any]]:
    rec = snaps.get((hyp_id, source, arm))
    return list(rec["papers"]) if rec else []


def _consensus_scored(papers: list[dict[str, Any]], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, p in enumerate(papers):
        score = (max(0.0, mapping["base"] - mapping["step"] * i) if mapping["kind"] == "rank_decay"
                 else mapping["value"])
        out.append({**p, "relevance_score": score, "source": "consensus"})
    return out


def _default_rule() -> dict[str, Any]:
    import json
    from .run_study import RULES_DIR

    return json.loads((RULES_DIR / "decision_rule.json").read_text())


def replay_kept(
    hyp_id: str,
    snaps: dict[tuple[str, str, str], dict[str, Any]],
    mix: str,
    *,
    consensus_mapping: str = "frozen",
    judge_filter: Callable[[str, dict[str, Any]], bool] | None = None,
    limit: int | None = None,
    rule: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The kept set a user would see for this hypothesis under ``mix`` (KG-first short-circuit)."""
    rule = rule or _default_rule()
    limit = limit or get_pipeline_config().literature_grounding.papers_per_hypothesis
    keep = (lambda r: judge_filter(hyp_id, r)) if judge_filter else (lambda r: True)

    kg = _papers(snaps, hyp_id, "kg")
    if kg:
        return [r for r in kg if keep(r)][:limit]

    priority = dict(PRIORITY)
    if mix == "current":
        lists = [_papers(snaps, hyp_id, s) for s in PRODUCTION_ORDER]
    elif mix == "a_fix_query":
        lists = [_papers(snaps, hyp_id, s, "arm2" if s == "pubmed" else "arm1") for s in PRODUCTION_ORDER]
    elif mix == "b_add_consensus":
        mapping = rule["consensus_score_mapping"][consensus_mapping]
        priority["consensus"] = mapping["priority"]
        cons = _consensus_scored(_papers(snaps, hyp_id, "consensus", "native"), mapping)
        lists = [_papers(snaps, hyp_id, "openalex"), cons] + [_papers(snaps, hyp_id, s) for s in PRODUCTION_ORDER[1:]]
    elif mix == "c_reweight":
        fetch_limit = rule["mixes"]["c_reweight"]["fetch_limit"]
        ordered = sorted(PRODUCTION_ORDER, key=lambda s: PRIORITY[s])
        lists = [[{**p, "relevance_score": max(0.0, 1 - i / fetch_limit)}
                  for i, p in enumerate(_papers(snaps, hyp_id, s))] for s in ordered]
    else:
        raise ValueError(f"unknown mix {mix}")
    records = [r for lst in lists for r in lst if keep(r)]
    return harness_merge(records, priority, limit)


# ---------------------------------------------------------------------------------------------
# P6: replay fidelity against the real run()
# ---------------------------------------------------------------------------------------------


def build_snapshot_cassette(hyp_id: str, snaps: dict, backfill: list[dict[str, Any]]) -> dict[str, Any]:
    """Cassette (``assessment/cassette.py`` format) of every Arm 1 response + recorded backfill."""
    inters = []
    for (h, _src, arm), rec in snaps.items():
        if h == hyp_id and arm == "arm1":
            inters.extend(rec.get("interactions") or [])
    inters.extend(backfill)
    out = []
    for i in inters:
        if i.get("exception_type"):
            continue  # no response to replay; production would see the same failure only live
        out.append({"request": {"method": i["method"], "url": i["url"], "content": i.get("content")},
                    "response": {"status_code": i["status_code"],
                                 "headers": {"content-type": i.get("content_type") or "application/json"},
                                 "text": i["text"]}})
    return {"metadata": {"hyp_id": hyp_id, "kind": "lit_sources_snapshot"}, "interactions": out}


async def p6_check(state: dict[str, Any], cassette: dict[str, Any],
                   harness_kept: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the REAL ``literature_grounding.run()`` under replay; kept sets must be identical.

    Mirrors ``setup_replay`` but keys requests on the key-redacted URL (snapshots never store
    secrets) and records unmatched requests, because the production clients swallow the error an
    unmatched request raises and would silently return ``[]``.
    """
    replayer = CassetteReplayer(cassette)
    unmatched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        probe = httpx.Request(request.method, sources.redact_url(str(request.url)), content=request.content)
        resp = replayer.get_response(probe)
        if resp is None:
            unmatched.append(f"{request.method} {sources.redact_url(str(request.url))}")
            raise httpx.ConnectError("no cassette match", request=request)
        return resp

    router = respx.MockRouter(assert_all_called=False)
    router.route().mock(side_effect=handler)
    with router:
        out = await lg.run(state)
    grounded = out["hypotheses"]
    prod_keys = [lg.get_unique_key(p) for h in grounded for p in h.literature_support]
    harness_keys = [unique_key(r) for r in harness_kept]
    failed = [e for e in out.get("literature_errors", []) if "grounding failed" in str(e)]
    result = {"identical": prod_keys == harness_keys and not unmatched and not failed,
              "production_keys": prod_keys, "harness_keys": harness_keys,
              "unmatched": unmatched, "grounding_failed": failed}
    if not result["identical"]:
        raise ReplayMismatch(result)
    return result


# ---------------------------------------------------------------------------------------------
# Metrics and diagnostics
# ---------------------------------------------------------------------------------------------


def record_paper_key(rec: dict[str, Any]) -> str | None:
    return rec.get("paper_key") or sources.paper_key(sources.canonical_ids(rec))


def kept_set_precision(kept_by_hyp: dict[str, list[dict[str, Any]]],
                       verdicts: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Hits / judged-with-abstract per hypothesis and pooled; no-abstract and not_run counted."""
    per: dict[str, float | None] = {}
    hits = judged = no_ab = not_run = 0
    for h, kept in kept_by_hyp.items():
        hh = jj = 0
        for r in kept:
            v = verdicts.get((h, record_paper_key(r))) or {"status": "not_run"}
            if v["status"] == "judged":
                jj += 1
                hh += int(bool(v["hit"]))
            elif v["status"] == "no_abstract":
                no_ab += 1
            else:
                not_run += 1
        per[h] = hh / jj if jj else None
        hits += hh
        judged += jj
    return {"pooled": hits / judged if judged else None, "per_hypothesis": per, "hits": hits,
            "judged": judged, "n_no_abstract": no_ab, "n_not_run": not_run}


def precision_at_k(snapshot_records: list[dict[str, Any]], verdicts: dict, k: int = 5) -> dict[tuple[str, str], dict]:
    """Per (source, arm): hits / returned-with-abstract over each call's first k papers."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in snapshot_records:
        cell = out.setdefault((rec["source"], rec["arm"]), {"returned": 0, "judged": 0, "hits": 0})
        for p in rec["papers"][:k]:
            cell["returned"] += 1
            v = verdicts.get((rec["hyp_id"], record_paper_key(p)))
            if v and v["status"] == "judged":
                cell["judged"] += 1
                cell["hits"] += int(bool(v["hit"]))
    for cell in out.values():
        cell["precision"] = cell["hits"] / cell["judged"] if cell["judged"] else None
    return out


def unique_hits(snapshot_records: list[dict[str, Any]], verdicts: dict) -> dict[str, int]:
    """Hits (matched by canonical key) that no other source found for the same hypothesis."""
    found: dict[tuple[str, str], set[str]] = {}
    for rec in snapshot_records:
        for p in rec["papers"]:
            key = record_paper_key(p)
            v = verdicts.get((rec["hyp_id"], key))
            if v and v["status"] == "judged" and v["hit"]:
                found.setdefault((rec["hyp_id"], key), set()).add(rec["source"])
    counts: dict[str, int] = {}
    for srcs in found.values():
        if len(srcs) == 1:
            s = next(iter(srcs))
            counts[s] = counts.get(s, 0) + 1
    return counts


def landmark_recall(snapshot_records: list[dict[str, Any]], pinned: list[dict[str, Any]]) -> dict[str, dict]:
    """ID-matched landmark recall per source on the known-true controls (judge-independent)."""
    trues = {p["hyp_id"]: p for p in pinned if p["kind"] == "control_true"}
    total = sum(len(p["landmarks"]) for p in trues.values())
    out: dict[str, dict[str, Any]] = {}
    for rec in snapshot_records:
        if rec["hyp_id"] not in trues:
            continue
        cell = out.setdefault(rec["source"], {"found": set(), "total": total})
        for lm_i, lm in enumerate(trues[rec["hyp_id"]]["landmarks"]):
            if any(sources.matches_landmark(sources.canonical_ids(p), lm) for p in rec["papers"]):
                cell["found"].add((rec["hyp_id"], lm_i))
    return {s: {"found": len(c["found"]), "total": c["total"],
                "recall": len(c["found"]) / c["total"] if c["total"] else None} for s, c in out.items()}


def false_support_count(kept_by_hyp: dict[str, list[dict[str, Any]]], verdicts: dict,
                        pinned: list[dict[str, Any]]) -> int:
    false_ids = {p["hyp_id"] for p in pinned if p["kind"] == "control_false"}
    n = 0
    for h, kept in kept_by_hyp.items():
        if h not in false_ids:
            continue
        for r in kept:
            v = verdicts.get((h, record_paper_key(r)))
            if v and v["status"] == "judged" and v["hit"] and v["direction"] == "supports":
                n += 1
    return n


def call_diagnostics(snapshot_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Call-state counts and median latency per (source, arm)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in snapshot_records:
        cell = out.setdefault(f"{rec['source']}/{rec['arm']}", {"states": {}, "latencies": []})
        cell["states"][rec["state"]] = cell["states"].get(rec["state"], 0) + 1
        if rec.get("latency_s") is not None:
            cell["latencies"].append(rec["latency_s"])
    for cell in out.values():
        lat = cell.pop("latencies")
        cell["median_latency_s"] = statistics.median(lat) if lat else None
    return out


def headline_frame(rows: list[dict[str, Any]], pinned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only real hypotheses ever enter headline metrics or decision-rule precision inputs."""
    real = {p["hyp_id"] for p in pinned if p["kind"] == REAL}
    return [r for r in rows if r["hyp_id"] in real]


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / denom


def paired_bootstrap_ci(a: list[float | None], b: list[float | None], *, n: int = 2000,
                        seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """CI of mean(b - a) resampling hypotheses (pairs with a missing side are dropped)."""
    diffs = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    lo_i = int((1 - level) / 2 * n)
    hi_i = min(n - 1, int((1 + level) / 2 * n) - 1)
    return (means[lo_i], means[hi_i])


def compare_mixes(pinned: list[dict[str, Any]], snaps: dict, verdicts: dict, rule: dict[str, Any],
                  judge_filter: Callable | None = None) -> dict[str, Any]:
    """Kept-set precision per mix on the real SEARCH-PATH subset, CIs vs current, P3 false support."""
    real = [p for p in pinned if p["kind"] == REAL]
    search = [p["hyp_id"] for p in real if not _papers(snaps, p["hyp_id"], "kg")]
    stats_cfg = rule["statistics"]

    def kept_for(mix, hyp_ids, **kw):
        return {h: replay_kept(h, snaps, mix, rule=rule, **kw) for h in hyp_ids}

    out: dict[str, Any] = {"search_path_n": len(search),
                           "kg_short_circuited_real": len(real) - len(search), "mixes": {}}
    base = kept_set_precision(kept_for("current", search), verdicts)
    false_ids = [p["hyp_id"] for p in pinned if p["kind"] == "control_false"]
    variants = [(m, {}) for m in MIXES] + [("b_add_consensus@best", {"consensus_mapping": "best"}),
                                          ("b_add_consensus@worst", {"consensus_mapping": "worst"})]
    if judge_filter is not None:
        variants.append(("current+judge_filter", {"judge_filter": judge_filter}))
    for name, kw in variants:
        mix = name.split("@")[0].split("+")[0]
        res = kept_set_precision(kept_for(mix, search, **kw), verdicts)
        a = [base["per_hypothesis"][h] for h in search]
        b = [res["per_hypothesis"][h] for h in search]
        out["mixes"][name] = {
            "precision": res["pooled"], "judged": res["judged"], "no_abstract": res["n_no_abstract"],
            "not_run": res["n_not_run"],
            "diff_ci_vs_current": paired_bootstrap_ci(a, b, n=stats_cfg["bootstrap_resamples"],
                                                      seed=stats_cfg["bootstrap_seed"]),
            "false_support": false_support_count(kept_for(mix, false_ids, **kw), verdicts, pinned),
        }
    return out


def diagnose_pubmed(pairs: list[tuple[int, int]], arm1_precision: float | None,
                    arm2_precision: float | None, fix_query_met: bool, rule: dict[str, Any]) -> str:
    """Three-way PubMed outcome. ``pairs`` = (production-client n, direct-client n) per hypothesis
    on the IDENTICAL Arm 1 sentence; only that matched pair can attribute a failure to the client."""
    direct_pos = [(p, d) for p, d in pairs if d > 0]
    share = rule["pubmed_client_failure"]["min_share_of_direct_positive_hypotheses"]
    if direct_pos and sum(1 for p, _ in direct_pos if p == 0) / len(direct_pos) >= share:
        return "client_failure"
    if fix_query_met:
        return "query_failure"
    low = rule["low"]["low_precision_max"]
    if (arm1_precision is None or arm1_precision < low) and (arm2_precision is None or arm2_precision < low):
        return "corpus_ranking_failure"
    return "fine"


# ---------------------------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------------------------

BLOCKED = "no verdict (uncertified)"


def _ci_above_zero(ci) -> bool:
    return ci is not None and ci[0] is not None and ci[0] == ci[0] and ci[0] > 0


def _ci_below_zero(ci) -> bool:
    return ci is not None and ci[1] is not None and ci[1] == ci[1] and ci[1] < 0


def apply_decision_rule(inputs: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """Pure mapping from measured inputs to verdicts, with the frozen precedence."""
    r = rule["rules"]
    blocked = (not inputs["certified"]) or inputs.get("not_run", 0) > 0
    diagnosis = inputs.get("pubmed_diagnosis")
    fq = inputs.get("fix_query") or {}
    fix_query_met = _ci_above_zero(fq.get("diff_ci")) and (fq.get("arm2_precision") or 0) >= r["fix_query"]["min_arm2_precision"]
    reweight_met = _ci_above_zero((inputs.get("reweight") or {}).get("diff_ci"))
    drop_cfg = r["drop"]

    table: dict[str, Any] = {
        "verdicts_blocked": blocked,
        "underpowered": inputs.get("search_path_n", 0) < rule["statistics"]["search_path_min_n"],
        "search_path_n": inputs.get("search_path_n"),
        "pubmed_diagnosis": diagnosis,
        "sources": {},
    }
    for s in TIER1:
        info = inputs["sources"][s]
        if not info.get("available", True):
            table["sources"][s] = {"verdict": "unavailable", "reason": info.get("reason")}
            continue
        if blocked:
            table["sources"][s] = {"verdict": BLOCKED}
            continue
        if s == "pubmed" and diagnosis == "client_failure":
            v = "fix client (#126)"
        elif s == "pubmed" and fix_query_met:
            v = "fix query"
        elif reweight_met:
            v = "reweight"
        elif info["zero_hits_all_mixes"] and info["judged_with_abstract"] >= drop_cfg["min_judged_with_abstract"] \
                and wilson_upper(info["hits"], info["judged_with_abstract"]) < drop_cfg["wilson_upper_max"]:
            v = "drop"
        elif info["zero_hits_all_mixes"]:
            v = "insufficient data"
        else:
            v = "keep"
        table["sources"][s] = {"verdict": v}

    cons = inputs.get("consensus") or {}
    if not cons.get("available", False):
        table["sources"]["consensus"] = {"verdict": "unavailable", "reason": cons.get("reason")}
    elif blocked:
        table["sources"]["consensus"] = {"verdict": BLOCKED}
    else:
        add = r["add"]
        mappings = cons.get("mappings") or {}
        holds = all(
            m in mappings
            and mappings[m]["unique_hit_hypotheses"] >= add["min_distinct_real_hypotheses_with_unique_kept_hits"]
            and mappings[m]["false_support_delta"] <= 0
            for m in ["frozen", *add["must_hold_under_mappings"]]
        )
        ok = cons.get("tos") == add["tos_gate"] and holds
        table["sources"]["consensus"] = {"verdict": "add" if ok else "keep out",
                                         "tos": cons.get("tos")}

    jf = inputs.get("judge_filter") or {}
    if blocked:
        table["judge_filter"] = BLOCKED
    else:
        table["judge_filter"] = "yes" if (_ci_above_zero(jf.get("precision_ci"))
                                          and _ci_below_zero(jf.get("false_support_ci"))) else "no"
    table["kg"] = "kept (product grounds; excluded from drop/reweight)"
    return table


def render_recommendation_table(table: dict[str, Any]) -> str:
    lines = ["| Source | Verdict | Note |", "|---|---|---|"]
    for s, v in table["sources"].items():
        note = v.get("reason") or (f"ToS: {v['tos']}" if v.get("tos") else "")
        lines.append(f"| {s} | {v['verdict']} | {note or ''} |")
    lines.append(f"| kg | {table.get('kg', '')} | |")
    lines.append(f"| Judge filter | {table['judge_filter']} | |")
    flags = []
    if table["verdicts_blocked"]:
        flags.append("Verdicts blocked: judge not certified or a judgment is not_run. Diagnostics only.")
    if table["underpowered"]:
        flags.append(f"Underpowered: search-path subset n={table.get('search_path_n')} (< minimum).")
    if table.get("pubmed_diagnosis"):
        flags.append(f"PubMed diagnosis: {table['pubmed_diagnosis']}.")
    return "\n".join(lines + [""] + [f"- {f}" for f in flags])
