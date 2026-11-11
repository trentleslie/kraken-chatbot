"""Unit 4 tail stages: normalize -> judge (+P2) -> gates -> audit export/ingest -> report.

Wires the Unit 1-3 functions over a run directory produced by ``run_study`` (snapshot.jsonl,
consensus_snapshot.json, pinned_hypotheses.json, captures/). Pure helpers are separated from the
live stages so they can be unit-tested. Every stage writes its output into the run directory.

    python -m kestrel_backend.evals.lit_sources.run_tail <stage> --run-dir <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ...pubmed_client import fetch_abstracts
from . import analyze, judge, sources
from .run_study import RULES_DIR, JsonlCheckpoint, Manifest, _atomic_write

TIER1 = analyze.TIER1 if hasattr(analyze, "TIER1") else ["openalex", "exa", "pubmed", "s2"]
CONSENSUS_TOS = "unknown"  # free-plan connector only; no API terms reviewed for automated use


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def load_snapshots(run_dir: Path) -> list[dict[str, Any]]:
    recs = JsonlCheckpoint(run_dir / "snapshot.jsonl", key_fields=("hyp_id", "source", "arm")).latest_rows()
    cons = run_dir / "consensus_snapshot.json"
    if cons.exists():
        recs += json.loads(cons.read_text())
    return recs


def keyed_snaps(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(r["hyp_id"], r["source"], r["arm"]): r for r in records}


def build_items(records: list[dict[str, Any]], pinned: list[dict[str, Any]],
                resolved_consensus: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """One judge item per (hypothesis, canonical paper), with every source that returned it.

    P2 twins: the corruption control and its clean base each get the base link's landmark injected.
    """
    by_hyp = {p["hyp_id"]: p for p in pinned}
    resolved_consensus = resolved_consensus or {}
    items: dict[str, dict[str, Any]] = {}

    def add(hyp_id: str, paper: dict[str, Any], source: str, arm: str, role: str = "returned") -> None:
        ids = sources.canonical_ids(paper)
        if source == "consensus":
            hit = resolved_consensus.get(paper.get("paper_id") or "")
            if hit:
                ids = {"pmid": hit.get("pmid"), "doi": hit.get("doi"),
                       "title_key": sources.normalize_title(hit.get("title"))}
        key = sources.paper_key(ids)
        if key is None:
            return
        item_id = f"{hyp_id}|{key}"
        it = items.setdefault(item_id, {
            "item_id": item_id, "hyp_id": hyp_id, "paper_key": key, "ids": ids,
            "claim": by_hyp[hyp_id]["hypothesis"]["claim"], "kind": by_hyp[hyp_id]["kind"],
            "title": sources.repair_text(paper.get("title")), "sources": [], "arms": [], "roles": [],
        })
        if source not in it["sources"]:
            it["sources"].append(source)
        if f"{source}/{arm}" not in it["arms"]:
            it["arms"].append(f"{source}/{arm}")
        if role not in it["roles"]:
            it["roles"].append(role)
        if source == "consensus":
            it["consensus_title"] = paper.get("title")
            it["consensus_year"] = paper.get("year")
            it["consensus_first_author"] = paper.get("first_author")
            hit = resolved_consensus.get(paper.get("paper_id") or "")
            if hit:
                it.update(resolved_title=hit.get("title"), resolved_year=hit.get("year"),
                          resolved_first_author=hit.get("first_author"))

    for rec in records:
        if rec["hyp_id"] not in by_hyp:
            continue
        for paper in rec.get("papers") or []:
            add(rec["hyp_id"], paper, rec["source"], rec["arm"])

    for p in pinned:
        if p["kind"] == "control_corrupt":
            for lm in p.get("inject_landmarks") or []:
                paper = {"paper_id": f"PMID:{lm['pmid']}", "doi": lm.get("doi"), "title": lm.get("title")}
                add(p["hyp_id"], paper, "injected", "p2", role="p2_corrupt")
                add(p["twin_of"], paper, "injected", "p2", role="p2_clean")
    return items


def verdict_map(judgments: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict]:
    out = {}
    for j in judgments:
        it = items.get(j["item_id"])
        if it:
            out[(it["hyp_id"], it["paper_key"])] = j
    return out


def annotate_paper_keys(records: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> None:
    """Give every snapshot paper the canonical key its judge item uses (Consensus via resolution)."""
    by_title = {}
    for it in items.values():
        if it.get("consensus_title"):
            by_title[(it["hyp_id"], it["consensus_title"])] = it["paper_key"]
    for rec in records:
        for p in rec.get("papers") or []:
            if rec["source"] == "consensus":
                p["paper_key"] = by_title.get((rec["hyp_id"], p.get("title")))
            else:
                p["paper_key"] = sources.paper_key(sources.canonical_ids(p))


def false_support_events(judgments: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> list[dict]:
    out = []
    for j in judgments:
        it = items.get(j["item_id"])
        if (it and it["kind"] == "control_false" and j.get("status") == "judged" and j.get("hit")
                and j.get("direction") == "supports"):
            out.append({**it, **j})
    return out


def pubmed_pairs(records: list[dict[str, Any]], pinned: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(production-client n, direct-client n) on the identical Arm 1 sentence, real hypotheses."""
    snaps = keyed_snaps(records)
    out = []
    for p in pinned:
        if p["kind"] != analyze.REAL:
            continue
        prod = snaps.get((p["hyp_id"], "pubmed", "arm1"))
        direct = snaps.get((p["hyp_id"], "pubmed", "arm1_direct"))
        if prod and direct and prod["state"] in ("ok", "empty") and direct["state"] in ("ok", "empty"):
            out.append((len(prod["papers"]), len(direct["papers"])))
    return out


# ---------------------------------------------------------------------------------------------
# Live stages
# ---------------------------------------------------------------------------------------------


async def stage_normalize(run_dir: Path) -> dict[str, Any]:  # pragma: no cover - live
    pinned = json.loads((run_dir / "pinned_hypotheses.json").read_text())
    records = load_snapshots(run_dir)

    resolved: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec["source"] != "consensus":
            continue
        for p in rec.get("papers") or []:
            cands = await sources.openalex_title_candidates(p.get("title") or "", n=5)
            hit = sources.resolve_consensus_title(p, cands)
            if hit:
                resolved[p.get("paper_id") or ""] = hit

    items = build_items(records, pinned, resolved)
    pmids = sorted({it["ids"]["pmid"] for it in items.values() if it["ids"].get("pmid")})
    efetched = await fetch_abstracts(pmids) if pmids else {}

    async def efetch(ids):
        return {i: efetched.get(i) for i in ids}

    cache: dict[str, tuple[str | None, str | None]] = {}
    for it in items.values():
        if it["paper_key"] not in cache:
            cons_abs = resolved.get(it.get("paper_key") or "", {}).get("abstract")
            ab, via = await sources.resolve_abstract(it["ids"], title=it.get("resolved_title") or it["title"],
                                                     efetch=efetch, by_doi=sources.openalex_abstract_by_doi,
                                                     by_title=sources.openalex_abstract_by_title)
            cache[it["paper_key"]] = (ab or cons_abs, via or ("openalex_title" if cons_abs else None))
        it["abstract"], it["abstract_via"] = cache[it["paper_key"]]
        it["resolved"] = bool(it["ids"].get("pmid") or it["ids"].get("doi"))
    _atomic_write(run_dir / "items.json", json.dumps(items, indent=1, default=str))
    rows = [{"source": s, "resolved": it["resolved"], "abstract": it["abstract"]}
            for it in items.values() for s in it["sources"]]
    report = sources.normalization_report(rows)
    _atomic_write(run_dir / "normalization_report.json", json.dumps(report, indent=1))
    return {"items": len(items), "consensus_resolved": len(resolved), "report": report}


async def stage_judge(run_dir: Path) -> dict[str, Any]:  # pragma: no cover - live
    items = json.loads((run_dir / "items.json").read_text())
    ck = JsonlCheckpoint(run_dir / "judgments.jsonl", key_fields=("item_id",))
    await judge.judge_items(list(items.values()), judge.default_call_model, checkpoint=ck)
    rows = ck.latest_rows()
    return {s: sum(1 for r in rows if r.get("status") == s) for s in ("judged", "no_abstract", "not_run")}


async def stage_p2(run_dir: Path, runs: int = 3) -> dict[str, Any]:  # pragma: no cover - live
    """Judge the injected landmark for the corrupted link and its clean twin, 3 live runs each."""
    items = json.loads((run_dir / "items.json").read_text())
    p2 = [it for it in items.values() if {"p2_corrupt", "p2_clean"} & set(it["roles"])]
    results = []
    for it in p2:
        role = "p2_corrupt" if "p2_corrupt" in it["roles"] else "p2_clean"
        for k in range(runs):
            rows = await judge.judge_items([{**it, "item_id": f"{it['item_id']}#{k}"}], judge.default_call_model)
            results.append({"role": role, "item_id": it["item_id"], "run": k, **rows[0]})
    corrupt = [r for r in results if r["role"] == "p2_corrupt"]
    clean = [r for r in results if r["role"] == "p2_clean"]
    passed = (len(corrupt) == runs and len(clean) == runs
              and all(r.get("status") == "judged" and not r.get("hit") for r in corrupt)
              and all(r.get("status") == "judged" and r.get("hit") for r in clean))
    out = {"passed": passed, "results": results}
    _atomic_write(run_dir / "p2.json", json.dumps(out, indent=1))
    return out


def stage_gates(run_dir: Path) -> dict[str, Any]:  # pragma: no cover - live
    pinned = json.loads((run_dir / "pinned_hypotheses.json").read_text())
    records = load_snapshots(run_dir)
    recall = analyze.landmark_recall(records, pinned)
    p1 = any((v["found"] or 0) > 0 for v in recall.values())
    zero = sorted(s for s, v in recall.items() if not v["found"])
    p2 = json.loads((run_dir / "p2.json").read_text())["passed"] if (run_dir / "p2.json").exists() else None
    out = {"p1_landmark_recall": recall, "p1_passed": p1, "p1_zero_recall_sources": zero,
           "p2_passed": p2, "halt": (not p1) or (p2 is False)}
    _atomic_write(run_dir / "gates.json", json.dumps(out, indent=1))
    return out


def stage_audit_export(run_dir: Path) -> dict[str, Any]:  # pragma: no cover - live
    rule = json.loads((RULES_DIR / "decision_rule.json").read_text())
    items = json.loads((run_dir / "items.json").read_text())
    judgments = JsonlCheckpoint(run_dir / "judgments.jsonl", key_fields=("item_id",)).latest_rows()
    enriched = [{**items[j["item_id"]], **j} for j in judgments if j["item_id"] in items]
    real = [j for j in enriched if j["kind"] == analyze.REAL]
    fs = false_support_events(judgments, items)
    cc = [it for it in items.values() if it.get("resolved_title")]
    sample = judge.draw_audit_sample(real, false_support_events=fs, consensus_checks=cc, rule=rule["audit"])
    _atomic_write(run_dir / "audit_sample.json", json.dumps(sample, indent=1, default=str))
    packet = judge.export_audit_packet(sample, run_dir / "audit_packet.jsonl", seed=rule["audit"]["seed"])
    return {"packet": str(packet), "readings": len(sample),
            "roles": {r: sum(1 for s in sample if s["audit_role"] == r) for r in
                      ("sample", "false_support", "consensus_identity")}}


async def stage_report(run_dir: Path) -> dict[str, Any]:  # pragma: no cover - live
    rule = json.loads((RULES_DIR / "decision_rule.json").read_text())
    pinned = json.loads((run_dir / "pinned_hypotheses.json").read_text())
    items = json.loads((run_dir / "items.json").read_text())
    judgments = JsonlCheckpoint(run_dir / "judgments.jsonl", key_fields=("item_id",)).latest_rows()
    sample = json.loads((run_dir / "audit_sample.json").read_text())
    labels = judge.ingest_audit_labels(run_dir / "audit_labels.jsonl",
                                       key_path=run_dir / "audit_packet.jsonl.key.json")

    records = load_snapshots(run_dir)
    annotate_paper_keys(records, items)
    snaps = keyed_snaps(records)
    verdicts = verdict_map(judgments, items)

    pairs = [(s["judge_hit"], labels[s["item_id"]], (s.get("sources") or ["?"])[0])
             for s in sample if s["audit_role"] == "sample" and s["item_id"] in labels]
    stats = judge.agreement_stats(pairs)
    certified, reasons = judge.certify(stats, rule["certification"])

    def jf(hyp_id, rec):
        v = verdicts.get((hyp_id, rec.get("paper_key") or analyze.record_paper_key(rec)))
        return bool(v and v.get("status") == "judged" and v.get("hit"))

    mixes = analyze.compare_mixes(pinned, snaps, verdicts, rule, judge_filter=jf)
    real_ids = {p["hyp_id"] for p in pinned if p["kind"] == analyze.REAL}
    real_recs = [r for r in records if r["hyp_id"] in real_ids]
    p_at_k = analyze.precision_at_k(real_recs, verdicts)
    uniq = analyze.unique_hits(real_recs, verdicts)
    recall = analyze.landmark_recall(records, pinned)
    diag = analyze.call_diagnostics(records)
    not_run = sum(1 for j in judgments if j.get("status") == "not_run")

    arm1 = (p_at_k.get(("pubmed", "arm1")) or {}).get("precision")
    arm2 = (p_at_k.get(("pubmed", "arm2")) or {}).get("precision")
    fq_ci = mixes["mixes"]["a_fix_query"]["diff_ci_vs_current"]
    fix_query_met = bool(fq_ci and fq_ci[0] is not None and fq_ci[0] > 0
                         and (arm2 or 0) >= rule["rules"]["fix_query"]["min_arm2_precision"])
    diagnosis = analyze.diagnose_pubmed(pubmed_pairs(records, pinned), arm1, arm2, fix_query_met, rule)

    def src_info(s):
        calls = [r for r in real_recs if r["source"] == s and r["arm"] == "arm1"]
        if calls and all(r["state"] in ("unavailable", "lookup_failed") for r in calls):
            return {"available": False, "reason": f"all {len(calls)} calls {calls[0]['state']}: {calls[0].get('reason')}"}
        cell = p_at_k.get((s, "arm1")) or {"hits": 0, "judged": 0}
        kept_hits = 0
        for name in ("current", "a_fix_query", "c_reweight"):
            mix = name
            for h in real_ids:
                for r in analyze.replay_kept(h, snaps, mix, rule=rule):
                    v = verdicts.get((h, r.get("paper_key") or analyze.record_paper_key(r)))
                    if r.get("source") == s and v and v.get("status") == "judged" and v.get("hit"):
                        kept_hits += 1
        return {"available": True, "zero_hits_all_mixes": kept_hits == 0, "hits": cell["hits"],
                "judged_with_abstract": cell["judged"]}

    cons_calls = [r for r in records if r["source"] == "consensus" and r["hyp_id"] in real_ids]
    cons_avail = bool(cons_calls) and any(r["state"] in ("ok", "empty") for r in cons_calls)
    base_fs = mixes["mixes"]["current"]["false_support"]

    def cons_mapping(name):
        m = mixes["mixes"][name]
        uh = set()
        for h in real_ids:
            cur = {r.get("paper_key") for r in analyze.replay_kept(h, snaps, "current", rule=rule)}
            mapping = name.split("@")[1] if "@" in name else "frozen"
            for r in analyze.replay_kept(h, snaps, "b_add_consensus", rule=rule, consensus_mapping=mapping):
                v = verdicts.get((h, r.get("paper_key")))
                if r.get("source") == "consensus" and r.get("paper_key") not in cur and v and v.get("hit"):
                    uh.add(h)
        return {"unique_hit_hypotheses": len(uh), "false_support_delta": m["false_support"] - base_fs}

    jf_mix = mixes["mixes"].get("current+judge_filter", {})
    inputs = {
        "certified": certified, "not_run": not_run, "pubmed_diagnosis": diagnosis,
        "search_path_n": mixes["search_path_n"],
        "fix_query": {"diff_ci": fq_ci, "arm2_precision": arm2},
        "reweight": {"diff_ci": mixes["mixes"]["c_reweight"]["diff_ci_vs_current"]},
        "sources": {s: src_info(s) for s in TIER1},
        "consensus": {"available": cons_avail, "tos": CONSENSUS_TOS,
                      "reason": None if cons_avail else "no successful connector calls",
                      "mappings": {"frozen": cons_mapping("b_add_consensus"),
                                   "best": cons_mapping("b_add_consensus@best"),
                                   "worst": cons_mapping("b_add_consensus@worst")}},
        "judge_filter": {"precision_ci": jf_mix.get("diff_ci_vs_current"),
                         "false_support_ci": [jf_mix.get("false_support", 0) - base_fs] * 2},
    }
    table = analyze.apply_decision_rule(inputs, rule)
    report = {"certification": {"certified": certified, "reasons": reasons, "stats": stats},
              "mixes": mixes, "precision_at_5": {f"{k[0]}/{k[1]}": v for k, v in p_at_k.items()},
              "unique_hits": uniq, "landmark_recall": recall, "call_diagnostics": diag,
              "pubmed_pairs": pubmed_pairs(records, pinned), "pubmed_diagnosis": diagnosis,
              "not_run": not_run, "inputs": inputs, "table": table}
    _atomic_write(run_dir / "report.json", json.dumps(report, indent=1, default=str))
    _atomic_write(run_dir / "recommendation.md", analyze.render_recommendation_table(table) + "\n")
    return {"certified": certified, "table": table}


def main() -> None:  # pragma: no cover - CLI glue
    ap = argparse.ArgumentParser(description="lit-source spike: Unit 4 tail stages")
    ap.add_argument("stage", choices=["normalize", "judge", "p2", "gates", "audit-export", "report"])
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    manifest = Manifest.load(run_dir)
    fn = {"normalize": stage_normalize, "judge": stage_judge, "p2": stage_p2, "gates": stage_gates,
          "audit-export": stage_audit_export, "report": stage_report}[a.stage]
    out = asyncio.run(fn(run_dir)) if asyncio.iscoroutinefunction(fn) else fn(run_dir)
    manifest.update_stage(a.stage, "complete", **{k: v for k, v in out.items() if isinstance(v, (int, bool, str))})
    print(json.dumps(out, indent=1, default=str)[:4000])
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
