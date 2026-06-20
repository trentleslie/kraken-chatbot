"""Graded v2 scorer probe: does a TIER-WEIGHTED leg score (not a binary both-legs gate) separate
mechanism-endpoint bridges from spurious ones at the CHAIN level?

The binary curated-causal gate was near-inert (only ~5% of real bridges have both legs curated-causal,
kg_bridge_leg_probe). But the signal is real (mech legs 4x spurious). This probe grades each leg by a
provenance tier weight, composes by weaker-leg (min), and tests separation between real 2-hop bridges
of mechanism endpoints vs spurious endpoints (rank-AUC + per-endpoint best bridge). Persists an artifact.

Usage: PYTHONPATH=. uv run python assessment_data/kg_bridge_graded_probe.py
"""

import asyncio
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.kestrel_backend.kestrel_client import call_kestrel_tool  # noqa: E402

CAUSAL = {
    "biolink:causes", "biolink:contributes_to", "biolink:affects", "biolink:regulates",
    "biolink:directly_physically_interacts_with", "biolink:catalyzes", "biolink:disrupts",
    "biolink:ameliorates_condition", "biolink:exacerbates_condition",
}
ASSOC = {
    "biolink:associated_with", "biolink:correlated_with", "biolink:gene_associated_with_condition",
    "biolink:genetically_associated_with", "biolink:has_adverse_event", "biolink:biomarker_for",
    "biolink:treats", "biolink:applied_to_treat", "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:in_clinical_trials_for", "biolink:contraindicated_in", "biolink:has_phenotype",
}
# Tier weight by (curated?, predicate class). Curated-causal dominates; text-mined causal is low
# (SemRep over-asserts causes); curated-neutral (e.g. related_to from MedlinePlus) is mid.
WEIGHT = {
    ("C", "causal"): 1.0, ("C", "neutral"): 0.5, ("C", "assoc"): 0.4,
    ("T", "causal"): 0.35, ("T", "neutral"): 0.2, ("T", "assoc"): 0.15,
}


def pclass(p):
    return "causal" if p in CAUSAL else ("assoc" if p in ASSOC else "neutral")


def curated(kl, ag):
    return kl in ("knowledge_assertion", "logical_entailment") and ag in (
        "manual_agent", "manual_validation_of_automated_agent")


ENDPOINTS = [
    ("metformin~T2D", "CHEBI:6801", "MONDO:0005148", "mech"),
    ("H.pylori~ulcer", "NCBITaxon:210", "MONDO:0004247", "mech"),
    ("imatinib~CML", "CHEBI:45783", "MONDO:0011996", "mech"),
    ("donepezil~Alzheimer", "CHEBI:53289", "MONDO:0004975", "mech"),
    ("levodopa~Parkinson", "CHEBI:15765", "MONDO:0005180", "mech"),
    ("caffeine~pancreatic_ca", "CHEBI:27732", "MONDO:0005192", "ctrl"),
    ("beta-carotene~lung_ca", "CHEBI:17579", "MONDO:0008903", "ctrl"),
    ("vitaminE~CVD", "CHEBI:18145", "MONDO:0004995", "ctrl"),
]


async def bridges_for(a, c):
    resp = await call_kestrel_tool("multi_hop_query", {
        "start_node_ids": [a], "end_node_ids": [c], "max_path_length": 2, "mode": "full", "limit": 200})
    content = resp.get("content", []) if isinstance(resp, dict) else []
    if not content:
        return [], {}
    data = json.loads(content[0]["text"])
    paths = []
    for res in data.get("results", []):
        for p in res.get("paths", []) or []:
            cur = [x for x in p if isinstance(x, str)] if isinstance(p, list) else []
            if len(cur) == 3:
                paths.append(cur)
    return paths[:10], data.get("edges", {})


def leg_weight(a, b, edges):
    """Best (max) tier weight over the leg's candidate edges; 0.0 if none."""
    best = 0.0
    for e in (edges.values() if isinstance(edges, dict) else edges):
        if not isinstance(e, dict):
            continue
        if {e.get("subject"), e.get("object")} == {a, b}:
            tier = "C" if curated(e.get("knowledge_level"), e.get("agent_type")) else "T"
            best = max(best, WEIGHT.get((tier, pclass(e.get("predicate"))), 0.0))
    return best


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


async def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mech_chain, ctrl_chain = [], []
    rows, out = [], {"timestamp": stamp, "weights": {f"{k[0]}-{k[1]}": v for k, v in WEIGHT.items()}}
    print(f"{'kind':5} {'endpoint':24} {'#br':4} {'chain scores (min-leg, best 5)':36} {'best'}")
    print("-" * 100)
    for label, a, c, kind in ENDPOINTS:
        try:
            paths, edges = await bridges_for(a, c)
        except Exception as ex:
            print(f"{kind:5} {label:24} ERR {str(ex)[:40]}"); continue
        scores = []
        for a0, b0, c0 in paths:
            chain = min(leg_weight(a0, b0, edges), leg_weight(b0, c0, edges))  # weaker-leg gating
            scores.append(round(chain, 2))
        (mech_chain if kind == "mech" else ctrl_chain).extend(scores)
        best = max(scores) if scores else 0.0
        rows.append({"label": label, "kind": kind, "n": len(scores), "best": best, "scores": scores})
        top = sorted(scores, reverse=True)[:5]
        print(f"{kind:5} {label:24} {len(scores):<4} {str(top):36} best={best:.2f}")
    print("-" * 100)
    a_bridge = auc(mech_chain, ctrl_chain)
    mech_best = [r["best"] for r in rows if r["kind"] == "mech"]
    ctrl_best = [r["best"] for r in rows if r["kind"] == "ctrl"]
    a_ep = auc(mech_best, ctrl_best)
    print(f"per-BRIDGE chain-score AUC (mech vs ctrl): {a_bridge:.2f}  "
          f"(mech mean {sum(mech_chain)/len(mech_chain):.2f}, ctrl mean {sum(ctrl_chain)/len(ctrl_chain):.2f})"
          if mech_chain and ctrl_chain else "no scores")
    print(f"per-ENDPOINT best-bridge AUC (mech vs ctrl): {a_ep}  mech_best={[round(x,2) for x in mech_best]} ctrl_best={[round(x,2) for x in ctrl_best]}")
    out.update({"rows": rows, "auc_per_bridge": a_bridge, "auc_per_endpoint_best": a_ep,
                "mech_chain": mech_chain, "ctrl_chain": ctrl_chain})
    rundir = Path(__file__).parent / "kg_bridge_leg_runs"; rundir.mkdir(exist_ok=True)
    art = rundir / f"graded_{stamp}.json"; art.write_text(json.dumps(out, indent=2))
    print(f"artifact: {art}")


if __name__ == "__main__":
    asyncio.run(main())
