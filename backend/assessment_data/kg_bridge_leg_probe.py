"""Stronger v2 validation: does the curated-causal gate work on REAL bridge LEGS (A->B->C)?

The earlier kg_provenance_probe tested DIRECT A->C edges — but the scorer grades 3-node chains
whose legs (A->B, B->C) run through a Kestrel-DISCOVERED middle node B. This probe fetches real
2-hop bridges (multi_hop_query max_path_length=2, mode=full) and, for each leg edge, classifies
(curated knowledge_level/agent_type AND causal predicate). It answers the adversarial review's
killer question: what fraction of real discovered-bridge legs are curated-causal? If low, v2
returns `insufficient` for most bridges → near-inert on exactly the novel cases the pipeline is for.

Persists a timestamped JSON artifact by default (expensive/live-run SOP).
Usage: PYTHONPATH=. uv run python assessment_data/kg_bridge_leg_probe.py
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
    "biolink:directly_physically_interacts_with", "biolink:catalyzes",
    "biolink:disrupts", "biolink:ameliorates_condition", "biolink:exacerbates_condition",
}
# NOTE: per the review, treats/applied_to_treat are TREATMENT assertions, not A->B->C mechanism
# edges — deliberately EXCLUDED from causal here so we don't inflate coverage on drug legs.
ASSOC = {
    "biolink:associated_with", "biolink:correlated_with", "biolink:gene_associated_with_condition",
    "biolink:genetically_associated_with", "biolink:has_adverse_event", "biolink:biomarker_for",
    "biolink:treats", "biolink:applied_to_treat", "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:in_clinical_trials_for", "biolink:contraindicated_in", "biolink:has_phenotype",
}


def pred_class(p):
    return "causal" if p in CAUSAL else ("assoc" if p in ASSOC else "neutral")


def is_curated(kl, agent):
    return kl in ("knowledge_assertion", "logical_entailment") and agent in (
        "manual_agent", "manual_validation_of_automated_agent")


# (label, curie_A, curie_C, kind)  kind: 'mech' = A~C is an established mechanism; 'ctrl' = spurious/assoc
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
MAX_BRIDGES_PER_PAIR = 10


def edge_prov(e):
    """(predicate, knowledge_level, agent_type) from a full-mode edge dict OR a schema tuple."""
    if isinstance(e, dict):
        return e.get("predicate"), e.get("knowledge_level"), e.get("agent_type"), e.get("subject"), e.get("object")
    return None  # tuple form handled separately if needed


async def bridges_for(a, c):
    resp = await call_kestrel_tool("multi_hop_query", {
        "start_node_ids": [a], "end_node_ids": [c], "max_path_length": 2, "mode": "full", "limit": 200})
    content = resp.get("content", []) if isinstance(resp, dict) else []
    if not content:
        return [], {}, str(resp)[:120]
    data = json.loads(content[0]["text"])
    edges = data.get("edges", {})
    paths = []
    for res in data.get("results", []):
        for p in res.get("paths", []) or []:
            if isinstance(p, list) and len([x for x in p if isinstance(x, str)]) == 3:
                paths.append([x for x in p if isinstance(x, str)])
    return paths[:MAX_BRIDGES_PER_PAIR], edges, None


def leg_edges(a, b, edges):
    out = []
    for e in (edges.values() if isinstance(edges, dict) else edges):
        prov = edge_prov(e)
        if prov and prov[3] in (a, b) and prov[4] in (a, b) and prov[3] != prov[4]:
            out.append(prov)
    return out


def leg_curated_causal(a, b, edges):
    for pred, kl, ag, s, o in leg_edges(a, b, edges):
        if is_curated(kl, ag) and pred_class(pred) == "causal":
            return True
    return False


async def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"timestamp": stamp, "max_path_length": 2, "endpoints": [], "all_legs": collections.Counter()}
    total_legs = cc_legs = 0
    print(f"{'kind':5} {'endpoint':24} {'#bridges':9} {'legs curated-causal':20} bridges both-legs-CC")
    print("-" * 100)
    for label, a, c, kind in ENDPOINTS:
        try:
            paths, edges, err = await bridges_for(a, c)
        except Exception as ex:
            print(f"{kind:5} {label:24} ERR {str(ex)[:50]}")
            continue
        if err:
            print(f"{kind:5} {label:24} no-paths ({err[:40]})")
            out["endpoints"].append({"label": label, "kind": kind, "bridges": 0})
            continue
        per_leg_cc = 0; per_leg_tot = 0; both_cc = 0
        leg_classes = collections.Counter()
        for path in paths:
            a0, b0, c0 = path
            l1 = leg_curated_causal(a0, b0, edges)
            l2 = leg_curated_causal(b0, c0, edges)
            per_leg_tot += 2; per_leg_cc += int(l1) + int(l2)
            both_cc += int(l1 and l2)
            for (x, y) in ((a0, b0), (b0, c0)):
                for pred, kl, ag, s, o in leg_edges(x, y, edges):
                    leg_classes[("C" if is_curated(kl, ag) else "T") + "-" + pred_class(pred)] += 1
        total_legs += per_leg_tot; cc_legs += per_leg_cc
        out["all_legs"].update(leg_classes)
        out["endpoints"].append({"label": label, "kind": kind, "bridges": len(paths),
                                 "leg_cc": per_leg_cc, "leg_total": per_leg_tot, "both_legs_cc": both_cc})
        rate = f"{per_leg_cc}/{per_leg_tot}" if per_leg_tot else "0/0"
        print(f"{kind:5} {label:24} {len(paths):<9} {rate:20} {both_cc} bridges with both legs CC")
    print("-" * 100)
    overall = f"{cc_legs}/{total_legs}" if total_legs else "0/0"
    pct = (100 * cc_legs / total_legs) if total_legs else 0
    print(f"OVERALL curated-causal LEG coverage on real discovered bridges: {overall} ({pct:.0f}%)")
    print(f"leg-class distribution: {dict(out['all_legs'])}")
    out["all_legs"] = dict(out["all_legs"])
    out["overall_curated_causal_legs"] = overall
    rundir = Path(__file__).parent / "kg_bridge_leg_runs"
    rundir.mkdir(exist_ok=True)
    art = rundir / f"{stamp}.json"
    art.write_text(json.dumps(out, indent=2))
    print(f"artifact: {art}")


if __name__ == "__main__":
    asyncio.run(main())
