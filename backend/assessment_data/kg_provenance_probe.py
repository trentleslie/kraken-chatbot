"""KG-provenance separation probe (Kestrel only — no LLM, no PubMed; near-zero cost).

Validates the bridge-grounding v2 bet: does a JOINT gate — a leg has a CURATED
(knowledge_level=knowledge_assertion/logical_entailment AND agent_type=manual_*) AND
CAUSAL-predicate edge — separate real mechanisms from spurious/association pairs on real
Kestrel edges?

Result (2026-06-17, hand-verified MONDO/CHEBI/NCBITaxon CURIEs): positives 6/6 with a
curated-causal edge, negatives 1/6 (the one being aspirin→ovarian, a defensible curated
chemoprevention signal). This is the OPPOSITE of v1's failure (which inverted, margin −0.333).

Two confirmations:
  1. The JOINT gate is load-bearing. caffeine→pancreatic has a curated `has_adverse_event`
     (associative → curation-alone fails) AND a text-mined `contributes_to` (causal →
     predicate-alone fails); only curated-AND-causal passes → correctly rejected.
  2. Resolution is the binding dependency. A name-resolved variant (resolve_via_api per name)
     failed on wrong-namespace CURIEs (disease→KEGG/PANTHER pathway, VO: invalid, wrong-species
     gene). v2 accuracy requires correct MONDO/HGNC/CHEBI CURIEs → the biomapper resolution fix.
     Set RESOLVE=1 to reproduce the resolution-failure variant.

Usage: PYTHONPATH=. uv run python assessment_data/kg_provenance_probe.py
"""

import asyncio
import collections
import json
import os

from dotenv import load_dotenv

load_dotenv()

from src.kestrel_backend.kestrel_client import call_kestrel_tool  # noqa: E402

CAUSAL = {
    "biolink:causes", "biolink:contributes_to", "biolink:affects", "biolink:regulates",
    "biolink:directly_physically_interacts_with", "biolink:catalyzes", "biolink:treats",
    "biolink:ameliorates_condition", "biolink:disrupts", "biolink:applied_to_treat",
    "biolink:preventative_for_condition",
}
ASSOC = {
    "biolink:associated_with", "biolink:correlated_with", "biolink:gene_associated_with_condition",
    "biolink:genetically_associated_with", "biolink:has_adverse_event", "biolink:biomarker_for",
    "biolink:contraindicated_in", "biolink:in_clinical_trials_for",
    "biolink:treats_or_applied_or_studied_to_treat", "biolink:mentioned_in_clinical_trials_for",
    "biolink:has_phenotype",
}


def pred_class(p):
    return "causal" if p in CAUSAL else ("assoc" if p in ASSOC else "neutral")


def is_curated(kl, agent):
    return kl in ("knowledge_assertion", "logical_entailment") and agent in (
        "manual_agent", "manual_validation_of_automated_agent")


# (label, curie_a, curie_b, gold)  — hand-verified CURIEs (the KG node names are printed to confirm)
PAIRS = [
    ("H.pylori>ulcer", "NCBITaxon:210", "MONDO:0004247", "+"),
    ("metformin>T2D", "CHEBI:6801", "MONDO:0005148", "+"),
    ("levodopa>Parkinson", "CHEBI:15765", "MONDO:0005180", "+"),
    ("imatinib>CML", "CHEBI:45783", "MONDO:0011996", "+"),
    ("donepezil>Alzheimer", "CHEBI:53289", "MONDO:0004975", "+"),
    ("warfarin>thrombosis", "CHEBI:10033", "MONDO:0000831", "+"),
    ("caffeine>pancreatic_ca", "CHEBI:27732", "MONDO:0005192", "-"),
    ("beta-carotene>lung_ca", "CHEBI:17579", "MONDO:0008903", "-"),
    ("vitaminE>CVD", "CHEBI:18145", "MONDO:0004995", "-"),
    ("aspirin>ovarian_ca", "CHEBI:15365", "MONDO:0008170", "-"),
    ("vitaminC>common_cold", "CHEBI:38290", "MONDO:0005709", "-"),
]


async def direct_edges(curie_a, curie_b):
    resp = await call_kestrel_tool(
        "one_hop_query", {"start_node_ids": curie_a, "mode": "full", "limit": 3000})
    content = resp.get("content", []) if isinstance(resp, dict) else []
    if not content:
        return [], {}
    data = json.loads(content[0]["text"])
    nodes = data.get("nodes", {})
    edges = [e for e in data.get("edges", {}).values()
             if isinstance(e, dict) and curie_b in (e.get("subject"), e.get("object"))]
    return edges, nodes


async def main():
    pos = neg = pos_cc = neg_cc = 0
    for label, a, b, gold in PAIRS:
        try:
            edges, nodes = await direct_edges(a, b)
        except Exception as ex:  # invalid prefix / unknown node etc.
            print(f"{gold} {label:24} ERR {str(ex)[:70]}")
            continue
        na = nodes.get(a, {}).get("name", a)[:18]
        nb = nodes.get(b, {}).get("name", b)[:22]
        tally = collections.Counter()
        cc = []
        for e in edges:
            cls = pred_class(e.get("predicate"))
            cur = is_curated(e.get("knowledge_level"), e.get("agent_type"))
            tally[("C" if cur else "T") + "-" + cls] += 1
            if cur and cls == "causal":
                cc.append(e.get("predicate"))
        has_cc = bool(cc)
        if gold == "+":
            pos += 1; pos_cc += has_cc
        else:
            neg += 1; neg_cc += has_cc
        print(f"{gold} {na:18}>{nb:22} cc={'Y' if has_cc else 'n'} "
              f"edges={sum(tally.values())} {dict(tally)}")
    print("-" * 80)
    print(f"JOINT GATE (curated AND causal): positives {pos_cc}/{pos}, negatives {neg_cc}/{neg} "
          f"(want HIGH positives, LOW negatives)")
    if os.getenv("RESOLVE"):
        print("(RESOLVE variant: re-run with resolve_via_api per name to see the resolution "
              "wrong-namespace failures — disease→KEGG/PANTHER pathway, VO: invalid, etc.)")


if __name__ == "__main__":
    asyncio.run(main())
