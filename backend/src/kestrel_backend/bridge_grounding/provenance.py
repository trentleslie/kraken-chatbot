"""L1 — bridge evidence-provenance classifier + per-leg labeler (deterministic; no LLM, no score).

For each ordered bridge leg (X→Y), read the KG edges between X and Y and report the **best evidence
tier** from their `knowledge_level` + `agent_type` + Biolink predicate class. The label says *what kind
of evidence* backs the leg; it makes no confidence/mechanism claim. Productionizes the classification
logic validated by `assessment_data/kg_bridge_leg_probe.py`.

Tier ordering (best→worst): curated-causal > curated-associative > curated-neutral > text-mined > none.
"""

import json
import logging
from typing import Any

from ..kestrel_client import call_kestrel_tool

logger = logging.getLogger(__name__)

# Causal-mechanism predicates (Biolink). NOTE: `treats`/`applied_to_treat` are TREATMENT assertions,
# not A→B→C mechanism edges, so they are ASSOCIATIVE here (the v2 finding that counting treats as
# causal inflated coverage on drug legs).
CAUSAL = {
    "biolink:causes", "biolink:contributes_to", "biolink:affects", "biolink:regulates",
    "biolink:directly_physically_interacts_with", "biolink:catalyzes", "biolink:disrupts",
    "biolink:ameliorates_condition", "biolink:exacerbates_condition", "biolink:preventative_for_condition",
}
ASSOCIATIVE = {
    "biolink:associated_with", "biolink:correlated_with", "biolink:gene_associated_with_condition",
    "biolink:genetically_associated_with", "biolink:has_adverse_event", "biolink:biomarker_for",
    "biolink:treats", "biolink:applied_to_treat", "biolink:treats_or_applied_or_studied_to_treat",
    "biolink:in_clinical_trials_for", "biolink:contraindicated_in", "biolink:has_phenotype",
}

_CURATED_KL = ("knowledge_assertion", "logical_entailment")
_CURATED_AGENT = ("manual_agent", "manual_validation_of_automated_agent")

# Best→worst; used to pick the strongest tier across a leg's candidate edges.
TIER_RANK = {
    "curated-causal": 4, "curated-associative": 3, "curated-neutral": 2, "text-mined": 1, "none": 0,
}


def predicate_class(predicate: str | None) -> str:
    """Biolink predicate → 'causal' | 'associative' | 'neutral' (static set, no bmt dependency)."""
    if predicate in CAUSAL:
        return "causal"
    if predicate in ASSOCIATIVE:
        return "associative"
    return "neutral"


def is_curated(knowledge_level: str | None, agent_type: str | None) -> bool:
    """True for a curated/asserted edge: an assertion knowledge level by a manual agent.

    A text-mined `causes` is NOT curated (the failure mode v1/v2 confirmed).
    """
    return knowledge_level in _CURATED_KL and agent_type in _CURATED_AGENT


def evidence_tier(edge: dict[str, Any]) -> str:
    """Evidence tier of a single edge dict: 'curated-<class>' if curated, else 'text-mined'."""
    if is_curated(edge.get("knowledge_level"), edge.get("agent_type")):
        return "curated-" + predicate_class(edge.get("predicate"))
    return "text-mined"


def _parse_full_edges(resp: Any) -> list[dict[str, Any]]:
    """Extract the list of full-mode edge dicts from a one_hop_query response (never raises)."""
    if not isinstance(resp, dict) or resp.get("isError"):
        return []
    content = resp.get("content") or []
    if not content:
        return []
    try:
        data = json.loads(content[0].get("text", ""))
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        return []
    edges = data.get("edges", {}) if isinstance(data, dict) else {}
    values = edges.values() if isinstance(edges, dict) else edges
    return [e for e in values if isinstance(e, dict)]


async def leg_tier(curie_x: str, curie_y: str) -> str:
    """Best evidence tier over the X–Y edges, or 'none' if there is no edge.

    Fetches one_hop_query from X in full mode (the proven probe call — start-only, NOT end_node_ids),
    then filters client-side to edges that also touch Y. Best-effort: returns 'none' on any failure.
    """
    try:
        resp = await call_kestrel_tool(
            "one_hop_query", {"start_node_ids": curie_x, "mode": "full", "limit": 3000})
    except Exception as e:  # best-effort: a Kestrel failure on one leg → 'none'
        logger.warning("bridge provenance: one_hop_query failed for %s: %s", curie_x, e)
        return "none"
    best = "none"
    for edge in _parse_full_edges(resp):
        if curie_y not in (edge.get("subject"), edge.get("object")):
            continue
        tier = evidence_tier(edge)
        if TIER_RANK[tier] > TIER_RANK[best]:
            best = tier
    return best


def bridge_label(leg1_tier: str, leg2_tier: str) -> str:
    """Compose the two legs' tiers into a human-readable chain summary."""
    tiers = (leg1_tier, leg2_tier)
    if leg1_tier == leg2_tier == "none":
        return "no KG edge"
    if "none" in tiers:
        return "one leg unsupported"
    if leg1_tier == leg2_tier == "curated-causal":
        return "both legs curated-causal"
    weaker = min(tiers, key=lambda t: TIER_RANK[t])
    return f"weakest leg {weaker}"
