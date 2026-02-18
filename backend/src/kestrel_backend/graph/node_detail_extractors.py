"""
Per-node detail extractors for pipeline streaming.

Each extractor pulls structured data from the accumulated DiscoveryState
after a node completes, returning a (summary, details_dict) tuple suitable
for PipelineNodeDetailMessage.
"""

from typing import Any


def _safe_model_dump(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def extract_node_details(
    node_name: str,
    accumulated_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    extractor = _EXTRACTORS.get(node_name, _extract_default)
    return extractor(accumulated_state)


def _extract_intake(state: dict) -> tuple[str, dict]:
    entities = state.get("raw_entities", [])
    query_type = state.get("query_type", "unknown")
    is_longitudinal = state.get("is_longitudinal", False)
    aliases = state.get("entity_aliases", {})
    directives = state.get("analytical_directives", [])
    study_ctx = state.get("study_context", {})

    parts = [f"{len(entities)} entities extracted"]
    if is_longitudinal:
        dur = state.get("duration_years")
        parts.append(f"longitudinal ({dur}yr)" if dur else "longitudinal")
    parts.append(f"query_type={query_type}")

    return ", ".join(parts), {
        "entities": entities,
        "query_type": query_type,
        "is_longitudinal": is_longitudinal,
        "duration_years": state.get("duration_years"),
        "aliases_count": len(aliases),
        "study_context": study_ctx,
        "directives": directives,
    }


def _extract_entity_resolution(state: dict) -> tuple[str, dict]:
    resolved = state.get("resolved_entities", [])
    succeeded = [e for e in resolved if getattr(e, "curie", None)]
    failed = [e for e in resolved if not getattr(e, "curie", None)]

    methods: dict[str, int] = {}
    for e in succeeded:
        m = getattr(e, "method", "unknown")
        if m and m.startswith("alias:"):
            m = "alias"
        methods[m] = methods.get(m, 0) + 1

    items = []
    for e in resolved:
        items.append({
            "raw_name": e.raw_name,
            "curie": e.curie,
            "resolved_name": getattr(e, "resolved_name", None),
            "category": getattr(e, "category", None),
            "confidence": getattr(e, "confidence", 0),
            "method": getattr(e, "method", "failed"),
        })

    summary = f"{len(succeeded)}/{len(resolved)} entities resolved"
    if methods:
        method_parts = [f"{v} {k}" for k, v in methods.items()]
        summary += f" ({', '.join(method_parts)})"

    return summary, {
        "resolved": len(succeeded),
        "failed": len(failed),
        "total": len(resolved),
        "methods": methods,
        "entities": items,
    }


def _extract_triage(state: dict) -> tuple[str, dict]:
    scores = state.get("novelty_scores", [])
    well = state.get("well_characterized_curies", [])
    moderate = state.get("moderate_curies", [])
    sparse = state.get("sparse_curies", [])
    cold = state.get("cold_start_curies", [])

    items = []
    for s in scores:
        items.append({
            "curie": s.curie,
            "raw_name": s.raw_name,
            "edge_count": s.edge_count,
            "classification": s.classification,
        })

    summary = (
        f"{len(well)} well-characterized, {len(moderate)} moderate, "
        f"{len(sparse)} sparse, {len(cold)} cold-start"
    )

    return summary, {
        "well_characterized": len(well),
        "moderate": len(moderate),
        "sparse": len(sparse),
        "cold_start": len(cold),
        "entities": items,
        "well_characterized_curies": well,
        "cold_start_curies": cold,
    }


def _extract_direct_kg(state: dict) -> tuple[str, dict]:
    findings = state.get("direct_findings", [])
    diseases = state.get("disease_associations", [])
    pathways = state.get("pathway_memberships", [])
    hubs = state.get("hub_flags", [])

    disease_items = []
    for d in diseases[:20]:
        disease_items.append({
            "entity": d.entity_curie,
            "disease": d.disease_name,
            "disease_curie": d.disease_curie,
            "predicate": d.predicate,
            "evidence": d.evidence_type,
            "preset": getattr(d, "discovery_preset", ""),
            "pmids": len(d.pmids),
        })

    pathway_items = []
    for p in pathways[:15]:
        pathway_items.append({
            "entity": p.entity_curie,
            "pathway": p.pathway_name,
            "pathway_curie": p.pathway_curie,
            "preset": getattr(p, "discovery_preset", ""),
        })

    summary = f"{len(diseases)} disease associations, {len(pathways)} pathways, {len(findings)} findings"
    if hubs:
        summary += f", {len(hubs)} hub flags"

    return summary, {
        "diseases_count": len(diseases),
        "pathways_count": len(pathways),
        "findings_count": len(findings),
        "hub_flags": hubs,
        "top_diseases": disease_items,
        "top_pathways": pathway_items,
    }


def _extract_cold_start(state: dict) -> tuple[str, dict]:
    findings = state.get("cold_start_findings", [])
    analogues = state.get("analogues_found", [])
    inferred = state.get("inferred_associations", [])

    analogue_items = []
    for a in analogues[:10]:
        analogue_items.append({
            "curie": a.curie,
            "name": a.name,
            "similarity": round(a.similarity, 3),
        })

    inferred_items = []
    for i in inferred[:10]:
        inferred_items.append({
            "source": i.source_entity,
            "target": i.target_name,
            "target_curie": i.target_curie,
            "logic": i.logic_chain,
            "confidence": i.confidence,
        })

    summary = f"{len(analogues)} analogues, {len(inferred)} inferred associations, {len(findings)} findings"

    return summary, {
        "analogues_count": len(analogues),
        "inferred_count": len(inferred),
        "findings_count": len(findings),
        "top_analogues": analogue_items,
        "top_inferred": inferred_items,
    }


def _extract_pathway_enrichment(state: dict) -> tuple[str, dict]:
    neighbors = state.get("shared_neighbors", [])
    themes = state.get("biological_themes", [])

    non_hub = [n for n in neighbors if not n.is_hub]

    theme_items = []
    for t in themes:
        theme_items.append({
            "category": t.category,
            "members_count": len(t.members),
            "member_names": t.member_names[:5],
            "input_coverage": t.input_coverage,
        })

    neighbor_items = []
    for n in non_hub[:10]:
        neighbor_items.append({
            "curie": n.curie,
            "name": n.name,
            "category": n.category,
            "degree": n.degree,
            "connected_inputs": n.connected_inputs,
        })

    summary = f"{len(neighbors)} shared neighbors ({len(non_hub)} non-hub), {len(themes)} biological themes"

    return summary, {
        "shared_neighbors_count": len(neighbors),
        "non_hub_count": len(non_hub),
        "themes_count": len(themes),
        "themes": theme_items,
        "top_neighbors": neighbor_items,
    }


def _extract_integration(state: dict) -> tuple[str, dict]:
    bridges = state.get("bridges", [])
    gaps = state.get("gap_entities", [])

    bridge_items = []
    for b in bridges[:10]:
        bridge_items.append({
            "path": b.path_description,
            "entity_names": b.entity_names,
            "tier": b.tier,
            "novelty": b.novelty,
            "significance": b.significance,
        })

    gap_items = []
    for g in gaps[:10]:
        gap_items.append({
            "name": g.name,
            "category": g.category,
            "reason": g.expected_reason,
            "interpretation": g.absence_interpretation,
            "informative": g.is_informative,
        })

    summary = f"{len(bridges)} bridges, {len(gaps)} gap entities"

    return summary, {
        "bridges_count": len(bridges),
        "gaps_count": len(gaps),
        "top_bridges": bridge_items,
        "top_gaps": gap_items,
    }


def _extract_temporal(state: dict) -> tuple[str, dict]:
    classifications = state.get("temporal_classifications", [])

    upstream = [c for c in classifications if c.classification == "upstream_cause"]
    downstream = [c for c in classifications if c.classification == "downstream_consequence"]
    parallel = [c for c in classifications if c.classification == "parallel_effect"]

    items = []
    for c in classifications[:15]:
        items.append({
            "entity": c.entity,
            "finding": c.finding_claim,
            "classification": c.classification,
            "reasoning": c.reasoning,
            "confidence": c.confidence,
        })

    summary = f"{len(upstream)} upstream, {len(downstream)} downstream, {len(parallel)} parallel"

    return summary, {
        "upstream_count": len(upstream),
        "downstream_count": len(downstream),
        "parallel_count": len(parallel),
        "total": len(classifications),
        "classifications": items,
    }


def _extract_synthesis(state: dict) -> tuple[str, dict]:
    report = state.get("synthesis_report", "")
    hypotheses = state.get("hypotheses", [])

    hyp_items = []
    for h in hypotheses[:10]:
        hyp_items.append({
            "title": h.title,
            "tier": h.tier,
            "confidence": h.confidence,
            "claim": h.claim[:200],
        })

    summary = f"{len(hypotheses)} hypotheses generated, {len(report)} char report"

    return summary, {
        "hypotheses_count": len(hypotheses),
        "report_length": len(report),
        "hypotheses": hyp_items,
    }


def _extract_literature_grounding(state: dict) -> tuple[str, dict]:
    """Extract literature grounding details for frontend display."""
    hypotheses = state.get("hypotheses", [])
    errors = state.get("literature_errors", [])

    # Count literature by source
    kg_count = 0
    openalex_count = 0
    exa_count = 0
    s2_count = 0
    total_papers = 0
    grounded_count = 0

    for h in hypotheses:
        if h.literature_support:
            grounded_count += 1
            for lit in h.literature_support:
                total_papers += 1
                if lit.source == "kg":
                    kg_count += 1
                elif lit.source == "openalex":
                    openalex_count += 1
                elif lit.source == "exa":
                    exa_count += 1
                elif lit.source == "s2":
                    s2_count += 1

    # Build summary
    parts = []
    if total_papers:
        parts.append(f"{total_papers} papers")
    if kg_count:
        parts.append(f"{kg_count} from KG")
    if openalex_count:
        parts.append(f"{openalex_count} from OpenAlex")
    if exa_count:
        parts.append(f"{exa_count} from Exa")
    if s2_count:
        parts.append(f"{s2_count} from S2")

    summary = ", ".join(parts) if parts else "No literature found"
    if grounded_count:
        summary += f" across {grounded_count}/{len(hypotheses)} hypotheses"

    # Top papers for display
    top_papers = []
    for h in hypotheses[:5]:
        for lit in h.literature_support[:1]:
            top_papers.append({
                "title": lit.title[:80] + "..." if len(lit.title) > 80 else lit.title,
                "source": lit.source,
                "url": lit.url,
            })

    return summary, {
        "total_papers": total_papers,
        "grounded_hypotheses": grounded_count,
        "total_hypotheses": len(hypotheses),
        "kg_count": kg_count,
        "openalex_count": openalex_count,
        "exa_count": exa_count,
        "s2_count": s2_count,
        "errors_count": len(errors),
        "top_papers": top_papers[:5],
    }


def _extract_default(state: dict) -> tuple[str, dict]:
    return "Node completed", {}


_EXTRACTORS = {
    "intake": _extract_intake,
    "entity_resolution": _extract_entity_resolution,
    "triage": _extract_triage,
    "direct_kg": _extract_direct_kg,
    "cold_start": _extract_cold_start,
    "pathway_enrichment": _extract_pathway_enrichment,
    "integration": _extract_integration,
    "temporal": _extract_temporal,
    "synthesis": _extract_synthesis,
    "literature_grounding": _extract_literature_grounding,
}
