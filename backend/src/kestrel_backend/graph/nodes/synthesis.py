"""
Synthesis Node: Generate final report from resolved entities and analysis findings.

Phase 5 Update: Transform from data formatter to hypothesis generation engine.

Architecture:
- Phase A: assemble_synthesis_context() - gather all state into context block
- Phase B: LLM query() call for natural language discovery report (if SDK available)
- Fallback: fallback_report() - structured markdown sections (current logic)
- Hypothesis Extraction: extract_hypotheses() - build Hypothesis objects from state

The ~18% validation gap calibration is applied to all Tier 3 hypotheses, based on
systematic review data showing that approximately 18% of computational predictions
progress to clinical investigation.
"""

import logging
import time
from typing import Any
from ..state import (
    DiscoveryState, EntityResolution, NoveltyScore, Finding,
    DiseaseAssociation, PathwayMembership, InferredAssociation, AnalogueEntity,
    SharedNeighbor, BiologicalTheme, Bridge, GapEntity, TemporalClassification,
    Hypothesis
)
from ...literature_utils import format_pmid_link

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Validation gap calibration constant
VALIDATION_GAP_NOTE = "~18% of computational predictions reach clinical investigation (systematic review calibration)"


# =============================================================================
# Synthesis Prompt for LLM-based Report Generation
# =============================================================================

SYNTHESIS_PROMPT = """You are a scientific discovery analyst generating a structured report from KRAKEN knowledge graph analysis.

Your task is to synthesize the accumulated evidence into an actionable discovery report.

## Report Structure

Generate a report with these sections:

### 1. Executive Summary (2-3 sentences)
The most important findings and their implications for the research question.

### 2. Key Findings (Tier 1-2)
Direct evidence from the knowledge graph:
- Disease associations with strong evidence (GWAS, curated databases)
- Validated pathway memberships
- Well-characterized entity relationships

### 3. Novel Predictions (Tier 3)
Speculative associations requiring validation:
- Each prediction MUST include the structural logic chain
- Cite the ~18% validation gap: approximately 18% of computational predictions progress to clinical investigation
- Prioritize by supporting evidence strength

### 4. Biological Themes
Emergent patterns from pathway enrichment:
- Shared biological processes connecting multiple input entities
- Hub-filtered insights (de-emphasize high-connectivity nodes)

### 5. Gap Analysis
Using Open World Assumption framing:
- Expected-but-absent entities and their interpretation
- Informative absences that reveal cohort characteristics

### 6. Temporal Context (if longitudinal study)
- Upstream causes vs downstream consequences
- Causal inference opportunities

### 7. Research Recommendations
Prioritized next steps:
- Experimental validations for high-value predictions
- Literature searches for emerging connections
- Follow-up analyses

## Critical Rules

1. LEAD WITH NOVEL FINDINGS - Don't bury interesting predictions
2. Every Tier 3 must have: logic chain + validation step + ~18% calibration note
3. Clearly distinguish KG facts (Tier 1) from inferences (Tier 3)
4. De-emphasize hub-flagged associations - they may be spurious
5. Highlight FDR-significant entities if present
6. Make the report ACTIONABLE - what should researchers do next?

## Evidence Attribution (REQUIRED)

Every factual claim in the report MUST be tagged with its evidence source:
- [KG Evidence] — finding came from Kestrel knowledge graph query results (direct_findings, disease_associations, pathway data, edge counts)
- [Model Knowledge] — claim is from general biomedical knowledge, not backed by KG query results in this analysis
- [Inferred] — derived by combining KG evidence with model knowledge

If a section has no KG-backed findings, state this explicitly: "No direct KG evidence was found for this connection. The following is based on [Model Knowledge]."

Do NOT present model knowledge as if it were KG-derived. Scientific integrity requires honest attribution.

Generate a clear, scientific report in markdown format.
"""


# =============================================================================
# Existing Format Functions (for fallback report)
# =============================================================================

def format_entity_summary(resolved: list[EntityResolution]) -> str:
    """Format resolved entities into a readable summary."""
    if not resolved:
        return "No entities were provided for resolution."

    successful = [e for e in resolved if e.method != "failed"]
    failed = [e for e in resolved if e.method == "failed"]

    lines = []

    # Header
    lines.append(f"## Entity Resolution Summary")
    lines.append(f"Successfully resolved: {len(successful)}/{len(resolved)} entities\n")

    # Successful resolutions
    if successful:
        lines.append("### Resolved Entities")
        for e in successful:
            category_short = e.category.replace("biolink:", "") if e.category else "Unknown"
            lines.append(f"- **{e.raw_name}** → `{e.curie}`")
            lines.append(f"  - Name: {e.resolved_name}")
            lines.append(f"  - Category: {category_short}")
            lines.append(f"  - Confidence: {e.confidence:.0%} ({e.method})")
            lines.append("")

    # Failed resolutions
    if failed:
        lines.append("### Unresolved Entities")
        for e in failed:
            lines.append(f"- {e.raw_name} (not found in knowledge graph)")
        lines.append("")

    return "\n".join(lines)


def format_novelty_summary(scores: list[NoveltyScore]) -> str:
    """Format novelty scores into a classification summary."""
    if not scores:
        return ""

    lines = ["## Entity Classification\n"]

    # Group by classification
    by_class = {
        "well_characterized": [],
        "moderate": [],
        "sparse": [],
        "cold_start": [],
    }
    for s in scores:
        by_class[s.classification].append(s)

    # Format each group
    if by_class["well_characterized"]:
        lines.append("### Well-Characterized (≥200 edges)")
        for s in by_class["well_characterized"]:
            lines.append(f"- {s.raw_name}: {s.edge_count} edges")
        lines.append("")

    if by_class["moderate"]:
        lines.append("### Moderate Coverage (20-199 edges)")
        for s in by_class["moderate"]:
            lines.append(f"- {s.raw_name}: {s.edge_count} edges")
        lines.append("")

    if by_class["sparse"]:
        lines.append("### Sparse Coverage (1-19 edges)")
        for s in by_class["sparse"]:
            lines.append(f"- {s.raw_name}: {s.edge_count} edges")
        lines.append("")

    if by_class["cold_start"]:
        lines.append("### Cold-Start (0 edges)")
        for s in by_class["cold_start"]:
            lines.append(f"- {s.raw_name}: no KG presence")
        lines.append("")

    return "\n".join(lines)


def format_disease_associations(diseases: list[DiseaseAssociation]) -> str:
    """Format disease associations with evidence types."""
    if not diseases:
        return ""

    lines = ["## Disease Associations\n"]

    # Group by entity
    by_entity: dict[str, list[DiseaseAssociation]] = {}
    for d in diseases:
        if d.entity_curie not in by_entity:
            by_entity[d.entity_curie] = []
        by_entity[d.entity_curie].append(d)

    for entity_curie, entity_diseases in by_entity.items():
        lines.append(f"### {entity_curie}")

        # Group by evidence type
        evidence_order = ["gwas", "curated", "text_mined", "predicted"]
        by_evidence: dict[str, list[DiseaseAssociation]] = {}
        for d in entity_diseases:
            if d.evidence_type not in by_evidence:
                by_evidence[d.evidence_type] = []
            by_evidence[d.evidence_type].append(d)

        for ev_type in evidence_order:
            if ev_type in by_evidence:
                evidence_label = {
                    "gwas": "GWAS Evidence",
                    "curated": "Curated Evidence",
                    "text_mined": "Text-Mined Evidence",
                    "predicted": "Predicted",
                }.get(ev_type, ev_type.title())

                lines.append(f"\n**{evidence_label}:**")
                for d in by_evidence[ev_type]:
                    pmid_str = ""
                    if d.pmids:
                        pmid_links = [format_pmid_link(pmid) for pmid in d.pmids[:3]]
                        pmid_str = f" [{', '.join(pmid_links)}]"
                    lines.append(f"- {d.disease_name} (`{d.disease_curie}`){pmid_str}")
                    lines.append(f"  - Predicate: {d.predicate}")
                    lines.append(f"  - Source: {d.source}")

        lines.append("")

    return "\n".join(lines)


def format_pathway_memberships(pathways: list[PathwayMembership]) -> str:
    """Format pathway memberships."""
    if not pathways:
        return ""

    lines = ["## Pathway & Biological Process Memberships\n"]

    # Group by entity
    by_entity: dict[str, list[PathwayMembership]] = {}
    for p in pathways:
        if p.entity_curie not in by_entity:
            by_entity[p.entity_curie] = []
        by_entity[p.entity_curie].append(p)

    for entity_curie, entity_pathways in by_entity.items():
        lines.append(f"### {entity_curie}")
        for p in entity_pathways:
            lines.append(f"- **{p.pathway_name}** (`{p.pathway_curie}`)")
            lines.append(f"  - Predicate: {p.predicate}")
            lines.append(f"  - Source: {p.source}")
        lines.append("")

    return "\n".join(lines)


def format_shared_neighbors(
    shared_neighbors: list[SharedNeighbor],
    themes: list[BiologicalTheme]
) -> str:
    """Format shared neighbors and biological themes from pathway enrichment."""
    if not shared_neighbors and not themes:
        return ""

    lines = ["## Pathway Enrichment Analysis\n"]
    lines.append("*Shared biological context connecting multiple input entities.*\n")

    # Show themes first (more useful summary)
    if themes:
        lines.append("### Biological Themes")
        for theme in themes[:5]:  # Top 5 themes
            category_short = theme.category.replace("biolink:", "")
            hub_warning = ""
            if theme.top_non_hub is None:
                hub_warning = " (all hubs)"
            
            lines.append(f"\n**{category_short}** — connects {theme.input_coverage} input entities{hub_warning}")
            for i, (curie, name) in enumerate(zip(theme.members[:5], theme.member_names[:5])):
                is_top = "*" if curie == theme.top_non_hub else ""
                lines.append(f"  - {name} (`{curie}`) {is_top}")
            if len(theme.members) > 5:
                lines.append(f"  - ... and {len(theme.members) - 5} more")
        lines.append("")

    # Show individual shared neighbors with details
    if shared_neighbors:
        # Separate hubs from specific neighbors
        specific = [sn for sn in shared_neighbors if not sn.is_hub]
        hubs = [sn for sn in shared_neighbors if sn.is_hub]

        if specific:
            lines.append("### Specific Shared Neighbors (Non-Hub)")
            for sn in sorted(specific, key=lambda x: len(x.connected_inputs), reverse=True)[:10]:
                category_short = sn.category.replace("biolink:", "")
                lines.append(f"- **{sn.name}** (`{sn.curie}`) — {category_short}")
                lines.append(f"  - Connects: {', '.join(sn.connected_inputs[:5])}")
                lines.append(f"  - Degree: {sn.degree} edges")
                if sn.predicates:
                    lines.append(f"  - Predicates: {', '.join(list(set(sn.predicates))[:3])}")
            lines.append("")

        if hubs:
            lines.append("### Hub Nodes (High Connectivity)")
            lines.append("*These nodes have >1000 edges and may represent non-specific associations.*\n")
            for sn in hubs[:5]:
                lines.append(f"- {sn.name} (`{sn.curie}`) — {sn.degree} edges")
            if len(hubs) > 5:
                lines.append(f"- ... and {len(hubs) - 5} more hub nodes")
            lines.append("")

    return "\n".join(lines)


def format_inferred_associations(
    inferences: list[InferredAssociation],
    analogues: list[AnalogueEntity]
) -> str:
    """Format cold-start inferred associations with logic chains."""
    if not inferences and not analogues:
        return ""

    lines = ["## Inferred Associations (Tier 3 - Speculative)\n"]
    lines.append("*These associations are inferred via semantic similarity to well-characterized entities.*")
    lines.append("*All findings require experimental validation.*\n")

    # Show analogues first if any
    if analogues:
        lines.append("### Semantic Analogues Found")
        # Group by similarity score
        sorted_analogues = sorted(analogues, key=lambda a: a.similarity, reverse=True)
        for a in sorted_analogues[:10]:  # Limit to top 10
            category_short = a.category.replace("biolink:", "") if a.category else "Unknown"
            lines.append(f"- **{a.name}** (`{a.curie}`) - Similarity: {a.similarity:.0%} ({category_short})")
        lines.append("")

    # Show inferences
    if inferences:
        lines.append("### Inferred Connections")
        # Group by source entity
        by_entity: dict[str, list[InferredAssociation]] = {}
        for i in inferences:
            if i.source_entity not in by_entity:
                by_entity[i.source_entity] = []
            by_entity[i.source_entity].append(i)

        for entity, entity_inferences in by_entity.items():
            lines.append(f"\n#### {entity}")
            for i in entity_inferences:
                confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(i.confidence, "")
                lines.append(f"\n{confidence_marker} **{i.target_name}** (`{i.target_curie}`)")
                lines.append(f"- Predicate: {i.predicate}")
                lines.append(f"- Supporting analogues: {i.supporting_analogues}")
                lines.append(f"- Logic chain: _{i.logic_chain}_")
                lines.append(f"- **Validation step**: {i.validation_step}")
        lines.append("")

    return "\n".join(lines)


def format_findings_summary(
    direct_findings: list[Finding],
    cold_start_findings: list[Finding]
) -> str:
    """Format analysis findings into a summary report."""
    all_findings = direct_findings + cold_start_findings
    if not all_findings:
        return ""

    lines = ["## Analysis Findings Summary\n"]

    # Sort by tier (1 = high confidence first)
    sorted_findings = sorted(all_findings, key=lambda f: f.tier)

    # Group by tier
    tier_labels = {
        1: "Tier 1 (High Confidence - Direct KG Evidence)",
        2: "Tier 2 (Moderate Confidence - Derived Associations)",
        3: "Tier 3 (Speculative - Semantic Inference)",
    }

    for tier in [1, 2, 3]:
        tier_findings = [f for f in sorted_findings if f.tier == tier]
        if tier_findings:
            lines.append(f"### {tier_labels[tier]}")
            for f in tier_findings:
                source_tag = f"[{f.source}]" if f.source else ""
                confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(f.confidence, "")
                lines.append(f"- {confidence_marker} **{f.entity}**: {f.claim} {source_tag}")
                if f.pmids:
                    pmid_links = [format_pmid_link(pmid) for pmid in f.pmids[:5]]
                    lines.append(f"  - PMIDs: {', '.join(pmid_links)}")
                if f.logic_chain:
                    lines.append(f"  - _Logic: {f.logic_chain}_")
            lines.append("")

    return "\n".join(lines)


def format_hub_warnings(hub_flags: list[str]) -> str:
    """Format warnings about high-degree hub nodes."""
    if not hub_flags:
        return ""

    unique_hubs = list(set(hub_flags))
    lines = ["## Hub Bias Warnings\n"]
    lines.append("*The following entities have very high connectivity (>1000 edges).*")
    lines.append("*Associations involving these entities may be spurious due to hub bias.*\n")

    for hub in unique_hubs:
        lines.append(f"- `{hub}`")
    lines.append("")

    return "\n".join(lines)


def format_bridges(bridges: list[Bridge]) -> str:
    """Format cross-type bridges discovered during integration analysis."""
    if not bridges:
        return ""

    lines = ["## Cross-Type Bridges\n"]
    lines.append("*Multi-hop paths connecting different entity types across the analysis.*\n")

    # Separate by tier
    tier2 = [b for b in bridges if b.tier == 2]
    tier3 = [b for b in bridges if b.tier == 3]

    if tier2:
        lines.append("### High-Confidence Bridges (Tier 2)")
        for b in tier2:
            novelty_tag = "Known" if b.novelty == "known" else "Inferred"
            lines.append(f"\n**{b.path_description}** [{novelty_tag}]")
            if b.entity_names:
                path_with_names = " -> ".join(
                    f"{name} (`{curie}`)"
                    for name, curie in zip(b.entity_names, b.entities)
                )
                lines.append(f"  - Path: {path_with_names}")
            else:
                lines.append(f"  - Entities: {' -> '.join(b.entities)}")
            if b.predicates:
                lines.append(f"  - Predicates: {' -> '.join(b.predicates)}")
            if b.significance:
                lines.append(f"  - **Significance**: {b.significance}")
        lines.append("")

    if tier3:
        lines.append("### Speculative Bridges (Tier 3)")
        for b in tier3:
            novelty_tag = "Known" if b.novelty == "known" else "Inferred"
            lines.append(f"\n**{b.path_description}** [{novelty_tag}]")
            if b.entity_names:
                path_with_names = " -> ".join(
                    f"{name} (`{curie}`)"
                    for name, curie in zip(b.entity_names, b.entities)
                )
                lines.append(f"  - Path: {path_with_names}")
            else:
                lines.append(f"  - Entities: {' -> '.join(b.entities)}")
            if b.significance:
                lines.append(f"  - **Significance**: {b.significance}")
        lines.append("")

    return "\n".join(lines)


def format_gap_entities(gaps: list[GapEntity]) -> str:
    """Format expected-but-absent entities with Open World Assumption framing."""
    if not gaps:
        return ""

    lines = ["## Gap Analysis (Expected-But-Absent Entities)\n"]
    lines.append("*These canonical markers were expected but not found in the analysis.*")
    lines.append("*Using Open World Assumption: absence means 'unstudied', not 'nonexistent'.*\n")

    # Separate informative vs non-informative gaps
    informative = [g for g in gaps if g.is_informative]
    standard = [g for g in gaps if not g.is_informative]

    if informative:
        lines.append("### Informative Absences")
        lines.append("*These absences may reveal unique characteristics of this cohort.*\n")
        for g in informative:
            category_short = g.category.replace("biolink:", "")
            curie_str = f" (`{g.curie}`)" if g.curie else ""
            lines.append(f"- **{g.name}**{curie_str} — {category_short}")
            lines.append(f"  - Expected because: {g.expected_reason}")
            lines.append(f"  - Interpretation: {g.absence_interpretation}")
        lines.append("")

    if standard:
        lines.append("### Standard Gaps")
        for g in standard:
            category_short = g.category.replace("biolink:", "")
            curie_str = f" (`{g.curie}`)" if g.curie else ""
            lines.append(f"- **{g.name}**{curie_str} — {category_short}")
            lines.append(f"  - Expected: {g.expected_reason}")
            lines.append(f"  - Interpretation: {g.absence_interpretation}")
        lines.append("")

    return "\n".join(lines)


def format_temporal_classifications(classifications: list[TemporalClassification]) -> str:
    """Format temporal classifications for longitudinal study findings."""
    if not classifications:
        return ""

    lines = ["## Temporal Analysis\n"]
    lines.append("*Findings classified by temporal relationship to disease progression.*\n")

    # Group by classification
    upstream = [c for c in classifications if c.classification == "upstream_cause"]
    downstream = [c for c in classifications if c.classification == "downstream_consequence"]
    parallel = [c for c in classifications if c.classification == "parallel_effect"]

    if upstream:
        lines.append("### Upstream Causes")
        lines.append("*Metabolic shifts PRECEDING disease manifestation.*\n")
        for c in upstream:
            confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(c.confidence, "")
            lines.append(f"{confidence_marker} **{c.entity}**")
            lines.append(f"  - Finding: {c.finding_claim}")
            lines.append(f"  - Reasoning: _{c.reasoning}_")
        lines.append("")

    if downstream:
        lines.append("### Downstream Consequences")
        lines.append("*Changes RESULTING FROM disease process.*\n")
        for c in downstream:
            confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(c.confidence, "")
            lines.append(f"{confidence_marker} **{c.entity}**")
            lines.append(f"  - Finding: {c.finding_claim}")
            lines.append(f"  - Reasoning: _{c.reasoning}_")
        lines.append("")

    if parallel:
        lines.append("### Parallel Effects")
        lines.append("*Concurrent changes not directly causal.*\n")
        for c in parallel:
            confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(c.confidence, "")
            lines.append(f"{confidence_marker} **{c.entity}**")
            lines.append(f"  - Finding: {c.finding_claim}")
            lines.append(f"  - Reasoning: _{c.reasoning}_")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Phase 5: New Functions
# =============================================================================

def format_study_context(state: DiscoveryState) -> str:
    """Format study-specific context including FDR entities and study metadata."""
    lines = []
    
    fdr_entities = state.get("fdr_entities", [])
    marginal_entities = state.get("marginal_entities", [])
    is_longitudinal = state.get("is_longitudinal", False)
    duration_years = state.get("duration_years")
    
    if not any([fdr_entities, marginal_entities, is_longitudinal]):
        return ""
    
    lines.append("## Study Context\n")
    
    if is_longitudinal:
        duration_str = f" ({duration_years} years)" if duration_years else ""
        lines.append(f"**Study Type**: Longitudinal{duration_str}\n")
    
    if fdr_entities:
        lines.append("### FDR-Significant Entities")
        lines.append("*These entities reached statistical significance after FDR correction.*\n")
        for entity in fdr_entities[:10]:
            lines.append(f"- {entity}")
        if len(fdr_entities) > 10:
            lines.append(f"- ... and {len(fdr_entities) - 10} more")
        lines.append("")
    
    if marginal_entities:
        lines.append("### Marginal Entities")
        lines.append("*These entities showed trends but did not reach FDR significance.*\n")
        for entity in marginal_entities[:10]:
            lines.append(f"- {entity}")
        if len(marginal_entities) > 10:
            lines.append(f"- ... and {len(marginal_entities) - 10} more")
        lines.append("")
    
    return "\n".join(lines)


def assemble_synthesis_context(state: DiscoveryState) -> str:
    """
    Assemble all accumulated state into a context block for LLM synthesis.
    
    This function gathers data from all previous nodes and formats it into
    a comprehensive context that the LLM can use to generate a discovery report.
    """
    sections = []
    
    # Query context
    raw_query = state.get("raw_query", "")
    query_type = state.get("query_type", "unknown")
    sections.append(f"# Analysis Context\n")
    sections.append(f"**Query**: {raw_query}")
    sections.append(f"**Type**: {query_type.title()}\n")
    
    # Study context (FDR entities, longitudinal info)
    study_context = format_study_context(state)
    if study_context:
        sections.append(study_context)
    
    # Entity resolution
    resolved = state.get("resolved_entities", [])
    sections.append(format_entity_summary(resolved))
    
    # Novelty classification
    novelty_scores = state.get("novelty_scores", [])
    novelty_section = format_novelty_summary(novelty_scores)
    if novelty_section:
        sections.append(novelty_section)
    
    # Hub warnings
    hub_flags = state.get("hub_flags", [])
    hub_section = format_hub_warnings(hub_flags)
    if hub_section:
        sections.append(hub_section)
    
    # Disease associations
    disease_associations = state.get("disease_associations", [])
    disease_section = format_disease_associations(disease_associations)
    if disease_section:
        sections.append(disease_section)
    
    # Pathway memberships
    pathway_memberships = state.get("pathway_memberships", [])
    pathway_section = format_pathway_memberships(pathway_memberships)
    if pathway_section:
        sections.append(pathway_section)
    
    # Pathway enrichment (shared neighbors and themes)
    shared_neighbors = state.get("shared_neighbors", [])
    biological_themes = state.get("biological_themes", [])
    enrichment_section = format_shared_neighbors(shared_neighbors, biological_themes)
    if enrichment_section:
        sections.append(enrichment_section)
    
    # Cross-type bridges
    bridges = state.get("bridges", [])
    bridges_section = format_bridges(bridges)
    if bridges_section:
        sections.append(bridges_section)
    
    # Gap analysis
    gap_entities = state.get("gap_entities", [])
    gaps_section = format_gap_entities(gap_entities)
    if gaps_section:
        sections.append(gaps_section)
    
    # Temporal classifications
    temporal_classifications = state.get("temporal_classifications", [])
    temporal_section = format_temporal_classifications(temporal_classifications)
    if temporal_section:
        sections.append(temporal_section)
    
    # Inferred associations
    inferred_associations = state.get("inferred_associations", [])
    analogues_found = state.get("analogues_found", [])
    inference_section = format_inferred_associations(inferred_associations, analogues_found)
    if inference_section:
        sections.append(inference_section)
    
    # All findings
    direct_findings = state.get("direct_findings", [])
    cold_start_findings = state.get("cold_start_findings", [])
    findings_section = format_findings_summary(direct_findings, cold_start_findings)
    if findings_section:
        sections.append(findings_section)
    
    return "\n".join(sections)


def fallback_report(state: DiscoveryState) -> str:
    """
    Generate a structured markdown report without LLM synthesis.
    
    This is the fallback used when the Claude Agent SDK is not available
    or when running tests. It preserves the existing report format.
    """
    resolved = state.get("resolved_entities", [])
    query_type = state.get("query_type", "unknown")
    raw_query = state.get("raw_query", "")
    is_longitudinal = state.get("is_longitudinal", False)
    novelty_scores = state.get("novelty_scores", [])
    direct_findings = state.get("direct_findings", [])
    cold_start_findings = state.get("cold_start_findings", [])
    disease_associations = state.get("disease_associations", [])
    pathway_memberships = state.get("pathway_memberships", [])
    inferred_associations = state.get("inferred_associations", [])
    analogues_found = state.get("analogues_found", [])
    hub_flags = state.get("hub_flags", [])
    shared_neighbors = state.get("shared_neighbors", [])
    biological_themes = state.get("biological_themes", [])
    bridges = state.get("bridges", [])
    gap_entities = state.get("gap_entities", [])
    temporal_classifications = state.get("temporal_classifications", [])

    # Build report sections
    report_lines = []

    # Header
    report_lines.append(f"# KRAKEN Analysis Report")
    report_lines.append(f"**Query Type**: {query_type.title()}")
    if is_longitudinal:
        duration = state.get("duration_years")
        duration_str = f" ({duration} years)" if duration else ""
        report_lines.append(f"**Study Type**: Longitudinal{duration_str}")
    report_lines.append(f"**Original Query**: {raw_query[:100]}{'...' if len(raw_query) > 100 else ''}\n")

    # Study context (FDR entities, etc.)
    study_context = format_study_context(state)
    if study_context:
        report_lines.append(study_context)

    # Entity resolution
    report_lines.append(format_entity_summary(resolved))

    # Novelty classification
    novelty_section = format_novelty_summary(novelty_scores)
    if novelty_section:
        report_lines.append(novelty_section)

    # Hub bias warnings (show early so users are aware)
    hub_section = format_hub_warnings(hub_flags)
    if hub_section:
        report_lines.append(hub_section)

    # Pathway enrichment (shared neighbors and themes) - Phase 4a
    enrichment_section = format_shared_neighbors(shared_neighbors, biological_themes)
    if enrichment_section:
        report_lines.append(enrichment_section)

    # Cross-type bridges - Phase 4b
    bridges_section = format_bridges(bridges)
    if bridges_section:
        report_lines.append(bridges_section)

    # Gap analysis - Phase 4b
    gaps_section = format_gap_entities(gap_entities)
    if gaps_section:
        report_lines.append(gaps_section)

    # Temporal classifications - Phase 4b (only for longitudinal studies)
    temporal_section = format_temporal_classifications(temporal_classifications)
    if temporal_section:
        report_lines.append(temporal_section)

    # Disease associations (structured data)
    disease_section = format_disease_associations(disease_associations)
    if disease_section:
        report_lines.append(disease_section)

    # Pathway memberships (structured data)
    pathway_section = format_pathway_memberships(pathway_memberships)
    if pathway_section:
        report_lines.append(pathway_section)

    # Inferred associations from cold-start (structured data)
    inference_section = format_inferred_associations(inferred_associations, analogues_found)
    if inference_section:
        report_lines.append(inference_section)

    # Analysis findings (summary from both branches)
    findings_section = format_findings_summary(direct_findings, cold_start_findings)
    if findings_section:
        report_lines.append(findings_section)

    # Summary stats
    total_findings = len(direct_findings) + len(cold_start_findings)
    total_diseases = len(disease_associations)
    total_pathways = len(pathway_memberships)
    total_inferences = len(inferred_associations)
    total_shared = len(shared_neighbors)
    total_themes = len(biological_themes)
    total_bridges = len(bridges)
    total_gaps = len(gap_entities)
    total_temporal = len(temporal_classifications)

    if any([total_findings, total_diseases, total_pathways, total_inferences,
            total_shared, total_themes, total_bridges, total_gaps, total_temporal]):
        report_lines.append("---")
        stats = []
        if total_findings > 0:
            stats.append(f"{total_findings} findings")
        if total_diseases > 0:
            stats.append(f"{total_diseases} disease associations")
        if total_pathways > 0:
            stats.append(f"{total_pathways} pathway memberships")
        if total_shared > 0:
            stats.append(f"{total_shared} shared neighbors")
        if total_themes > 0:
            stats.append(f"{total_themes} biological themes")
        if total_bridges > 0:
            stats.append(f"{total_bridges} cross-type bridges")
        if total_gaps > 0:
            stats.append(f"{total_gaps} gap entities")
        if total_temporal > 0:
            stats.append(f"{total_temporal} temporal classifications")
        if total_inferences > 0:
            stats.append(f"{total_inferences} inferred associations")
        report_lines.append(f"*Generated: {', '.join(stats)}*")

    # Next steps hint (only if we have resolved entities but no findings)
    successful = [e for e in resolved if e.method != "failed"]
    if successful and not findings_section:
        report_lines.append("---")
        report_lines.append("*Ready for further analysis. The full workflow will explore ")
        report_lines.append("relationships, pathways, and generate hypotheses for these entities.*")

    # Errors section if any
    errors = state.get("errors", [])
    if errors:
        report_lines.append("\n### Warnings & Errors")
        for error in errors[:10]:  # Limit to first 10 errors
            report_lines.append(f"- {error}")
        if len(errors) > 10:
            report_lines.append(f"- ... and {len(errors) - 10} more warnings")

    return "\n".join(report_lines)


def extract_hypotheses(state: DiscoveryState) -> list[Hypothesis]:
    """
    Extract structured Hypothesis objects from accumulated state.
    
    Builds hypotheses programmatically from:
    - cold_start_findings (Tier 3 inferences)
    - bridges (Tier 2-3 cross-type connections)
    
    All hypotheses include the ~18% validation gap note.
    """
    hypotheses: list[Hypothesis] = []
    
    # From cold-start findings (Tier 3 inferences)
    cold_start_findings = state.get("cold_start_findings", [])
    for finding in cold_start_findings:
        # Skip placeholder or pending findings
        if not finding.claim or "pending" in finding.claim.lower():
            continue
        
        # Only process Tier 3 findings (speculative)
        if finding.tier == 3:
            hypotheses.append(Hypothesis(
                title=f"Inferred role of {finding.entity}",
                tier=3,
                claim=finding.claim,
                supporting_entities=[finding.entity],
                contradicting_entities=[],
                structural_logic=finding.logic_chain or "Based on analogue inference",
                confidence=finding.confidence,
                validation_steps=[
                    f"Search literature for {finding.entity} associations",
                    f"Validate in independent cohort",
                ],
                validation_gap_note=VALIDATION_GAP_NOTE,
            ))
    
    # From cross-type bridges
    bridges = state.get("bridges", [])
    for bridge in bridges:
        if not isinstance(bridge, Bridge):
            continue
        
        # Only create hypothesis if bridge has significance
        if not bridge.significance:
            continue
        
        # Build the logic chain from path
        if bridge.entity_names and bridge.predicates:
            logic = f"{' -> '.join(bridge.entity_names)} via {', '.join(bridge.predicates)}"
        else:
            logic = bridge.path_description
        
        # Get target entity name for validation step
        target_name = bridge.entity_names[-1] if bridge.entity_names else "target"
        
        hypotheses.append(Hypothesis(
            title=f"Bridge: {bridge.path_description}",
            tier=bridge.tier,
            claim=bridge.significance,
            supporting_entities=bridge.entities,
            contradicting_entities=[],
            structural_logic=logic,
            confidence="moderate",
            validation_steps=[
                f"Verify path in literature",
                f"Check {target_name} in GWAS Catalog",
            ],
            validation_gap_note=VALIDATION_GAP_NOTE,
        ))
    
    # From inferred associations (cold-start analogues)
    inferred_associations = state.get("inferred_associations", [])
    for inference in inferred_associations:
        if not isinstance(inference, InferredAssociation):
            continue
        
        hypotheses.append(Hypothesis(
            title=f"Inferred: {inference.source_entity} -> {inference.target_name}",
            tier=3,
            claim=f"{inference.source_entity} may be associated with {inference.target_name} via {inference.predicate}",
            supporting_entities=[inference.source_entity, inference.target_curie],
            contradicting_entities=[],
            structural_logic=inference.logic_chain,
            confidence=inference.confidence,
            validation_steps=[inference.validation_step],
            validation_gap_note=VALIDATION_GAP_NOTE,
        ))
    
    return hypotheses


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Generate a synthesis report and extract hypotheses from all analysis phases.

    Phase 5 architecture:
    - Phase A: Assemble all state into context block
    - Phase B: LLM synthesis (if SDK available) or fallback report
    - Hypothesis extraction: Build structured Hypothesis objects from state

    Returns:
        synthesis_report: Formatted markdown report
        hypotheses: List of Hypothesis objects with validation steps and gap calibration
    """
    # Count input findings
    direct_findings = state.get("direct_findings", [])
    cold_start_findings = state.get("cold_start_findings", [])
    total_findings = len(direct_findings) + len(cold_start_findings)
    logger.info("Starting synthesis with %d findings", total_findings)
    start = time.time()

    # Phase A: Assemble context (always done)
    context = assemble_synthesis_context(state)
    
    # Phase B: LLM synthesis or fallback
    if HAS_SDK:
        try:
            options = ClaudeAgentOptions(
                system_prompt=SYNTHESIS_PROMPT,
                allowed_tools=[],  # No tools - pure reasoning
                max_turns=1,
                permission_mode="bypassPermissions",
            )
            
            result_text = []
            async for event in query(prompt=context, options=options):
                if hasattr(event, 'content'):
                    for block in event.content:
                        if hasattr(block, 'text'):
                            result_text.append(block.text)
            
            report = "\n".join(result_text) if result_text else fallback_report(state)
        except Exception as e:
            # Fallback on any SDK error
            report = fallback_report(state)
            # Could log error here if needed
    else:
        report = fallback_report(state)
    
    # Extract structured hypotheses (always, from state not LLM output)
    hypotheses = extract_hypotheses(state)

    duration = time.time() - start
    logger.info(
        "Completed synthesis in %.1fs — hypotheses=%d, report_length=%d",
        duration, len(hypotheses), len(report)
    )

    return {
        "synthesis_report": report,
        "hypotheses": hypotheses,
    }
