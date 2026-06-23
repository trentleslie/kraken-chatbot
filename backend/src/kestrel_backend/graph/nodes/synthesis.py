"""
Synthesis Node: Generate final report from resolved entities and analysis findings.

Phase 5 Update: Transform from data formatter to hypothesis generation engine.

Architecture:
- Phase A: assemble_synthesis_context() - gather all state into context block
- Phase B: LLM query() call for natural language discovery report (if SDK available)
- Fallback: fallback_report() - structured markdown sections (current logic)

Hypotheses are no longer produced here: hypothesis_extraction.run() builds them upstream
and literature_grounding.run() grounds them, both before synthesis. Synthesis reads the
already-validated `bridges` and grounded `hypotheses` from state (Unit 2 of the
ground-before-synthesis reorg).
"""

import logging
import time
from typing import Any
from ..state import (
    DiscoveryState, EntityResolution, NoveltyScore, Finding,
    DiseaseAssociation, PathwayMembership, InferredAssociation, AnalogueEntity,
    SharedNeighbor, BiologicalTheme, Bridge, GapEntity, TemporalClassification,
    Hypothesis,
)
from ...literature_utils import format_pmid_link
from ..pipeline_config import get_pipeline_config
from ..sdk_utils import HAS_SDK, ClaudeAgentOptions, query_with_usage
from ..state_contracts import validate_state, SynthesisInput, SynthesisOutput
from ...writing_style import RESEARCH_REGISTER
# References-table assembly lives with grounding's module but is OWNED by synthesis now (R6):
# synthesis appends it from the grounded hypotheses in state. (literature_grounding does not
# import synthesis, so this one-way import introduces no cycle.)
from .literature_grounding import build_references_table

logger = logging.getLogger(__name__)


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
- [Literature] — claim is supported by a grounded abstract/passage in the "Literature Evidence" section below (cite it: author/year or title)
- [Model Knowledge] — claim is from general biomedical knowledge, not backed by KG query results or grounded literature in this analysis
- [Inferred] — derived by combining KG evidence, grounded literature, and/or model knowledge

When a hypothesis has grounded literature in the "Literature Evidence" section, cite that evidence with a
[Literature] tag in its structural-logic / logic-chain narrative — do not leave grounded evidence unused.
Only tag [Literature] when the cited abstract genuinely supports the claim; a tangential paper is not support
(say so rather than over-claiming). The presence of grounded abstracts means literature was *fetched* for that
hypothesis, NOT that the claim is verified — calibrate confidence on the evidence content, not its mere presence.

If a section has no KG-backed findings, state this explicitly: "No direct KG evidence was found for this connection. The following is based on [Model Knowledge]."

Do NOT present model knowledge as if it were KG-derived or literature-backed. Scientific integrity requires honest attribution.

## Module-Level Reasoning (multi-entity input)

If the input is a module (many entities analyzed together — you will see "Module-Level Disease
Recurrence", "Module-Level Pathway Recurrence", and a "Member Prioritization Table" instead of
per-entity dumps), treat the entities as a coordinated group, not a list:
- LEAD with the unifying biological theme that explains why these entities co-vary as a module.
- Build the Key Findings from the Module-Level Recurrence sections (diseases/pathways shared across
  members) and use the Member Prioritization Table to call out the highest-leverage individual members.
- Do not enumerate every member; synthesize the module's story, then highlight outliers.
For a single entity (per-entity sections present, no module sections), report as usual.

Generate a clear, scientific report in markdown format.
"""

# Emit the report's prose in the canonical research register. Appended (not inlined)
# so the register stays single-sourced in writing_style.py. The register self-declares
# it is subordinate to the structure / evidence-tag rules above, so it shapes voice
# without overriding the report contract. Budget: ~480 chars over the ~100K-token
# headroom that SynthesisConfig.max_context_chars leaves — negligible.
SYNTHESIS_PROMPT += f"\n\n## Writing register\n\n{RESEARCH_REGISTER}\n"


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


# Disease evidence strength, strongest first (matches format_disease_associations' evidence_order).
_EVIDENCE_STRENGTH = {"gwas": 0, "curated": 1, "text_mined": 2, "predicted": 3}
_EVIDENCE_LABEL = {0: "gwas", 1: "curated", 2: "text_mined", 3: "predicted"}


def count_recurrence_qualifying(pairs, min_members: int) -> int:
    """Count groups whose distinct-member set meets ``min_members``.

    Single source for the disease/pathway "qualifying" total reported in synthesis_context_stats.
    Mirrors the dedup the aggregators apply (distinct member entity per group), so the reported
    total cannot drift from the aggregator's actual cut (plan 004 R2). ``pairs`` is an iterable of
    ``(group_curie, member_curie)``.
    """
    groups: dict[str, set[str]] = {}
    for group_curie, member_curie in pairs:
        groups.setdefault(group_curie, set()).add(member_curie)
    return sum(1 for members in groups.values() if len(members) >= min_members)


def aggregate_shared_diseases(
    disease_associations: list[DiseaseAssociation],
    min_members: int,
    max_items: int,
) -> str:
    """Module-level disease recurrence: diseases shared across ≥ min_members distinct members.

    Dedupes ``(entity_curie, disease_curie)`` before counting *distinct* member entities (the
    additive reducers can carry duplicates from parallel branches), keeping the **strongest**
    ``evidence_type`` per (entity, disease) so evidence strength is not silently dropped. Ranks by
    distinct-member count, then evidence strength; caps to ``max_items``. Returns ``""`` when nothing
    qualifies (so single-entity / small queries emit nothing — R5).
    """
    if not disease_associations:
        return ""

    # disease_curie -> {entity_curie: strongest_evidence_rank}
    by_disease: dict[str, dict[str, int]] = {}
    names: dict[str, str] = {}
    for d in disease_associations:
        members = by_disease.setdefault(d.disease_curie, {})
        rank = _EVIDENCE_STRENGTH.get(d.evidence_type, 99)
        if d.entity_curie not in members or rank < members[d.entity_curie]:
            members[d.entity_curie] = rank
        names.setdefault(d.disease_curie, d.disease_name)

    qualifying = [
        (curie, members) for curie, members in by_disease.items() if len(members) >= min_members
    ]
    if not qualifying:
        return ""

    # member count desc, then strongest evidence (lowest rank) asc
    qualifying.sort(key=lambda cm: (-len(cm[1]), min(cm[1].values())))
    qualifying = qualifying[:max_items]

    lines = ["## Module-Level Disease Recurrence\n"]
    lines.append(f"*Diseases associated with multiple module members (shared by ≥{min_members}).*\n")
    for curie, members in qualifying:
        strongest = _EVIDENCE_LABEL.get(min(members.values()), "—")
        lines.append(
            f"- **{names[curie]}** (`{curie}`) — {len(members)} members [strongest: {strongest}]"
        )
        lines.append(f"  - Members: {', '.join(sorted(members))}")
    lines.append("")
    return "\n".join(lines)


def aggregate_shared_pathways(
    pathway_memberships: list[PathwayMembership],
    min_members: int,
    max_items: int,
) -> str:
    """Module-level pathway recurrence: pathways/processes shared across ≥ min_members members.

    Dedupes ``(entity_curie, pathway_curie)`` (set of distinct members), filters by threshold, ranks
    by member count, caps to ``max_items``. Returns ``""`` when nothing qualifies (R5).
    """
    if not pathway_memberships:
        return ""

    by_pathway: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    for p in pathway_memberships:
        by_pathway.setdefault(p.pathway_curie, set()).add(p.entity_curie)
        names.setdefault(p.pathway_curie, p.pathway_name)

    qualifying = [(c, m) for c, m in by_pathway.items() if len(m) >= min_members]
    if not qualifying:
        return ""

    qualifying.sort(key=lambda cm: -len(cm[1]))
    qualifying = qualifying[:max_items]

    lines = ["## Module-Level Pathway Recurrence\n"]
    lines.append(
        f"*Pathways/processes shared across multiple module members (shared by ≥{min_members}).*\n"
    )
    for curie, members in qualifying:
        lines.append(f"- **{names[curie]}** (`{curie}`) — {len(members)} members")
        lines.append(f"  - Members: {', '.join(sorted(members))}")
    lines.append("")
    return "\n".join(lines)


def format_member_table(
    resolved_entities: list[EntityResolution],
    novelty_scores: list[NoveltyScore],
    disease_associations: list[DiseaseAssociation],
    max_rows: int,
) -> str:
    """Compact one-row-per-member prioritization table (the per-member axis of a module report).

    Joins ``novelty_scores`` (edge_count + classification bucket) with ``resolved_entities``
    (name/category) by curie — over the **union** of curies so a curie present in only one source
    still renders (graceful join, no KeyError). "Top disease" is the member's strongest-evidence
    ``DiseaseAssociation`` (or "—"). Sorted by edge_count desc; capped to ``max_rows`` (top-N) with a
    "… and N more members" elision so a 217-member table cannot itself become a dump.
    """
    name_by_curie = {e.curie: (e.resolved_name or e.raw_name) for e in resolved_entities if e.curie}
    cat_by_curie = {e.curie: e.category for e in resolved_entities if e.curie}
    novelty_by_curie = {n.curie: n for n in novelty_scores}

    # strongest-evidence disease name per entity
    top_disease: dict[str, tuple[int, str]] = {}
    for d in disease_associations:
        rank = _EVIDENCE_STRENGTH.get(d.evidence_type, 99)
        cur = top_disease.get(d.entity_curie)
        if cur is None or rank < cur[0]:
            top_disease[d.entity_curie] = (rank, d.disease_name)

    curies = list(dict.fromkeys(list(name_by_curie) + list(novelty_by_curie)))
    if not curies:
        return ""

    def _edges(curie: str) -> int:
        n = novelty_by_curie.get(curie)
        return n.edge_count if n is not None else -1

    curies.sort(key=_edges, reverse=True)
    shown = curies[:max_rows]
    hidden = len(curies) - len(shown)

    lines = ["## Member Prioritization Table\n"]
    lines.append("| Member | Category | Bucket | Edges | Top Disease |")
    lines.append("|---|---|---|---|---|")
    for curie in shown:
        name = name_by_curie.get(curie, curie)
        category = (cat_by_curie.get(curie) or "—")
        if category != "—":
            category = category.replace("biolink:", "")
        n = novelty_by_curie.get(curie)
        bucket = n.classification if n is not None else "—"
        edges = str(n.edge_count) if n is not None else "—"
        disease = top_disease.get(curie, (0, "—"))[1]
        lines.append(f"| {name} (`{curie}`) | {category} | {bucket} | {edges} | {disease} |")
    if hidden > 0:
        lines.append(f"\n*… and {hidden} more members*")
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


# Within-tier confidence ranking (high first) for capping the findings section.
_CONFIDENCE_RANK = {"high": 0, "moderate": 1, "low": 2}


def format_findings_summary(
    direct_findings: list[Finding],
    cold_start_findings: list[Finding],
    max_per_tier: int | None = None,
) -> str:
    """Format analysis findings into a summary report.

    When ``max_per_tier`` is set, each tier is ranked by confidence (high->moderate->low) and capped to
    that many findings, with a "… and N more (tier T)" elision line for the remainder. Findings are the
    dominant synthesis-context section at module scale (58% of the 882K overflow), so this cap is the
    load-bearing reduction. ``max_per_tier=None`` keeps the historical unbounded behavior.
    """
    all_findings = direct_findings + cold_start_findings
    if not all_findings:
        return ""

    lines = ["## Analysis Findings Summary\n"]

    # Group by tier
    tier_labels = {
        1: "Tier 1 (High Confidence - Direct KG Evidence)",
        2: "Tier 2 (Moderate Confidence - Derived Associations)",
        3: "Tier 3 (Speculative - Semantic Inference)",
    }

    for tier in [1, 2, 3]:
        tier_findings = [f for f in all_findings if f.tier == tier]
        if not tier_findings:
            continue
        # Strongest-confidence first so the cap keeps the most reliable findings, not an arbitrary slice.
        tier_findings.sort(key=lambda f: _CONFIDENCE_RANK.get(f.confidence, 3))
        shown = tier_findings if max_per_tier is None else tier_findings[:max_per_tier]
        elided = len(tier_findings) - len(shown)
        lines.append(f"### {tier_labels[tier]}")
        for f in shown:
            source_tag = f"[{f.source}]" if f.source else ""
            confidence_marker = {"high": "[HIGH]", "moderate": "[MOD]", "low": "[LOW]"}.get(f.confidence, "")
            lines.append(f"- {confidence_marker} **{f.entity}**: {f.claim} {source_tag}")
            if f.pmids:
                pmid_links = [format_pmid_link(pmid) for pmid in f.pmids[:5]]
                lines.append(f"  - PMIDs: {', '.join(pmid_links)}")
            if f.logic_chain:
                lines.append(f"  - _Logic: {f.logic_chain}_")
        if elided > 0:
            lines.append(f"- … and {elided} more (tier {tier})")
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


def format_bridges(
    bridges: list[Bridge],
    grounding_labels: dict[tuple[str, ...], str] | None = None,
) -> str:
    """Format cross-type bridges discovered during integration analysis.

    ``grounding_labels`` maps a bridge's ``tuple(entities)`` to its evidence-provenance chain
    label (from the bridge_grounding node, via ``grounded_bridges``). When present, the label is
    rendered per bridge so the researcher sees what kind of evidence backs each leg.
    """
    if not bridges:
        return ""

    labels = grounding_labels or {}

    # Initialise `lines` BEFORE the _render closure that appends to it: the closure captures it by
    # reference, so defining the list first keeps the dependency obvious and avoids an UnboundLocalError
    # if a future refactor ever calls _render before this point.
    lines = ["## Cross-Type Bridges\n"]
    lines.append("*Multi-hop paths connecting different entity types across the analysis.*\n")

    def _render(b: Bridge, show_predicates: bool) -> None:
        # show_predicates preserves the original behavior: Tier 2 lists per-hop predicates,
        # Tier 3 (speculative) deliberately omits them.
        novelty_tag = "Known" if b.novelty == "known" else "Inferred"
        lines.append(f"\n**{b.path_description}** [{novelty_tag}]")
        if b.entity_names:
            path_with_names = " -> ".join(
                f"{name} (`{curie}`)" for name, curie in zip(b.entity_names, b.entities)
            )
            lines.append(f"  - Path: {path_with_names}")
        else:
            lines.append(f"  - Entities: {' -> '.join(b.entities)}")
        if show_predicates and b.predicates:
            lines.append(f"  - Predicates: {' -> '.join(b.predicates)}")
        if b.significance:
            lines.append(f"  - **Significance**: {b.significance}")
        label = labels.get(tuple(b.entities))
        if label:
            lines.append(f"  - **Evidence provenance**: {label}")

    # Separate by tier
    tier2 = [b for b in bridges if b.tier == 2]
    tier3 = [b for b in bridges if b.tier == 3]

    if tier2:
        lines.append("### High-Confidence Bridges (Tier 2)")
        for b in tier2:
            _render(b, show_predicates=True)
        lines.append("")

    if tier3:
        lines.append("### Speculative Bridges (Tier 3)")
        for b in tier3:
            _render(b, show_predicates=False)
        lines.append("")

    return "\n".join(lines)


def grounding_labels_from_state(state: DiscoveryState) -> dict[tuple[str, ...], str]:
    """Build a {tuple(entities) -> chain label} map from grounded_bridges for bridge rendering."""
    labels: dict[tuple[str, ...], str] = {}
    for gb in state.get("grounded_bridges", []) or []:
        grounding = getattr(gb, "grounding", None)
        if grounding is not None and getattr(grounding, "label", ""):
            labels[tuple(gb.entities)] = grounding.label
    return labels


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


# Caps for rendering grounded literature into the synthesis context (R5). Based on the Spike-0 budget
# (max_papers_per_hyp=4, max_abstract_chars=1500) and kept well under max_buffer_size.
# NOTE: this is a DISPLAY cap, deliberately >= LiteratureGroundingConfig.papers_per_hypothesis
# (default 3) so the renderer never hides papers that grounding actually attached. Under the default
# config a hypothesis carries <= 3 entries, so the slice and the "+N more papers" path only activate
# if that grounding config value is raised above 4 — intentional operator headroom, not dead code.
MAX_LIT_PAPERS_PER_HYPOTHESIS = 4
MAX_LIT_ABSTRACT_CHARS = 1500


def _truncate_abstract(text: str, limit: int = MAX_LIT_ABSTRACT_CHARS) -> str:
    """Trim an abstract/passage to the per-entry budget, marking truncation."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "… [truncated]"


def format_literature_evidence(hypotheses: list[Hypothesis]) -> str:
    """Render grounded literature (abstracts / key passages) for the synthesis context (R5).

    Ground-before-synthesis delivers value here: synthesis reasons over the grounded abstracts, not
    just a trailing references table. Only S2-inline and PMID-backfilled entries carry an ``abstract``
    body; OpenAlex/Exa entries carry only title/citation (sometimes a ``key_passage``) — render what
    exists, never fabricate.

    Legibility + calibration (guards mis-calibrated trust): each hypothesis carries a grounded/ungrounded
    marker so a reader can tell "not grounded" from "unsupported". "Grounded" means abstracts were
    *fetched* for the hypothesis (only the top hypotheses by tier are attempted), NOT that the claim is
    verified or stronger; "ungrounded" spans both "no papers found" and "ranked below the grounding cap",
    so the absence of a [Literature] citation is not evidence of weakness.
    """
    if not hypotheses:
        return ""

    grounded = [h for h in hypotheses if getattr(h, "literature_support", None)]
    ungrounded = [h for h in hypotheses if not getattr(h, "literature_support", None)]

    if not grounded:
        # Hypotheses exist but none carry literature (well-characterized-only run, all below the cap,
        # or no papers found). Emit only a short calibration note so the absence reads correctly.
        return (
            "## Literature Evidence\n\n"
            "_No hypotheses in this run have grounded literature attached. Absence of literature here "
            "means none was fetched or found — NOT that the hypotheses are unsupported._"
        )

    lines = [
        "## Literature Evidence\n",
        "_\"Grounded\" means abstracts were fetched for a hypothesis (top hypotheses by tier only); it "
        "does NOT mean the claim is verified or stronger. Cite these with a [Literature] tag only where "
        "they genuinely support a claim._\n",
    ]

    for h in grounded:
        all_lits = list(h.literature_support or [])
        lits = all_lits[:MAX_LIT_PAPERS_PER_HYPOTHESIS]
        more = f" (+{len(all_lits) - len(lits)} more papers)" if len(all_lits) > len(lits) else ""
        lines.append(f"### {h.title}  — ✓ literature-grounded{more}")
        for lit in lits:
            cite_bits = [b for b in [lit.authors, str(lit.year) if lit.year else None] if b]
            citation = ", ".join(cite_bits) if cite_bits else (lit.source or "source unknown")
            header = f"- **{lit.title or 'Untitled'}** ({citation})"
            if lit.doi:
                header += f" doi:{lit.doi}"
            lines.append(header)
            body = lit.abstract or lit.key_passage or ""
            if body:
                lines.append(f"  {_truncate_abstract(body)}")
            # else: OpenAlex/Exa entry with no abstract body — title/citation already rendered.
        lines.append("")

    if ungrounded:
        names = ", ".join(h.title for h in ungrounded[:10])
        extra = f" (+{len(ungrounded) - 10} more)" if len(ungrounded) > 10 else ""
        lines.append(
            f"_Ungrounded hypotheses (no abstracts fetched — speculative, ranked below the grounding "
            f"cap, or no papers found; not a sign of weakness): {names}{extra}._"
        )

    return "\n".join(lines)


def _disease_pathway_sections(state: DiscoveryState) -> list[str]:
    """Disease + pathway sections, module-aware.

    For module queries (>= ``module_mode_min_entities`` distinct resolved entities) emit cross-entity
    recurrence aggregation + the per-member table in place of the unbounded per-entity dumps (which
    were 21% + 17% of the module-scale overflow). Below that threshold (single/pair/triple queries)
    keep the existing per-entity sections verbatim (R5). The threshold is the *module-mode* switch,
    distinct from ``min_members_for_recurrence`` (which gates the recurrence lists).
    """
    cfg = get_pipeline_config().synthesis
    disease_associations = state.get("disease_associations", [])
    pathway_memberships = state.get("pathway_memberships", [])
    resolved = state.get("resolved_entities", [])
    novelty_scores = state.get("novelty_scores", [])
    distinct_entities = len({e.curie for e in resolved if e.curie})

    out: list[str] = []
    if distinct_entities >= cfg.module_mode_min_entities:
        candidates = [
            aggregate_shared_diseases(
                disease_associations, cfg.min_members_for_recurrence, cfg.max_aggregated_diseases
            ),
            aggregate_shared_pathways(
                pathway_memberships, cfg.min_members_for_recurrence, cfg.max_aggregated_pathways
            ),
            format_member_table(
                resolved, novelty_scores, disease_associations, cfg.max_member_table_rows
            ),
        ]
    else:
        candidates = [
            format_disease_associations(disease_associations),
            format_pathway_memberships(pathway_memberships),
        ]
    for section in candidates:
        if section:
            out.append(section)
    return out


_CHARS_PER_TOKEN = 3.5  # matches the assemble_synthesis_context tripwire estimate
_MODEL_WINDOW_TOKENS = 200_000  # the model input window the char budget is a proxy for


def _compute_context_stats(state: DiscoveryState, context: str) -> dict[str, Any]:
    """Context-compression + budget telemetry for ``synthesis_context_stats`` (plan 004).

    Counts that are pure functions of inputs already in scope (findings, member table, literature)
    are derived directly; the one filter-dependent count (disease/pathway qualifying, post
    ``min_members_for_recurrence``) comes from ``count_recurrence_qualifying`` so it matches the
    aggregator's cut (R2). ``module_mode`` gates which capped sections exist (per-entity mode uses
    uncapped disease/pathway formatters, so those rows are omitted rather than reported as no-elision).
    """
    cfg = get_pipeline_config().synthesis

    resolved = state.get("resolved_entities", [])
    distinct_entities = len({e.curie for e in resolved if e.curie})
    module_mode = distinct_entities >= cfg.module_mode_min_entities

    all_findings = state.get("direct_findings", []) + state.get("cold_start_findings", [])
    f_shown = f_total = 0
    for tier in (1, 2, 3):
        n = sum(1 for f in all_findings if f.tier == tier)
        f_total += n
        f_shown += min(n, cfg.max_findings_per_tier)
    sections: dict[str, dict[str, int]] = {
        "findings": {"shown": f_shown, "total": f_total, "elided": f_total - f_shown},
    }

    if module_mode:
        diseases = state.get("disease_associations", [])
        d_total = count_recurrence_qualifying(
            ((d.disease_curie, d.entity_curie) for d in diseases), cfg.min_members_for_recurrence
        )
        d_shown = min(d_total, cfg.max_aggregated_diseases)
        sections["diseases"] = {"shown": d_shown, "total": d_total, "elided": d_total - d_shown}

        pathways = state.get("pathway_memberships", [])
        p_total = count_recurrence_qualifying(
            ((p.pathway_curie, p.entity_curie) for p in pathways), cfg.min_members_for_recurrence
        )
        p_shown = min(p_total, cfg.max_aggregated_pathways)
        sections["pathways"] = {"shown": p_shown, "total": p_total, "elided": p_total - p_shown}

        # member table joins resolved curies + novelty curies (union), mirroring format_member_table
        novelty = state.get("novelty_scores", [])
        member_curies = {e.curie for e in resolved if e.curie} | {n.curie for n in novelty}
        m_total = len(member_curies)
        m_shown = min(m_total, cfg.max_member_table_rows)
        sections["member_table"] = {"shown": m_shown, "total": m_total, "elided": m_total - m_shown}

    hypotheses = state.get("hypotheses", [])
    attached = sum(1 for h in hypotheses if getattr(h, "literature_support", None))

    chars = len(context)
    est_tokens = round(chars / _CHARS_PER_TOKEN)
    return {
        "context_chars": chars,
        "context_est_tokens": est_tokens,
        "max_context_chars": cfg.max_context_chars,
        "char_budget_pct": round(chars / cfg.max_context_chars * 100, 1) if cfg.max_context_chars else 0.0,
        "window_tokens": _MODEL_WINDOW_TOKENS,
        "window_pct": round(est_tokens / _MODEL_WINDOW_TOKENS * 100, 1),
        "module_mode": module_mode,
        "module_mode_threshold": cfg.module_mode_min_entities,
        "distinct_entities": distinct_entities,
        "sections": sections,
        "literature": {"attached": attached, "total": len(hypotheses)},
    }


def assemble_synthesis_context(state: DiscoveryState, stats_out: dict | None = None) -> str:
    """
    Assemble all accumulated state into a context block for LLM synthesis.

    This function gathers data from all previous nodes and formats it into
    a comprehensive context that the LLM can use to generate a discovery report.

    When ``stats_out`` is provided, it is populated in place with context-compression telemetry
    (plan 004); computation is best-effort and never affects the returned context string.
    """
    cfg = get_pipeline_config().synthesis
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
    
    # Disease associations + pathway memberships (module-aware: aggregation + member table at
    # module scale, per-entity dumps for small queries — these two sections were 38% of the overflow)
    sections.extend(_disease_pathway_sections(state))

    # Pathway enrichment (shared neighbors and themes)
    shared_neighbors = state.get("shared_neighbors", [])
    biological_themes = state.get("biological_themes", [])
    enrichment_section = format_shared_neighbors(shared_neighbors, biological_themes)
    if enrichment_section:
        sections.append(enrichment_section)
    
    # Cross-type bridges
    bridges = state.get("bridges", [])
    bridges_section = format_bridges(bridges, grounding_labels_from_state(state))
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
    
    # All findings (capped per tier — the dominant section, 58% of the module-scale overflow)
    direct_findings = state.get("direct_findings", [])
    cold_start_findings = state.get("cold_start_findings", [])
    findings_section = format_findings_summary(
        direct_findings, cold_start_findings, max_per_tier=cfg.max_findings_per_tier
    )
    if findings_section:
        sections.append(findings_section)

    # Grounded literature evidence (R5): abstracts/passages the model should reason over and cite
    # with [Literature]. This is the value-delivering change — synthesis now sees the grounded
    # abstracts (produced upstream by literature_grounding), not just a trailing references table.
    hypotheses = state.get("hypotheses", [])
    literature_section = format_literature_evidence(hypotheses)
    if literature_section:
        sections.append(literature_section)

    context = "\n".join(sections)

    # Backstop tripwire (not a truncator): the per-section caps should keep us well under budget.
    # The real ceiling is the model's ~200K-token input window; max_context_chars is a char proxy,
    # so log an estimated token count (~3.5 chars/token for this CURIE-dense content) to make the
    # warning interpretable. Reaching here means a cap is mis-set — surface it loudly.
    if len(context) > cfg.max_context_chars:
        logger.warning(
            "synthesis context %d chars (~%dK est. tokens) exceeds max_context_chars=%d "
            "(~200K-token window) — a per-section cap is likely mis-set",
            len(context), round(len(context) / 3.5 / 1000), cfg.max_context_chars,
        )

    if stats_out is not None:
        try:
            stats_out.update(_compute_context_stats(state, context))
        except Exception:  # noqa: BLE001 — telemetry is best-effort, never break context assembly
            logger.warning("synthesis_context_stats computation failed", exc_info=True)

    return context


def fallback_report(state: DiscoveryState) -> str:
    """
    Generate a structured markdown report without LLM synthesis.
    
    This is the fallback used when the Claude Agent SDK is not available
    or when running tests. It preserves the existing report format, and — like
    assemble_synthesis_context — is module-aware (aggregation + member table at module scale)
    so the degraded path is also bounded, not an 882KB raw dump.
    """
    cfg = get_pipeline_config().synthesis
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
    bridges_section = format_bridges(bridges, grounding_labels_from_state(state))
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

    # Disease + pathway (module-aware: aggregation + member table at module scale, per-entity
    # dumps for small queries — keeps the fallback path bounded too)
    report_lines.extend(_disease_pathway_sections(state))

    # Inferred associations from cold-start (structured data)
    inference_section = format_inferred_associations(inferred_associations, analogues_found)
    if inference_section:
        report_lines.append(inference_section)

    # Analysis findings (summary from both branches) — capped per tier (dominant section)
    findings_section = format_findings_summary(
        direct_findings, cold_start_findings, max_per_tier=cfg.max_findings_per_tier
    )
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


@validate_state(SynthesisInput, SynthesisOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Generate the final synthesis report from all analysis phases.

    Reads the already-validated `bridges` and the grounded `hypotheses` from state (produced
    upstream by hypothesis_extraction and literature_grounding) and renders the report:
    - Phase A: Assemble all state into context block
    - Phase B: LLM synthesis (if SDK available) or fallback report
    - R6: append the references table (built from the grounded hypotheses) to both paths

    Returns:
        synthesis_report: Formatted markdown report (the only domain output — hypotheses and
            bridges are owned upstream and are NOT re-emitted here)
        model_usages: SDK usage record(s) for cost tracking, if any
    """
    # Count input findings
    direct_findings = state.get("direct_findings", [])
    cold_start_findings = state.get("cold_start_findings", [])
    total_findings = len(direct_findings) + len(cold_start_findings)
    logger.info("Starting synthesis with %d findings", total_findings)
    start = time.time()

    # Bridges are already validated and hypotheses already produced upstream by the
    # hypothesis_extraction node (and grounded by literature_grounding); synthesis reads them
    # from state rather than recomputing — no validation happens here anymore.
    bridges = state.get("bridges", [])

    # Phase B: Assemble context (always done). Capture context-compression telemetry via the
    # out-param (plan 004); best-effort inside assemble, so this never affects the report.
    context_stats: dict[str, Any] = {}
    context = assemble_synthesis_context(state, stats_out=context_stats)

    # Phase C: LLM synthesis or fallback
    usage_record = None
    # Run-level marker emitted when synthesis silently degrades to the deterministic
    # fallback (plan Unit 7). The motivating case — context overflow — succeeds at the
    # HTTP level but returns empty text, so the run otherwise reads as status=complete,
    # errors=0 and the degradation is invisible. The marker surfaces it in the report.
    fallback_marker: str | None = None
    if HAS_SDK:
        try:
            options = ClaudeAgentOptions(
                system_prompt=SYNTHESIS_PROMPT,
                allowed_tools=[],  # No tools - pure reasoning
                max_turns=1,
                permission_mode="bypassPermissions",
            )

            text, usage_record = await query_with_usage(
                prompt=context,
                options=options,
                node_name="synthesis",
            )
            # NOTE: query_with_usage joins text blocks with "", not "\n" (previous behavior).
            # All other nodes use "".join(); synthesis now matches them.

            if text.strip():
                report = text
            else:
                report = fallback_report(state)
                fallback_marker = (
                    "synthesis: LLM returned empty output; fell back to deterministic "
                    "report (possible context overflow)"
                )
        except Exception as e:
            # R3: a genuine SDK synthesis failure must be VISIBLE, never silent. Log with the
            # traceback and record it in the additive state["errors"] channel (so coverage/monitoring
            # see the degradation) instead of silently emitting a deterministic dump as if all was well.
            logger.warning(
                "synthesis LLM call failed, using fallback report: %s", e, exc_info=True
            )
            report = fallback_report(state)
            fallback_marker = (
                f"synthesis: LLM call failed ({type(e).__name__}); fell back to "
                "deterministic report"
            )
    else:
        # SDK unavailable is an environment condition, not a run-time degradation —
        # no marker (avoids polluting errors in SDK-less dev/test environments).
        report = fallback_report(state)

    # Hypotheses are produced upstream (hypothesis_extraction) and grounded by
    # literature_grounding; read them from state.
    hypotheses = state.get("hypotheses", [])

    # R6: synthesis owns the references table now (grounding stopped appending it in Unit 3).
    # Append it AFTER the SDK/fallback convergence point so BOTH the LLM-success path and the
    # fallback_report path emit it — the fallback omission was the highest-risk silent regression.
    references_table = build_references_table(hypotheses)
    if references_table:
        report = report + "\n" + references_table

    duration = time.time() - start
    logger.info(
        "Completed synthesis in %.1fs — hypotheses=%d, report_length=%d, bridges=%d",
        duration, len(hypotheses), len(report), len(bridges)
    )

    # Report-only return (R4/R12): hypotheses and bridges are produced/owned upstream now, so
    # synthesis must NOT re-emit them. extra='ignore' on SynthesisOutput would silently let a
    # stray `bridges` return through, so it is removed deliberately, not relied on to be dropped.
    result: dict[str, Any] = {
        "synthesis_report": report,
        "model_usages": [usage_record] if usage_record else [],
    }
    # Context-compression telemetry (plan 004) — single-writer plain field, last-write-wins.
    if context_stats:
        result["synthesis_context_stats"] = context_stats
    # errors uses an operator.add reducer; only emit on a degraded fallback (Unit 7).
    if fallback_marker:
        result["errors"] = [fallback_marker]
    return result
