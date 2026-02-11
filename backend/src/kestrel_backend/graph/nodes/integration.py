"""
Integration Node: Cross-type bridge detection and gap analysis.

This node performs two key functions:
1. Bridge Detection: Find multi-hop paths connecting different entity types
   (e.g., metabolite → gene → disease)
2. Gap Analysis: Identify expected-but-absent entities using Open World Assumption
   (e.g., canonical T2D markers like BCAAs, HbA1c)

The node synthesizes findings from all previous analysis steps:
- Disease associations from direct_kg
- Pathway memberships from direct_kg
- Cold-start inferences from cold_start
- Shared biological themes from pathway_enrichment
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ..state import (
    DiscoveryState, Bridge, GapEntity, Finding,
    DiseaseAssociation, PathwayMembership, InferredAssociation, BiologicalTheme
)

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpSSEServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP server configuration
KESTREL_URL = "https://kestrel.nathanpricelab.com/mcp"


INTEGRATION_PROMPT = """You are a biomedical knowledge graph integration analyst.

Given accumulated findings from multiple analysis phases, perform two tasks:

## Task 1: Cross-Type Bridge Detection

Find paths that connect different entity types across the input entities:
- metabolite → gene → disease
- protein → pathway → phenotype
- sparse entity → analogue → disease

Use one_hop_query and hybrid_search to discover multi-hop connections
that weren't directly identified in single-entity analysis.

## Task 2: Gap Analysis (Open World Assumption)

Given the study context (disease, cohort type, duration), identify
EXPECTED-but-ABSENT entities. These are canonical markers that SHOULD
appear in such a study but were not found in the input.

Common T2D conversion markers:
- BCAAs (leucine, isoleucine, valine) — early metabolic dysfunction
- Insulin / C-peptide — beta cell function
- HbA1c — glycemic control
- Ceramides — lipotoxicity
- HOMA-IR / fasting glucose — insulin resistance

CRITICAL: Frame absences using the Open World Assumption:
- "Not found in this cohort" ≠ "Does not exist"
- Absence may indicate: not measured, below detection, or genuine difference
- Informative absences reveal unique cohort characteristics

Return ONLY a valid JSON object:
{
  "bridges": [
    {
      "path_description": "metabolite → gene → disease",
      "entities": ["CHEBI:123", "HGNC:456", "MONDO:789"],
      "entity_names": ["glucose", "SLC2A2", "type 2 diabetes"],
      "predicates": ["biolink:affects_expression_of", "biolink:gene_associated_with_condition"],
      "tier": 2,
      "novelty": "known",
      "significance": "Glucose directly affects SLC2A2 expression, linking metabolic state to T2D risk"
    }
  ],
  "gaps": [
    {
      "name": "BCAAs",
      "category": "biolink:ChemicalEntity",
      "curie": null,
      "expected_reason": "Canonical early markers of T2D conversion in longitudinal studies",
      "absence_interpretation": "Not measured in this cohort or below detection threshold",
      "is_informative": false
    }
  ]
}

If no bridges or gaps are found, return:
{"bridges": [], "gaps": []}
"""


def summarize_diseases(diseases: list[DiseaseAssociation]) -> str:
    """Create a summary of disease associations for the prompt."""
    if not diseases:
        return "No disease associations found."

    lines = []
    for d in diseases[:20]:  # Limit for prompt size
        lines.append(f"- {d.entity_curie} → {d.disease_name} ({d.disease_curie}) via {d.predicate}")
    return "\n".join(lines)


def summarize_pathways(pathways: list[PathwayMembership]) -> str:
    """Create a summary of pathway memberships for the prompt."""
    if not pathways:
        return "No pathway memberships found."

    lines = []
    for p in pathways[:20]:
        lines.append(f"- {p.entity_curie} in {p.pathway_name} ({p.pathway_curie})")
    return "\n".join(lines)


def summarize_cold_start(inferences: list[InferredAssociation]) -> str:
    """Create a summary of cold-start inferences for the prompt."""
    if not inferences:
        return "No cold-start inferences."

    lines = []
    for i in inferences[:15]:
        lines.append(f"- {i.source_entity} → {i.target_name} ({i.target_curie}): {i.logic_chain}")
    return "\n".join(lines)


def summarize_themes(themes: list[BiologicalTheme]) -> str:
    """Create a summary of biological themes for the prompt."""
    if not themes:
        return "No shared biological themes."

    lines = []
    for t in themes[:10]:
        members_str = ", ".join(t.member_names[:3])
        lines.append(f"- {t.category}: {members_str} (covers {t.input_coverage} inputs)")
    return "\n".join(lines)


def build_study_context(state: DiscoveryState) -> str:
    """Build study context from state for gap analysis."""
    parts = []

    # Longitudinal info
    if state.get("is_longitudinal"):
        duration = state.get("duration_years")
        if duration:
            parts.append(f"Longitudinal study: {duration} years")
        else:
            parts.append("Longitudinal study")

    # Query text for context
    raw_query = state.get("raw_query", "")
    if raw_query:
        parts.append(f"Query: {raw_query[:200]}")

    # Entity types present
    resolved = state.get("resolved_entities", [])
    categories = set()
    for e in resolved:
        if e.category:
            categories.add(e.category.replace("biolink:", ""))
    if categories:
        parts.append(f"Entity types: {', '.join(categories)}")

    return "\n".join(parts) if parts else "No specific study context available"


def parse_integration_result(
    result_text: str
) -> tuple[list[Bridge], list[GapEntity], list[str]]:
    """
    Parse LLM response into structured integration objects.

    Returns:
        tuple of (bridges, gap_entities, errors)
    """
    bridges: list[Bridge] = []
    gaps: list[GapEntity] = []
    errors: list[str] = []
    data = None

    # Tier 1: Look for JSON code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Tier 2: Look for bare JSON object
    if data is None:
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*\}', result_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # Tier 3: Try entire response
    if data is None:
        try:
            data = json.loads(result_text.strip())
        except json.JSONDecodeError:
            errors.append("Failed to parse integration JSON response")
            return bridges, gaps, errors

    if data is None:
        return bridges, gaps, errors

    # Parse bridges
    for b in data.get("bridges", []):
        try:
            bridges.append(Bridge(
                path_description=b.get("path_description", "unknown → unknown"),
                entities=b.get("entities", []),
                entity_names=b.get("entity_names", []),
                predicates=b.get("predicates", []),
                tier=b.get("tier", 3),
                novelty=b.get("novelty", "inferred"),
                significance=b.get("significance", ""),
            ))
        except Exception as e:
            errors.append(f"Error parsing bridge: {e}")

    # Parse gap entities
    for g in data.get("gaps", []):
        try:
            gaps.append(GapEntity(
                name=g.get("name", "Unknown"),
                category=g.get("category", "biolink:NamedThing"),
                curie=g.get("curie"),
                expected_reason=g.get("expected_reason", ""),
                absence_interpretation=g.get(
                    "absence_interpretation",
                    "Not found in this cohort (Open World Assumption)"
                ),
                is_informative=g.get("is_informative", False),
            ))
        except Exception as e:
            errors.append(f"Error parsing gap entity: {e}")

    return bridges, gaps, errors


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Perform integration analysis: bridge detection and gap analysis.

    This node runs after pathway_enrichment and before temporal/synthesis.
    It synthesizes all accumulated findings to find cross-type connections
    and identify expected-but-absent entities.

    Returns:
        bridges: Cross-type connections discovered
        gap_entities: Expected-but-absent entities
        direct_findings: Findings generated from bridges
        errors: Any errors encountered
    """
    logger.info("Starting integration")
    start = time.time()

    # Check if we have any findings to integrate
    disease_associations = state.get("disease_associations", [])
    pathway_memberships = state.get("pathway_memberships", [])
    inferred_associations = state.get("inferred_associations", [])
    biological_themes = state.get("biological_themes", [])
    resolved = state.get("resolved_entities", [])

    # If nothing to integrate, return empty
    if not any([disease_associations, pathway_memberships, inferred_associations,
                biological_themes, resolved]):
        logger.info("No findings to integrate, skipping")
        return {
            "bridges": [],
            "gap_entities": [],
            "errors": ["No findings to integrate"],
        }

    # Check SDK availability
    if not HAS_SDK:
        return {
            "bridges": [],
            "gap_entities": [],
            "errors": ["Claude Agent SDK not available for integration analysis"],
        }

    # Build context summaries
    disease_summary = summarize_diseases(disease_associations)
    pathway_summary = summarize_pathways(pathway_memberships)
    cold_start_summary = summarize_cold_start(inferred_associations)
    theme_summary = summarize_themes(biological_themes)
    study_context = build_study_context(state)

    # Build entity list
    entity_list = "\n".join([
        f"- {e.curie or e.raw_name} ({e.resolved_name or e.raw_name}, {e.category or 'unknown'})"
        for e in resolved[:20]  # Limit for prompt size
    ])

    # Construct the full prompt
    full_prompt = f"""{INTEGRATION_PROMPT}

## Study Context
{study_context}

## Input Entities
{entity_list}

## Accumulated Findings

### Disease Associations
{disease_summary}

### Pathway Memberships
{pathway_summary}

### Cold-Start Inferences
{cold_start_summary}

### Biological Themes
{theme_summary}

Analyze these findings to identify cross-type bridges and expected-but-absent entities.
"""

    try:
        # Configure Kestrel MCP server
        kestrel_config: McpSSEServerConfig = {
            "type": "sse",
            "url": KESTREL_URL,
        }

        options = ClaudeAgentOptions(
            allowed_tools=[
                "mcp__kestrel__one_hop_query",
                "mcp__kestrel__hybrid_search",
                "mcp__kestrel__get_nodes",
            ],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=5,
            permission_mode="auto",
        )

        # Execute the query
        result = await query(full_prompt, options=options)
        result_text = result.response if hasattr(result, "response") else str(result)

        # Parse the result
        bridges, gaps, parse_errors = parse_integration_result(result_text)

        # Create findings from significant bridges
        findings: list[Finding] = []
        for bridge in bridges:
            if bridge.significance:
                findings.append(Finding(
                    entity=", ".join(bridge.entities[:3]),
                    claim=f"Bridge path: {bridge.path_description} — {bridge.significance}",
                    tier=bridge.tier,
                    source="integration",
                    confidence="moderate" if bridge.novelty == "known" else "low",
                ))

        # Create findings for informative gaps
        for gap in gaps:
            if gap.is_informative:
                findings.append(Finding(
                    entity=gap.name,
                    claim=f"Expected but absent: {gap.name} — {gap.absence_interpretation}",
                    tier=3,
                    source="integration_gap",
                    confidence="moderate",
                ))

        duration = time.time() - start
        logger.info(
            "Completed integration in %.1fs — bridges=%d, gaps=%d",
            duration, len(bridges), len(gaps)
        )

        return {
            "bridges": bridges,
            "gap_entities": gaps,
            "direct_findings": findings,  # Uses operator.add reducer
            "errors": parse_errors,
        }

    except Exception as e:
        duration = time.time() - start
        logger.error("Integration analysis failed after %.1fs: %s", duration, str(e))
        return {
            "bridges": [],
            "gap_entities": [],
            "errors": [f"Integration analysis failed: {str(e)}"],
        }
