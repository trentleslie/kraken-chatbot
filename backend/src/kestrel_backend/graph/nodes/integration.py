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
    DiseaseAssociation, PathwayMembership, InferredAssociation, BiologicalTheme,
    EntityResolution
)
from ...kestrel_client import multi_hop_query

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpStdioServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP command for stdio-based server (same as entity_resolution)
KESTREL_COMMAND = "uvx"
KESTREL_ARGS = ["mcp-client-kestrel"]


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


async def detect_bridges_via_api(
    resolved_entities: list[EntityResolution],
    max_hops: int = 3,
    max_paths_per_pair: int = 5,
) -> tuple[list[Bridge], list[str]]:
    """
    Detect cross-type bridges using the multi_hop_query API.

    This function groups entities by category and queries for paths between
    entities of different types (e.g., metabolite -> gene, gene -> disease).

    Args:
        resolved_entities: List of resolved entities with CURIEs and categories
        max_hops: Maximum path length to search (default 3)
        max_paths_per_pair: Maximum paths per entity pair (default 5)

    Returns:
        tuple of (bridges, errors)
    """
    bridges: list[Bridge] = []
    errors: list[str] = []

    # Group entities by category
    by_category: dict[str, list[EntityResolution]] = {}
    for entity in resolved_entities:
        if not entity.curie or entity.method == "failed":
            continue
        category = entity.category or "biolink:NamedThing"
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(entity)

    # Need at least 2 different categories to have cross-type bridges
    if len(by_category) < 2:
        logger.info("detect_bridges_via_api: need at least 2 entity categories (have %d)", len(by_category))
        return bridges, errors

    categories = list(by_category.keys())
    logger.info(
        "detect_bridges_via_api: found %d categories: %s",
        len(categories),
        [f"{cat} ({len(by_category[cat])})" for cat in categories]
    )

    # Query cross-category pairs
    # Limit to avoid combinatorial explosion: max 3 pairs
    pairs_checked = 0
    max_pairs = 3

    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            if pairs_checked >= max_pairs:
                logger.info("detect_bridges_via_api: reached max_pairs limit (%d)", max_pairs)
                break

            # Select representative entities from each category (max 2 per category)
            start_entities = by_category[cat1][:2]
            end_entities = by_category[cat2][:2]

            start_curies = [e.curie for e in start_entities if e.curie]
            end_curies = [e.curie for e in end_entities if e.curie]

            if not start_curies or not end_curies:
                continue

            logger.info(
                "detect_bridges_via_api: querying %s (%d) -> %s (%d), max_hops=%d",
                cat1, len(start_curies), cat2, len(end_curies), max_hops
            )

            try:
                # Use doubly-pinned multi_hop_query to find paths
                result = await multi_hop_query(
                    start_node_ids=start_curies,
                    end_node_ids=end_curies,
                    max_hops=max_hops,
                    limit=max_paths_per_pair * len(start_curies) * len(end_curies),
                )

                if result.get("isError"):
                    error_text = result.get("content", [{}])[0].get("text", "Unknown error")
                    errors.append(f"multi_hop_query error for {cat1}->{cat2}: {error_text}")
                    continue

                # Parse paths from result
                paths = parse_multi_hop_result(result, start_entities, end_entities, cat1, cat2)
                bridges.extend(paths)
                pairs_checked += 1

            except Exception as e:
                logger.error("Error querying %s -> %s: %s", cat1, cat2, str(e))
                errors.append(f"Exception querying {cat1}->{cat2}: {str(e)}")

        if pairs_checked >= max_pairs:
            break

    logger.info("detect_bridges_via_api: found %d bridges from %d category pairs", len(bridges), pairs_checked)
    return bridges, errors


def parse_multi_hop_result(
    result: dict,
    start_entities: list[EntityResolution],
    end_entities: list[EntityResolution],
    cat1: str,
    cat2: str,
) -> list[Bridge]:
    """
    Parse multi_hop_query results into Bridge objects.

    Args:
        result: The MCP tool result from multi_hop_query
        start_entities: Starting entities for context
        end_entities: Ending entities for context
        cat1: Starting category name
        cat2: Ending category name

    Returns:
        List of Bridge objects
    """
    bridges: list[Bridge] = []

    try:
        # Extract JSON content from MCP response
        content = result.get("content", [])
        if not content:
            return bridges

        # The first content item should be JSON text
        json_text = content[0].get("text", "")
        if not json_text:
            return bridges

        # Parse JSON response
        data = json.loads(json_text)

        # Handle different response formats
        # Expected format: {"paths": [...]} or direct list of paths
        paths = data.get("paths", data) if isinstance(data, dict) else data

        if not isinstance(paths, list):
            logger.warning("parse_multi_hop_result: unexpected format, expected list of paths")
            return bridges

        # Convert each path to a Bridge
        for path_data in paths[:10]:  # Limit to top 10 paths
            try:
                # Extract path components
                # Expected: {"nodes": [...], "edges": [...], "predicates": [...]}
                nodes = path_data.get("nodes", [])
                predicates = path_data.get("predicates", [])
                node_names = path_data.get("node_names", [])

                if len(nodes) < 2:
                    continue

                # Build path description
                cat1_short = cat1.replace("biolink:", "")
                cat2_short = cat2.replace("biolink:", "")
                hop_count = len(nodes) - 1  # Number of hops = nodes - 1
                path_description = f"{cat1_short} → {cat2_short} ({hop_count} hops)"

                # Determine tier: 2 if path is short (1-2 hops), else 3
                tier = 2 if hop_count <= 2 else 3

                # Generate significance from path structure
                if node_names and len(node_names) == len(nodes):
                    significance = f"Path connects {node_names[0]} to {node_names[-1]}"
                else:
                    significance = f"Path connects {nodes[0]} to {nodes[-1]}"

                bridges.append(Bridge(
                    path_description=path_description,
                    entities=nodes,
                    entity_names=node_names if node_names else [],
                    predicates=predicates if predicates else [],
                    tier=tier,
                    novelty="known",  # From KG, not inferred
                    significance=significance,
                ))

            except Exception as e:
                logger.warning("Error parsing path: %s", str(e))
                continue

    except json.JSONDecodeError as e:
        logger.error("Failed to parse multi_hop_query JSON: %s", str(e))
    except Exception as e:
        logger.error("Unexpected error parsing multi_hop_result: %s", str(e))

    return bridges


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
        # Phase A: Bridge detection using multi_hop_query API
        logger.info("Starting API-based bridge detection...")
        api_bridges, api_errors = await detect_bridges_via_api(
            resolved_entities=resolved,
            max_hops=3,
            max_paths_per_pair=5,
        )

        # Phase B: Gap analysis using LLM (reasoning-intensive)
        logger.info("Starting LLM-based gap analysis...")

        # Configure Kestrel MCP server (stdio-based, same as entity_resolution)
        kestrel_config = McpStdioServerConfig(
            type="stdio",
            command=KESTREL_COMMAND,
            args=KESTREL_ARGS,
        )

        # Updated prompt focused on gap analysis only
        gap_analysis_prompt = f"""{INTEGRATION_PROMPT}

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

TASK: Focus ONLY on gap analysis (expected-but-absent entities).
Bridge detection is handled separately via API.

Return ONLY a valid JSON object:
{{
  "gaps": [
    {{
      "name": "BCAAs",
      "category": "biolink:ChemicalEntity",
      "curie": null,
      "expected_reason": "Canonical early markers of T2D conversion",
      "absence_interpretation": "Not measured or below detection",
      "is_informative": false
    }}
  ]
}}

If no gaps found, return: {{"gaps": []}}
"""

        options = ClaudeAgentOptions(
            allowed_tools=[
                "mcp__kestrel__one_hop_query",
                "mcp__kestrel__hybrid_search",
                "mcp__kestrel__get_nodes",
            ],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=5,
            permission_mode="bypassPermissions",
        )

        # Execute the query using async generator pattern
        result_text_parts = []
        async for event in query(prompt=gap_analysis_prompt, options=options):
            if hasattr(event, 'content'):
                for block in event.content:
                    if hasattr(block, 'text'):
                        result_text_parts.append(block.text)

        result_text = "".join(result_text_parts)

        # Parse only gap analysis from LLM result
        _, gaps, parse_errors = parse_integration_result(result_text)

        # Combine bridges from API with gaps from LLM
        bridges = api_bridges
        parse_errors.extend(api_errors)

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
