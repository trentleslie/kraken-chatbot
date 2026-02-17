"""
Pathway Enrichment Node: Find shared biological context across resolved entities.

This node analyzes the FULL set of resolved entities (not branch-specific) to find
neighbors that are shared by 2+ input entities. Shared neighbors indicate common
biological themes and provide context for understanding relationships.

Key capabilities:
- Query neighborhoods for all resolved entities
- Identify shared neighbors (appear in 2+ neighborhoods)
- Flag hub nodes (degree > 1000) that create spurious associations
- Group shared neighbors by category to identify biological themes
- Rank themes by input coverage and specificity
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ..state import (
    DiscoveryState, SharedNeighbor, BiologicalTheme, Finding
)

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

# Hub threshold - nodes with more edges are flagged
HUB_THRESHOLD = 1000

# Minimum edge count to include entity in pathway analysis
# Only moderate (20-199) and well-characterized (200+) entities have enough
# KG connections to produce meaningful shared neighbor analysis
MIN_EDGE_COUNT = 20

# Timeout for SDK query (8 minutes for multi-entity analysis)
SDK_QUERY_TIMEOUT = 480

# System prompt for pathway enrichment analysis
PATHWAY_ENRICHMENT_PROMPT = """You are a biomedical knowledge graph analyst finding shared biological context.

You have access to these Kestrel MCP tools:
- one_hop_query: Query neighbors of an entity. Use with curie parameter to get connected nodes.
- get_nodes: Get details about specific nodes including edge count (degree).

TASK: For the given list of entities, find neighbors that are shared by 2+ input entities.

STEP 1: For each input entity CURIE, call one_hop_query to get its neighbors.
        Example: one_hop_query(curie="CHEBI:28757") returns neighbors of fructose.

STEP 2: Identify neighbors that appear in 2+ entity neighborhoods (shared neighbors).
        Track which input entities connect to each shared neighbor.

STEP 3: For promising shared neighbors, call get_nodes to check their degree (edge count).
        If degree > 1000, mark as hub (less specific, connects to everything).

STEP 4: Group shared neighbors by category to identify biological themes.

CRITICAL: Hub nodes (degree >1000) like GO:0005515 (protein binding), GO:0005737 (cytoplasm)
connect to everything and provide less insight. Flag them clearly.

Return ONLY a valid JSON object (no other text):
{
  "shared_neighbors": [
    {
      "curie": "GO:0006915",
      "name": "apoptotic process",
      "category": "biolink:BiologicalProcess",
      "degree": 245,
      "is_hub": false,
      "connected_inputs": ["CHEBI:28757", "CHEBI:4208"],
      "predicates": ["biolink:participates_in"]
    }
  ],
  "themes": [
    {
      "category": "biolink:BiologicalProcess",
      "members": ["GO:0006915", "GO:0006954"],
      "member_names": ["apoptotic process", "inflammatory response"],
      "input_coverage": 4,
      "top_non_hub": "GO:0006915"
    }
  ]
}

If no shared neighbors are found after querying, return:
{"shared_neighbors": [], "themes": []}
"""


def parse_enrichment_result(
    result_text: str
) -> tuple[list[SharedNeighbor], list[BiologicalTheme], list[str]]:
    """
    Parse LLM response into structured enrichment objects.

    Uses multi-tier JSON extraction for robustness.

    Returns:
        tuple of (shared_neighbors, themes, errors)
    """
    shared_neighbors: list[SharedNeighbor] = []
    themes: list[BiologicalTheme] = []
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
            errors.append("Failed to parse pathway enrichment JSON response")
            return shared_neighbors, themes, errors

    if data is None:
        return shared_neighbors, themes, errors

    # Parse shared neighbors
    for sn in data.get("shared_neighbors", []):
        try:
            shared_neighbors.append(SharedNeighbor(
                curie=sn.get("curie", ""),
                name=sn.get("name", "Unknown"),
                category=sn.get("category", "biolink:NamedThing"),
                degree=sn.get("degree", 0),
                is_hub=sn.get("is_hub", False) or sn.get("degree", 0) > HUB_THRESHOLD,
                connected_inputs=sn.get("connected_inputs", []),
                predicates=sn.get("predicates", []),
            ))
        except Exception as e:
            errors.append(f"Error parsing shared neighbor: {e}")

    # Parse themes
    for theme in data.get("themes", []):
        try:
            themes.append(BiologicalTheme(
                category=theme.get("category", "biolink:NamedThing"),
                members=theme.get("members", []),
                member_names=theme.get("member_names", []),
                input_coverage=theme.get("input_coverage", 0),
                top_non_hub=theme.get("top_non_hub"),
            ))
        except Exception as e:
            errors.append(f"Error parsing biological theme: {e}")

    return shared_neighbors, themes, errors


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Analyze pathway enrichment across all resolved entities.

    This node runs after both direct_kg and cold_start branches converge.
    It looks at ALL resolved entities with valid CURIEs to find shared
    biological context.

    Returns:
        shared_neighbors: List of neighbors shared by 2+ input entities
        biological_themes: Themes grouped by category
        errors: Any errors encountered during analysis
    """
    logger.info("Starting pathway_enrichment")
    start = time.time()

    # Collect all resolved entities with valid CURIEs
    resolved = state.get("resolved_entities", [])
    valid_entities = [
        e for e in resolved
        if e.curie and e.method != "failed"
    ]

    if not valid_entities:
        logger.warning("Skipped pathway_enrichment: no valid entities")
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": ["No valid entities for pathway enrichment"],
        }

    # Get edge counts from novelty scores to filter sparse entities
    novelty_scores = state.get("novelty_scores", [])
    edge_count_map = {ns.curie: ns.edge_count for ns in novelty_scores}

    # Filter to entities with enough KG connections (moderate + well-characterized)
    # Sparse entities (<20 edges) rarely share neighbors and waste SDK turns
    selected_entities = [
        e for e in valid_entities
        if edge_count_map.get(e.curie, 0) >= MIN_EDGE_COUNT
    ]
    filtered_count = len(valid_entities) - len(selected_entities)

    # Log selection
    logger.info(
        "pathway_enrichment: selected %d/%d entities (edge_count >= %d): %s",
        len(selected_entities), len(valid_entities), MIN_EDGE_COUNT,
        [(e.curie, edge_count_map.get(e.curie, 0)) for e in selected_entities]
    )
    if filtered_count > 0:
        logger.info(
            "pathway_enrichment: filtered out %d sparse entities (edge_count < %d)",
            filtered_count, MIN_EDGE_COUNT
        )

    # Skip if fewer than 2 entities after filtering
    if len(selected_entities) < 2:
        logger.warning(
            "Skipped pathway_enrichment: need at least 2 entities with edge_count >= %d (have %d)",
            MIN_EDGE_COUNT, len(selected_entities)
        )
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": [f"Need at least 2 entities with edge_count >= {MIN_EDGE_COUNT} for pathway enrichment"],
        }

    # Check SDK availability
    if not HAS_SDK:
        # Return placeholder for testing without SDK
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": ["Claude Agent SDK not available for pathway enrichment"],
        }

    # Build entity list for the user prompt (include edge counts for context)
    entity_list = "\n".join([
        f"- {e.curie} ({e.resolved_name or e.raw_name}, {edge_count_map.get(e.curie, 0)} edges)"
        for e in selected_entities
    ])

    # User prompt with entity count to help LLM budget tool calls
    user_prompt = f"""Analyze these {len(selected_entities)} entities and find shared neighbors:

{entity_list}

You have {len(selected_entities)} entities to query. Budget your tool calls:
- Use one_hop_query for each entity (~{len(selected_entities)} calls)
- Use get_nodes sparingly for key shared neighbors
- Return JSON when done"""

    try:
        # Configure Kestrel MCP server (stdio-based, same as entity_resolution)
        kestrel_config = McpStdioServerConfig(
            type="stdio",
            command=KESTREL_COMMAND,
            args=KESTREL_ARGS,
        )

        options = ClaudeAgentOptions(
            system_prompt=PATHWAY_ENRICHMENT_PROMPT,
            allowed_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=25,  # Enough for ~N one_hop_query + get_nodes + JSON synthesis
            permission_mode="bypassPermissions",
        )

        # Execute the query using async generator pattern with timeout
        result_text_parts: list[str] = []

        async def collect_events() -> None:
            """Collect SDK query events into result_text_parts."""
            async for event in query(prompt=user_prompt, options=options):
                if hasattr(event, 'content'):
                    for block in event.content:
                        if hasattr(block, 'text'):
                            result_text_parts.append(block.text)

        try:
            await asyncio.wait_for(collect_events(), timeout=SDK_QUERY_TIMEOUT)
        except asyncio.TimeoutError:
            duration = time.time() - start
            logger.error(
                "pathway_enrichment SDK query TIMED OUT after %ds (%.1fs elapsed)",
                SDK_QUERY_TIMEOUT, duration
            )
            return {
                "shared_neighbors": [],
                "biological_themes": [],
                "errors": [f"SDK query timed out after {SDK_QUERY_TIMEOUT}s"],
            }

        result_text = "".join(result_text_parts)

        # Log raw response for diagnosis
        logger.info(
            "pathway_enrichment raw response: length=%d, preview=%s",
            len(result_text), repr(result_text[:500]) if result_text else "empty"
        )

        # Parse the result
        shared_neighbors, themes, parse_errors = parse_enrichment_result(result_text)

        # Log parse errors if any
        if parse_errors:
            logger.warning("pathway_enrichment parse errors: %s", parse_errors)

        # Create findings from top themes
        findings: list[Finding] = []
        for theme in themes[:5]:  # Top 5 themes
            if theme.input_coverage >= 2:
                non_hub_count = sum(
                    1 for sn in shared_neighbors 
                    if sn.curie in theme.members and not sn.is_hub
                )
                category_name = theme.category.replace("biolink:", "")
                findings.append(Finding(
                    entity=", ".join(theme.members[:3]),
                    claim=f"Shared {category_name} context: {', '.join(theme.member_names[:3])} connects {theme.input_coverage} input entities",
                    tier=2,
                    source="pathway_enrichment",
                    confidence="high" if non_hub_count > 0 else "moderate",
                ))

        duration = time.time() - start
        logger.info(
            "Completed pathway_enrichment in %.1fs — themes=%d, shared_neighbors=%d",
            duration, len(themes), len(shared_neighbors)
        )

        return {
            "shared_neighbors": shared_neighbors,
            "biological_themes": themes,
            "direct_findings": findings,  # Add to direct_findings via reducer
            "errors": parse_errors,
        }

    except Exception as e:
        duration = time.time() - start
        logger.error("Pathway enrichment failed after %.1fs: %s", duration, str(e))
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": [f"Pathway enrichment failed: {str(e)}"],
        }
