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
    from claude_agent_sdk.types import McpSSEServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP server configuration
KESTREL_URL = "https://kestrel.nathanpricelab.com/mcp"

# Hub threshold - nodes with more edges are flagged
HUB_THRESHOLD = 1000

# System prompt for pathway enrichment analysis
PATHWAY_ENRICHMENT_PROMPT = """You are a pathway enrichment analyst for biomedical knowledge graphs.

Given a set of entities (CURIEs with their categories), find shared biological
context by querying their neighborhoods:

1. For each entity, use one_hop_query to get neighbors (limit to 50 per entity)
2. Find neighbors that appear in 2+ entity neighborhoods (shared neighbors)
3. For each shared neighbor found:
   - Get its degree using get_nodes to check edge count
   - If degree > 1000, flag as HUB (these are less specific)
   - Note the predicate types connecting it to the input entities
   - Note its category
4. Group shared neighbors by category to identify biological themes
5. Rank themes by: number of input entities connected, specificity (non-hub first)

IMPORTANT: Discount hub nodes (degree >1000) like GO:0005515 (protein binding),
GO:0005737 (cytoplasm), UBERON:0000061 (anatomical structure), etc. These connect 
to everything and do not provide specific biological insight. Still report them but 
flag them clearly as hubs.

Return ONLY a valid JSON object:
{
  "shared_neighbors": [
    {
      "curie": "GO:0006915",
      "name": "apoptotic process",
      "category": "biolink:BiologicalProcess",
      "degree": 245,
      "is_hub": false,
      "connected_inputs": ["HGNC:11998", "HGNC:7989"],
      "predicates": ["biolink:participates_in", "biolink:actively_involved_in"]
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

If no shared neighbors are found, return:
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

    # Skip if only 1 entity (need 2+ to find shared neighbors)
    if len(valid_entities) < 2:
        logger.warning("Skipped pathway_enrichment: need at least 2 entities (have %d)", len(valid_entities))
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": ["Need at least 2 entities for pathway enrichment"],
        }

    # Check SDK availability
    if not HAS_SDK:
        # Return placeholder for testing without SDK
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": ["Claude Agent SDK not available for pathway enrichment"],
        }

    # Build entity list for the prompt
    entity_list = "\n".join([
        f"- {e.curie} ({e.resolved_name or e.raw_name}, {e.category or 'unknown'})"
        for e in valid_entities
    ])

    # Construct the full prompt
    full_prompt = f"""{PATHWAY_ENRICHMENT_PROMPT}

Entities to analyze:
{entity_list}

Find shared neighbors and biological themes for these {len(valid_entities)} entities.
"""

    try:
        # Configure Kestrel MCP server
        kestrel_config: McpSSEServerConfig = {
            "type": "sse",
            "url": KESTREL_URL,
        }

        options = ClaudeAgentOptions(
            allowed_tools=["mcp__kestrel__one_hop_query", "mcp__kestrel__get_nodes"],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=6,  # More turns for multi-entity analysis
            permission_mode="auto",
        )

        # Execute the query
        result = await query(prompt=full_prompt, options=options)
        result_text = result.response if hasattr(result, "response") else str(result)

        # Parse the result
        shared_neighbors, themes, parse_errors = parse_enrichment_result(result_text)

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
