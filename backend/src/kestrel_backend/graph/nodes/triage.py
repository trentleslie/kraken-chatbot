"""
Triage & Route Node: Classify entities by KG connectivity for routing.

This node performs two-phase operation:
1. Novelty Scoring - Count edges for each resolved entity via one_hop_query
2. Route Planning - Classify entities into buckets for conditional routing

Classification thresholds:
- cold_start: 0 edges (no KG presence)
- sparse: 1-19 edges (limited connectivity)
- moderate: 20-199 edges (reasonable coverage)
- well_characterized: >=200 edges (rich KG representation)

The route_after_triage function in builder.py uses these classifications
to route entities to the appropriate analysis branches (direct_kg or cold_start).
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ..state import DiscoveryState, NoveltyScore, EntityResolution

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

# Concise prompt for edge counting
EDGE_COUNT_PROMPT = """You are a knowledge graph edge counter.
Given a CURIE, use one_hop_query to retrieve its neighbors.
Count the total number of edges (relationships) returned.

Return ONLY a valid JSON object with this exact structure (no other text):
{"curie": "PREFIX:ID", "edge_count": N}

If the query fails or returns no results, return:
{"curie": "PREFIX:ID", "edge_count": 0}

Be extremely concise. No explanations."""

# Batch size for parallel edge counting
BATCH_SIZE = 6

# Classification thresholds
THRESHOLD_WELL_CHARACTERIZED = 200
THRESHOLD_MODERATE = 20
THRESHOLD_SPARSE = 1


def classify_by_edge_count(edge_count: int) -> str:
    """Classify entity by edge count thresholds."""
    if edge_count >= THRESHOLD_WELL_CHARACTERIZED:
        return "well_characterized"
    elif edge_count >= THRESHOLD_MODERATE:
        return "moderate"
    elif edge_count >= THRESHOLD_SPARSE:
        return "sparse"
    else:
        return "cold_start"


def parse_edge_count_result(curie: str, raw_name: str, result_text: str) -> NoveltyScore:
    """Parse LLM response into NoveltyScore object."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{[^{}]+\}", result_text)
        if json_match:
            data = json.loads(json_match.group())
            edge_count = int(data.get("edge_count", 0))
            return NoveltyScore(
                curie=curie,
                raw_name=raw_name,
                edge_count=edge_count,
                classification=classify_by_edge_count(edge_count),
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback for parse failures - treat as cold_start
    return NoveltyScore(
        curie=curie,
        raw_name=raw_name,
        edge_count=0,
        classification="cold_start",
    )


async def count_edges_single(entity: EntityResolution) -> NoveltyScore:
    """
    Count edges for a single resolved entity using Claude Agent SDK.

    Returns NoveltyScore with edge_count=0 (cold_start) on any error.
    """
    curie = entity.curie
    raw_name = entity.raw_name

    # Skip entities that failed resolution
    if not curie or entity.method == "failed":
        return NoveltyScore(
            curie=curie or raw_name,
            raw_name=raw_name,
            edge_count=0,
            classification="cold_start",
        )

    if not HAS_SDK:
        # SDK not available - return mock for testing
        return NoveltyScore(
            curie=curie,
            raw_name=raw_name,
            edge_count=0,
            classification="cold_start",
        )

    try:
        kestrel_config = McpStdioServerConfig(
            type="stdio",
            command=KESTREL_COMMAND,
            args=KESTREL_ARGS,
        )

        options = ClaudeAgentOptions(
            system_prompt=EDGE_COUNT_PROMPT,
            allowed_tools=[
                "mcp__kestrel__one_hop_query",
                "mcp__kestrel__get_nodes",
            ],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=2,
            permission_mode="bypassPermissions",
        )

        result_text_parts = []
        async for event in query(prompt=f"Count edges for: {curie}", options=options):
            if hasattr(event, 'content'):
                for block in event.content:
                    if hasattr(block, 'text'):
                        result_text_parts.append(block.text)

        result_text = "".join(result_text_parts)

        # Debug logging for triage edge counting
        if not result_text:
            logger.warning("Triage got empty result_text for %s", curie)
        else:
            logger.debug("Triage result for %s: %s", curie, result_text[:200] if len(result_text) > 200 else result_text)

        return parse_edge_count_result(curie, raw_name, result_text)

    except Exception as e:
        logger.warning("Edge counting failed for %s: %s", curie, str(e))
        return NoveltyScore(
            curie=curie,
            raw_name=raw_name,
            edge_count=0,
            classification="cold_start",
        )


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Triage resolved entities by KG connectivity.

    Phase 1: Count edges for each resolved entity (parallel batches)
    Phase 2: Classify into routing buckets

    Returns:
        novelty_scores: List of NoveltyScore objects
        well_characterized_curies: CURIEs with >=200 edges
        moderate_curies: CURIEs with 20-199 edges
        sparse_curies: CURIEs with 1-19 edges
        cold_start_curies: CURIEs with 0 edges
    """
    logger.info("Starting triage")
    start = time.time()

    resolved = state.get("resolved_entities", [])

    # Filter to only entities with valid CURIEs
    valid_entities = [e for e in resolved if e.curie and e.method != "failed"]

    if not valid_entities:
        # No valid entities - everything goes to cold_start or synthesis skips
        failed_names = [e.raw_name for e in resolved if e.method == "failed"]
        return {
            "novelty_scores": [],
            "well_characterized_curies": [],
            "moderate_curies": [],
            "sparse_curies": [],
            "cold_start_curies": failed_names,  # Failed resolutions as cold-start
            "errors": [],
        }

    all_scores: list[NoveltyScore] = []
    errors: list[str] = []

    # Process in batches for controlled parallelism
    for batch in chunk(valid_entities, BATCH_SIZE):
        batch_results = await asyncio.gather(
            *[count_edges_single(e) for e in batch],
            return_exceptions=True,
        )

        for entity, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                errors.append(f"Edge counting failed for {entity.curie}: {str(result)}")
                all_scores.append(NoveltyScore(
                    curie=entity.curie or entity.raw_name,
                    raw_name=entity.raw_name,
                    edge_count=0,
                    classification="cold_start",
                ))
            else:
                all_scores.append(result)

    # Classify into routing buckets
    well_characterized = [s.curie for s in all_scores if s.classification == "well_characterized"]
    moderate = [s.curie for s in all_scores if s.classification == "moderate"]
    sparse = [s.curie for s in all_scores if s.classification == "sparse"]
    cold_start = [s.curie for s in all_scores if s.classification == "cold_start"]

    # Add failed resolutions to cold_start bucket
    failed_names = [e.raw_name for e in resolved if e.method == "failed"]
    cold_start.extend(failed_names)

    duration = time.time() - start
    logger.info(
        "Completed triage in %.1fs — well_char=%d, moderate=%d, sparse=%d, cold_start=%d",
        duration, len(well_characterized), len(moderate), len(sparse), len(cold_start)
    )

    return {
        "novelty_scores": all_scores,
        "well_characterized_curies": well_characterized,
        "moderate_curies": moderate,
        "sparse_curies": sparse,
        "cold_start_curies": cold_start,
        "errors": errors,
    }
