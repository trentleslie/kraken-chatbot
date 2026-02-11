"""
Triage & Route Node: Classify entities by KG connectivity for routing.

This node performs two-tier edge counting and classification:

Tier 1 (API): Direct one_hop_query with mode="preview" (~100ms each)
  - Fast, returns results_count which is the edge count
  - Runs all entities in parallel

Tier 2 (LLM): Falls back to Claude Agent SDK for failures
  - Handles cases where API returns errors
  - More expensive but can retry with different parameters

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

from ...kestrel_client import call_kestrel_tool
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

# Semaphore to serialize SDK calls and prevent concurrent CLI spawn issues
SDK_SEMAPHORE = asyncio.Semaphore(1)

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


async def count_edges_via_api(entity: EntityResolution) -> NoveltyScore | None:
    """
    Tier 1: Count edges via direct Kestrel API call.

    Uses one_hop_query with mode="preview" which returns results_count (edge count).
    Returns None if API fails (triggering Tier 2 fallback).
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

    try:
        # Call one_hop_query with preview mode - returns counts instead of full data
        result = await call_kestrel_tool("one_hop_query", {
            "start_node_ids": curie,
            "mode": "preview",
            "direction": "both",
            "limit": 10000,  # High limit to get accurate count
        })

        is_error = result.get("isError", False)
        content = result.get("content", [])

        if is_error or not content:
            logger.debug("Tier 1 triage '%s': API error or no content", curie)
            return None

        # Parse response
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Tier 1 triage '%s': Could not parse JSON", curie)
            return None

        # results_count is the edge count
        edge_count = int(data.get("results_count", 0))
        classification = classify_by_edge_count(edge_count)

        logger.info(
            "Tier 1 triage '%s': edges=%d, classification=%s",
            curie, edge_count, classification
        )

        return NoveltyScore(
            curie=curie,
            raw_name=raw_name,
            edge_count=edge_count,
            classification=classification,
        )

    except Exception as e:
        logger.warning("Tier 1 triage '%s': Exception - %s", curie, str(e))
        return None


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
        async with SDK_SEMAPHORE:
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
                max_buffer_size=10 * 1024 * 1024,  # 10MB buffer for large KG responses
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
    Triage resolved entities by KG connectivity using two-tier approach.

    Tier 1 (API): Direct one_hop_query with mode="preview" for all entities
    Tier 2 (LLM): Falls back to Claude Agent SDK for any failures

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
        # No valid entities - everything goes to cold_start
        failed_names = [e.raw_name for e in resolved if e.method == "failed"]
        return {
            "novelty_scores": [],
            "well_characterized_curies": [],
            "moderate_curies": [],
            "sparse_curies": [],
            "cold_start_curies": failed_names,
            "errors": [],
        }

    all_scores: list[NoveltyScore | None] = [None] * len(valid_entities)
    errors: list[str] = []

    # ========== TIER 1: API Edge Counting ==========
    tier1_start = time.time()
    logger.info("Tier 1 (API): Counting edges for %d entities", len(valid_entities))

    # Run all API calls in parallel
    tier1_results = await asyncio.gather(
        *[count_edges_via_api(e) for e in valid_entities],
        return_exceptions=True,
    )

    tier1_success = 0
    tier1_failed_indices = []

    for i, (entity, result) in enumerate(zip(valid_entities, tier1_results)):
        if isinstance(result, Exception):
            logger.debug("Tier 1 triage '%s': Exception - %s", entity.curie, str(result))
            tier1_failed_indices.append(i)
        elif result is not None:
            all_scores[i] = result
            tier1_success += 1
        else:
            tier1_failed_indices.append(i)

    tier1_duration = time.time() - tier1_start
    logger.info(
        "Tier 1 (API) counted edges for %d/%d entities in %.1fs",
        tier1_success, len(valid_entities), tier1_duration
    )

    # ========== TIER 2: LLM Edge Counting ==========
    if tier1_failed_indices and HAS_SDK:
        tier2_start = time.time()
        failed_entities = [valid_entities[i] for i in tier1_failed_indices]
        logger.info(
            "Tier 2 (LLM): Processing %d entities that failed Tier 1",
            len(failed_entities)
        )

        tier2_results = []
        for batch in chunk(failed_entities, BATCH_SIZE):
            batch_results = await asyncio.gather(
                *[count_edges_single(e) for e in batch],
                return_exceptions=True,
            )
            tier2_results.extend(batch_results)

        # Map results back
        for idx, result in zip(tier1_failed_indices, tier2_results):
            if isinstance(result, Exception):
                errors.append(f"Edge counting failed for {valid_entities[idx].curie}: {str(result)}")
                all_scores[idx] = NoveltyScore(
                    curie=valid_entities[idx].curie or valid_entities[idx].raw_name,
                    raw_name=valid_entities[idx].raw_name,
                    edge_count=0,
                    classification="cold_start",
                )
            else:
                all_scores[idx] = result

        tier2_duration = time.time() - tier2_start
        tier2_success = sum(1 for i in tier1_failed_indices if all_scores[i] and all_scores[i].edge_count > 0)
        logger.info(
            "Tier 2 (LLM) counted edges for %d/%d entities in %.1fs",
            tier2_success, len(tier1_failed_indices), tier2_duration
        )
    elif tier1_failed_indices:
        # SDK not available - mark remaining as cold_start
        for idx in tier1_failed_indices:
            all_scores[idx] = NoveltyScore(
                curie=valid_entities[idx].curie or valid_entities[idx].raw_name,
                raw_name=valid_entities[idx].raw_name,
                edge_count=0,
                classification="cold_start",
            )

    # Ensure no None values
    final_scores = []
    for i, s in enumerate(all_scores):
        if s is None:
            final_scores.append(NoveltyScore(
                curie=valid_entities[i].curie or valid_entities[i].raw_name,
                raw_name=valid_entities[i].raw_name,
                edge_count=0,
                classification="cold_start",
            ))
        else:
            final_scores.append(s)

    # Classify into routing buckets
    well_characterized = [s.curie for s in final_scores if s.classification == "well_characterized"]
    moderate = [s.curie for s in final_scores if s.classification == "moderate"]
    sparse = [s.curie for s in final_scores if s.classification == "sparse"]
    cold_start = [s.curie for s in final_scores if s.classification == "cold_start"]

    # Add failed resolutions to cold_start bucket
    failed_names = [e.raw_name for e in resolved if e.method == "failed"]
    cold_start.extend(failed_names)

    duration = time.time() - start
    logger.info(
        "Completed triage in %.1fs — well_char=%d, moderate=%d, sparse=%d, cold_start=%d (tier1=%d, tier2=%d)",
        duration, len(well_characterized), len(moderate), len(sparse), len(cold_start),
        tier1_success, len(valid_entities) - tier1_success
    )

    return {
        "novelty_scores": final_scores,
        "well_characterized_curies": well_characterized,
        "moderate_curies": moderate,
        "sparse_curies": sparse,
        "cold_start_curies": cold_start,
        "errors": errors,
    }
