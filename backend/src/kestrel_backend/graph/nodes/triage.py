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
import time
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ..state import DiscoveryState, NoveltyScore, EntityResolution
from ..state_contracts import validate_state, TriageInput, TriageOutput

logger = logging.getLogger(__name__)


# Classification thresholds
THRESHOLD_WELL_CHARACTERIZED = 200
THRESHOLD_MODERATE = 20
THRESHOLD_SPARSE = 1


async def count_edges_via_api(entity: EntityResolution) -> NoveltyScore | None:
    """
    Tier 1: Count edges via direct Kestrel API call.

    Uses one_hop_query with mode="preview" which returns results_count (edge count).
    Returns None if the count fails (the caller then defaults the entity to
    cold_start).

    A single in-place retry covers genuinely *time-varying* failures (server
    ``isError`` / exception) so a transient hiccup does not silently downgrade a
    well-characterized entity to cold_start (#61). *Deterministic* per-CURIE
    failures (empty content, unparseable JSON) are NOT retried — an identical-args
    retry would re-fail — and fall to the cold_start default by returning None.
    """
    curie = entity.curie
    raw_name = entity.raw_name

    # Skip entities that failed resolution (deterministic — no query, no retry)
    if not curie or entity.method == "failed":
        return NoveltyScore(
            curie=curie or raw_name,
            raw_name=raw_name,
            edge_count=0,
            classification="cold_start",
        )

    max_attempts = 2  # one retry, for transient failures only
    for attempt in range(max_attempts):
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

            if is_error:
                # Transient server-side error — retry once, then give up.
                logger.debug(
                    "Tier 1 triage '%s': API isError (attempt %d/%d)",
                    curie, attempt + 1, max_attempts,
                )
                continue

            if not content:
                # Deterministic empty response for this CURIE — do not retry.
                logger.debug("Tier 1 triage '%s': no content", curie)
                return None

            # Parse response
            text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Deterministic malformed response — do not retry.
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
            # Transient (timeout / connection) — retry once, then give up.
            logger.warning(
                "Tier 1 triage '%s': Exception (attempt %d/%d) - %s",
                curie, attempt + 1, max_attempts, str(e),
            )
            continue

    # All attempts exhausted on transient failures → cold_start default (None).
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


@validate_state(TriageInput, TriageOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Triage resolved entities by KG connectivity using Tier-1 HTTP only (#61).

    Tier 1 (API): Direct one_hop_query with mode="preview" for all entities,
      with a single in-place retry for transient isError/exception failures.
      Deterministic failures (empty content, bad JSON) are not retried.
      Entities whose count still fails after the retry default to cold_start
      (the broken stdio-MCP Tier-2 LLM fallback was removed).

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

    # ========== Tier 2 removed — failed counts default to cold_start (#61) ==========
    # The Tier-2 LLM edge-counting fallback configured the broken stdio MCP, so with
    # no working tools it could only guess a number. count_edges_via_api now retries
    # once on *transient* failures (server isError / exception); any entity that still
    # failed (None) is left unset here and backfilled to cold_start by the no-None
    # block below — never an SDK-guessed count.
    #
    # NOTE (honest reroute): classify_by_edge_count has no 'unknown' bucket, so a count
    # that fails even after the retry → edge_count=0 → cold_start, and route_after_triage
    # sends it to the cold-start analogue branch instead of direct_kg. A genuinely
    # well-characterized entity whose count failed is therefore downgraded. This is
    # PRE-EXISTING behavior (the prior default was already cold_start); the migration
    # does not worsen it, it just reaches the same default without a fabricated number.
    model_usages: list = []
    if tier1_failed_indices:
        for idx in tier1_failed_indices:
            entity = valid_entities[idx]
            logger.info(
                "FALLBACK_EVENT node=triage entity=%s curie=%s "
                "reason=tier1_edge_count_failed action=default_cold_start tier=2_dropped",
                getattr(entity, "raw_name", str(entity)),
                getattr(entity, "curie", "unknown"),
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
        "Completed triage in %.1fs — well_char=%d, moderate=%d, sparse=%d, cold_start=%d "
        "(tier1_ok=%d, tier1_failed=%d)",
        duration, len(well_characterized), len(moderate), len(sparse), len(cold_start),
        tier1_success, len(valid_entities) - tier1_success
    )

    result = {
        "novelty_scores": final_scores,
        "well_characterized_curies": well_characterized,
        "moderate_curies": moderate,
        "sparse_curies": sparse,
        "cold_start_curies": cold_start,
        "errors": errors,
    }
    if model_usages:
        result["model_usages"] = model_usages
    return result
