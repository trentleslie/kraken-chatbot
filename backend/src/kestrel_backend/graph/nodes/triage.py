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
from ..pipeline_config import get_pipeline_config
from ..state import DiscoveryState, NoveltyScore, EntityResolution
from ..state_contracts import validate_state, TriageInput, TriageOutput

logger = logging.getLogger(__name__)


# Classification thresholds
THRESHOLD_WELL_CHARACTERIZED = 200
THRESHOLD_MODERATE = 20
THRESHOLD_SPARSE = 1

# Edge-count retry: a couple of attempts with a short backoff cover transient Kestrel hiccups
# (server isError / timeout) without re-firing instantly into the same load (plan 2026-06-23-001).
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 0.5


async def count_edges_via_api(entity: EntityResolution, sem: asyncio.Semaphore) -> NoveltyScore | None:
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

    max_attempts = _MAX_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            # Call one_hop_query with preview mode - returns counts instead of full data.
            # The semaphore bounds in-flight Kestrel calls; it wraps only the call (not the retry
            # backoff) so a sleeping retry releases its slot to other entities.
            async with sem:
                result = await call_kestrel_tool("one_hop_query", {
                    "start_node_ids": curie,
                    "mode": "preview",
                    "limit": 10000,  # High limit to get accurate count
                })

            is_error = result.get("isError", False)
            content = result.get("content", [])

            if is_error:
                # Transient server-side error — back off, retry, then give up.
                logger.debug(
                    "Tier 1 triage '%s': API isError (attempt %d/%d)",
                    curie, attempt + 1, max_attempts,
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(_RETRY_BACKOFF_S)
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
            # Transient (timeout / connection) — back off, retry, then give up.
            logger.warning(
                "Tier 1 triage '%s': Exception (attempt %d/%d) - %s",
                curie, attempt + 1, max_attempts, str(e),
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(_RETRY_BACKOFF_S)
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

    # Run API calls concurrently but BOUNDED — an unbounded fan-out over hundreds of entities
    # thundering-herds Kestrel into timeouts (plan 2026-06-23-001). Per-invocation semaphore.
    sem = asyncio.Semaphore(get_pipeline_config().triage.kestrel_concurrency)
    tier1_results = await asyncio.gather(
        *[count_edges_via_api(e, sem) for e in valid_entities],
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

    # ========== Measurement failures reroute to direct-KG, never silent cold_start ==========
    # count_edges_via_api retries transient failures (server isError / exception) with backoff;
    # an entity whose count still cannot be MEASURED (None) is NOT a genuine 0-edge entity. A real
    # 0 returns a NoveltyScore (edge_count=0 -> cold_start); only a measurement failure is None.
    # Routing a measurement failure to cold_start would make a Kestrel overload read as "no KG
    # presence" (the silent-degradation anti-pattern, cf. synthesis overflow, PR #85). Instead we
    # route it to the direct-KG path via the `moderate` bucket and emit a visible marker.
    model_usages: list = []
    if tier1_failed_indices:
        for idx in tier1_failed_indices:
            entity = valid_entities[idx]
            logger.info(
                "FALLBACK_EVENT node=triage entity=%s curie=%s "
                "reason=tier1_edge_count_failed action=reroute_direct_kg_moderate",
                getattr(entity, "raw_name", str(entity)),
                getattr(entity, "curie", "unknown"),
            )

    # Ensure no None values; a None means the count FAILED (not measured 0) -> moderate + marker.
    final_scores = []
    for i, s in enumerate(all_scores):
        if s is None:
            ent = valid_entities[i]
            cur = ent.curie or ent.raw_name
            final_scores.append(NoveltyScore(
                curie=cur,
                raw_name=ent.raw_name,
                edge_count=0,
                classification="moderate",
            ))
            errors.append(
                f"triage: edge-count failed for {cur} ({ent.raw_name}); "
                "routed to direct-KG (moderate)"
            )
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

    # Unit 0b — forward triage-outcome counter (Decision 3). The human log line above is for
    # operators; this machine-parseable record lets us size the production query mix (what fraction
    # of runs produce speculative sparse/cold_start hypotheses vs. well-characterized-only) once real
    # traffic accrues, so the well-characterized latency cost of ground-before-synthesis becomes an
    # evidenced choice. `produces_speculative` is the routing-relevant predicate (sparse OR cold_start
    # entities are what generate cold-start hypotheses). One structured line per run; no new table.
    logger.info(
        "triage_outcome %s",
        json.dumps({
            "event": "triage_outcome",
            "well_characterized": len(well_characterized),
            "moderate": len(moderate),
            "sparse": len(sparse),
            "cold_start": len(cold_start),
            "tier1_ok": tier1_success,
            "tier1_failed": len(valid_entities) - tier1_success,
            "produces_speculative": bool(sparse or cold_start),
            "duration_seconds": round(duration, 2),
        }),
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
