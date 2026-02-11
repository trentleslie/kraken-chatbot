"""
Entity Resolution Node: Two-tier entity name resolution using Kestrel KG.

This node resolves raw entity names to knowledge graph identifiers (CURIEs)
using a two-tier approach:

Tier 1 (API): Direct hybrid_search calls via Kestrel HTTP client (~100ms each)
  - Fast, reliable - uses Kestrel's ranking which is typically accurate
  - Top result with score > 0.6 is accepted

Tier 2 (LLM): Falls back to Claude Agent SDK for ambiguous cases
  - Handles complex synonyms, abbreviations, partial matches
  - More expensive but can reason about alternatives

This hybrid approach optimizes for speed while maintaining accuracy.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ..state import DiscoveryState, EntityResolution

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpStdioServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP command for stdio-based server
KESTREL_COMMAND = "uvx"
KESTREL_ARGS = ["mcp-client-kestrel"]

# Semaphore to serialize SDK calls and prevent concurrent CLI spawn issues
SDK_SEMAPHORE = asyncio.Semaphore(1)

# Enhanced prompt for entity resolution with retry strategies
RESOLUTION_PROMPT = """You are an expert biomedical entity resolver for the Kestrel knowledge graph.

## Your Task
Resolve the given entity name to its canonical CURIE (Compact URI) identifier.

## Resolution Strategy (try in order)
1. **Exact match**: Use hybrid_search with the exact entity name first
2. **Synonym search**: If no result, try common synonyms:
   - Chemical abbreviations: "BHBA" → "beta-hydroxybutyrate" → "3-hydroxybutyric acid"
   - Gene aliases: Many genes have multiple symbols
3. **Gene symbol handling**: For all-caps 2-6 character names (like KIF6, NLGN1, TP53):
   - These are likely gene symbols → search directly, expect NCBIGene CURIEs
4. **Metabolite variants**: Try without prefixes/suffixes:
   - "N-lactoyl phenylalanine" → also try "lactoylphenylalanine"
   - "16-hydroxypalmitate" → also try "hydroxypalmitate"
5. **Partial matching**: Use text_search for fuzzy matches if hybrid_search fails

## Important
- Always try at least 2-3 search variations before giving up
- Gene symbols (all caps, 2-6 chars) are common - search as-is
- Metabolites may need chemical name variations

## Output Format
Return ONLY valid JSON (no other text):
{"curie": "PREFIX:ID", "name": "Canonical Name", "category": "biolink:Category", "confidence": 0.95}

If truly not found after trying all strategies:
{"curie": null, "name": null, "category": null, "confidence": 0.0}"""

# More aggressive retry prompt for failed entities
RETRY_PROMPT = """You are an expert biomedical entity resolver. This entity FAILED initial resolution - try harder!

## Aggressive Search Strategies
1. **Alternative spellings**: Try with/without hyphens, spaces, prefixes (N-, 16-, etc.)
2. **Chemical synonyms**: Search IUPAC names, common names, and abbreviations
3. **Partial matches**: Use text_search with key substrings
4. **Category hints**: If it looks like a gene (all caps, 2-6 chars), search gene databases
5. **Metabolite variations**: Strip numeric prefixes, try base compound names

## Examples of successful resolutions
- "N-lactoyl phenylalanine" → try "lactoylphenylalanine", "lactoyl-phenylalanine"
- "16-hydroxypalmitate" → try "hydroxypalmitate", "hydroxypalmitic acid"
- "hexadecanedioate" → try "hexadecanedioic acid", "C16-DC"
- "KIF6" → search as gene symbol directly (NCBIGene)

## Output
Return ONLY valid JSON:
{"curie": "PREFIX:ID", "name": "Canonical Name", "category": "biolink:Category", "confidence": 0.95}

If truly not found: {"curie": null, "name": null, "category": null, "confidence": 0.0}"""

# Batch size for parallel resolution
BATCH_SIZE = 6

# Minimum score threshold for Tier 1 API resolution
# Below this score, entities fall through to Tier 2 LLM
TIER1_MIN_SCORE = 0.6


async def resolve_via_api(entity: str) -> EntityResolution | None:
    """
    Tier 1: Attempt to resolve entity via direct Kestrel API call.

    Uses hybrid_search and takes the top-scored result if confidence is high enough.
    Returns None if resolution fails or confidence is too low (triggering Tier 2).

    Confidence mapping from hybrid_search score:
    - score > 1.5 → confidence 0.95 (exact + vector match)
    - score > 1.0 → confidence 0.90
    - score > 0.8 → confidence 0.80
    - score > 0.6 → confidence 0.70
    - score < 0.6 → fall through to Tier 2 (returns None)
    """
    try:
        # Call hybrid_search directly - parameter is 'search_text' not 'query'
        result = await call_kestrel_tool("hybrid_search", {
            "search_text": entity,
            "limit": 1,  # Only need top result
        })

        # Debug logging - show raw API response
        is_error = result.get("isError", False)
        content = result.get("content", [])
        logger.info(
            "Tier 1 '%s': isError=%s, content_len=%d",
            entity, is_error, len(content)
        )

        if is_error:
            logger.debug("Tier 1 '%s': API returned error", entity)
            return None

        # Parse the search results
        if not content:
            logger.info("Tier 1 '%s': No content in response", entity)
            return None

        # Extract JSON from content
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.info("Tier 1 '%s': Could not parse JSON response", entity)
            return None

        # Response format is {search_text: [results]} - extract our entity's results
        if isinstance(data, dict):
            # Try exact key match first, then case-insensitive
            results = data.get(entity) or data.get(entity.lower()) or []
            if not results and len(data) == 1:
                # Single result dict - take the first (and only) value
                results = list(data.values())[0]
        else:
            results = data

        if not results:
            logger.info("Tier 1 '%s': No search results", entity)
            return None

        # Take top result
        top = results[0]
        score = float(top.get("score", 0))
        curie = top.get("id") or top.get("curie")
        name = top.get("name") or top.get("label")
        # categories is a list in the API response
        categories = top.get("categories", [])
        category = categories[0] if categories else top.get("category")

        # Map score to confidence
        if score > 1.5:
            confidence = 0.95
        elif score > 1.0:
            confidence = 0.90
        elif score > 0.8:
            confidence = 0.80
        elif score > TIER1_MIN_SCORE:
            confidence = 0.70
        else:
            # Score too low - fall through to Tier 2
            logger.info(
                "Tier 1 '%s': Score %.2f below threshold %.2f, falling through to Tier 2",
                entity, score, TIER1_MIN_SCORE
            )
            return None

        logger.info(
            "Tier 1 '%s': resolved to %s (score=%.2f, confidence=%.2f)",
            entity, curie, score, confidence
        )

        return EntityResolution(
            raw_name=entity,
            curie=curie,
            resolved_name=name,
            category=category,
            confidence=confidence,
            method="api",  # Mark as API-resolved
        )

    except Exception as e:
        logger.warning("Tier 1 '%s': Exception - %s", entity, str(e))
        return None


def parse_resolution_result(entity: str, result_text: str) -> EntityResolution:
    """
    Parse LLM response into EntityResolution object.

    Handles JSON extraction from potentially noisy LLM output.
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]+\}', result_text)
        if json_match:
            data = json.loads(json_match.group())

            curie = data.get("curie")
            name = data.get("name")
            category = data.get("category")
            confidence = float(data.get("confidence", 0.0))

            # Determine resolution method based on confidence
            if curie and confidence >= 0.9:
                method = "exact"
            elif curie and confidence >= 0.7:
                method = "fuzzy"
            elif curie:
                method = "semantic"
            else:
                method = "failed"

            return EntityResolution(
                raw_name=entity,
                curie=curie,
                resolved_name=name,
                category=category,
                confidence=confidence,
                method=method,
            )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        pass

    # Fallback for parse failures
    return EntityResolution(
        raw_name=entity,
        curie=None,
        resolved_name=None,
        category=None,
        confidence=0.0,
        method="failed",
    )


async def resolve_single_entity(entity: str, is_retry: bool = False) -> EntityResolution:
    """
    Resolve a single entity name to a CURIE using Claude Agent SDK.

    Args:
        entity: The entity name to resolve
        is_retry: If True, use more aggressive retry prompt

    Returns EntityResolution with method="failed" on any error.
    """
    if not HAS_SDK:
        # SDK not available - return mock for testing
        return EntityResolution(
            raw_name=entity,
            curie=None,
            resolved_name=None,
            category=None,
            confidence=0.0,
            method="failed",
        )

    try:
        async with SDK_SEMAPHORE:
            kestrel_config = McpStdioServerConfig(
                type="stdio",
                command=KESTREL_COMMAND,
                args=KESTREL_ARGS,
            )

            # Use more aggressive prompt for retry attempts
            prompt_to_use = RESOLUTION_PROMPT
            if is_retry:
                prompt_to_use = RETRY_PROMPT

            options = ClaudeAgentOptions(
                system_prompt=prompt_to_use,
                allowed_tools=[
                    "mcp__kestrel__hybrid_search",
                    "mcp__kestrel__text_search",
                    "mcp__kestrel__get_nodes",
                    "mcp__kestrel__get_node_info",
                    "mcp__kestrel__get_neighbors",
                ],
                mcp_servers={"kestrel": kestrel_config},
                max_turns=5,  # Increased from 2 to allow iterative search refinement
                permission_mode="bypassPermissions",
                max_buffer_size=10 * 1024 * 1024,  # 10MB buffer for large KG responses
            )

            result_text_parts = []
            async for event in query(prompt=f"Resolve: {entity}", options=options):
                if hasattr(event, "content"):
                    for block in event.content:
                        if hasattr(block, "text"):
                            result_text_parts.append(block.text)

            result_text = "".join(result_text_parts)
            return parse_resolution_result(entity, result_text)

    except Exception as e:
        return EntityResolution(
            raw_name=entity,
            curie=None,
            resolved_name=None,
            category=None,
            confidence=0.0,
            method="failed",
        )


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Resolve all raw entities to knowledge graph identifiers.

    Implements two-tier resolution:

    Tier 1 (API): Try direct hybrid_search for all entities in parallel
      - Fast (~100ms each), uses Kestrel's reliable ranking
      - Accepts top result if score > 0.6

    Tier 2 (LLM): Falls back to Claude Agent SDK for failed entities
      - Handles complex synonyms, abbreviations, partial matches
      - Uses standard prompt first, then aggressive retry prompt

    Returns resolved_entities list and any errors encountered.
    """
    entities = state.get("raw_entities", [])
    logger.info("Starting entity_resolution with %d entities", len(entities))
    start = time.time()

    if not entities:
        logger.info("No entities to resolve, skipping")
        return {
            "resolved_entities": [],
            "errors": [],
        }

    all_results: list[EntityResolution | None] = [None] * len(entities)
    errors: list[str] = []

    # ========== TIER 1: API Resolution ==========
    tier1_start = time.time()
    logger.info("Tier 1 (API): Attempting direct resolution for %d entities", len(entities))

    # Run all API calls in parallel (they're fast and independent)
    tier1_results = await asyncio.gather(
        *[resolve_via_api(e) for e in entities],
        return_exceptions=True,
    )

    tier1_resolved = 0
    tier1_failed_indices = []

    for i, (entity, result) in enumerate(zip(entities, tier1_results)):
        if isinstance(result, Exception):
            logger.debug("Tier 1 '%s': Exception - %s", entity, str(result))
            tier1_failed_indices.append(i)
        elif result is not None:
            # Successfully resolved via API
            all_results[i] = result
            tier1_resolved += 1
        else:
            # Returned None - needs Tier 2
            tier1_failed_indices.append(i)

    tier1_duration = time.time() - tier1_start
    logger.info(
        "Tier 1 (API) resolved %d/%d entities in %.1fs",
        tier1_resolved, len(entities), tier1_duration
    )

    # ========== TIER 2: LLM Resolution ==========
    if tier1_failed_indices and HAS_SDK:
        tier2_start = time.time()
        failed_entities = [entities[i] for i in tier1_failed_indices]
        logger.info(
            "Tier 2 (LLM): Processing %d entities that failed Tier 1",
            len(failed_entities)
        )

        # First pass: Standard resolution in batches
        tier2_results = []
        for batch in chunk(failed_entities, BATCH_SIZE):
            batch_results = await asyncio.gather(
                *[resolve_single_entity(e) for e in batch],
                return_exceptions=True,
            )
            tier2_results.extend(batch_results)

        # Map results back to all_results
        for idx, result in zip(tier1_failed_indices, tier2_results):
            if isinstance(result, Exception):
                errors.append(f"Resolution failed for '{entities[idx]}': {str(result)}")
                all_results[idx] = EntityResolution(
                    raw_name=entities[idx],
                    curie=None,
                    resolved_name=None,
                    category=None,
                    confidence=0.0,
                    method="failed",
                )
            else:
                all_results[idx] = result

        # Second pass: Aggressive retry for still-failed entities
        still_failed_indices = [i for i in tier1_failed_indices if all_results[i] and not all_results[i].curie]

        if still_failed_indices:
            still_failed_entities = [entities[i] for i in still_failed_indices]
            logger.info(
                "Tier 2 retry: %d entities still unresolved, trying aggressive prompt",
                len(still_failed_entities)
            )

            retry_batch_size = max(2, BATCH_SIZE // 2)
            retry_results = []

            for batch in chunk(still_failed_entities, retry_batch_size):
                batch_results = await asyncio.gather(
                    *[resolve_single_entity(e, is_retry=True) for e in batch],
                    return_exceptions=True,
                )
                retry_results.extend(batch_results)

            # Merge successful retries back
            for idx, retry_result in zip(still_failed_indices, retry_results):
                if isinstance(retry_result, EntityResolution) and retry_result.curie:
                    all_results[idx] = retry_result

        tier2_duration = time.time() - tier2_start
        tier2_resolved = sum(1 for i in tier1_failed_indices if all_results[i] and all_results[i].curie)
        logger.info(
            "Tier 2 (LLM) resolved %d/%d entities in %.1fs",
            tier2_resolved, len(tier1_failed_indices), tier2_duration
        )
    elif tier1_failed_indices:
        # SDK not available - mark remaining as failed
        for idx in tier1_failed_indices:
            all_results[idx] = EntityResolution(
                raw_name=entities[idx],
                curie=None,
                resolved_name=None,
                category=None,
                confidence=0.0,
                method="failed",
            )

    # Ensure no None values remain
    final_results = []
    for i, r in enumerate(all_results):
        if r is None:
            final_results.append(EntityResolution(
                raw_name=entities[i],
                curie=None,
                resolved_name=None,
                category=None,
                confidence=0.0,
                method="failed",
            ))
        else:
            final_results.append(r)

    # Calculate final stats
    resolved = [r for r in final_results if r.curie]
    failed = [r for r in final_results if not r.curie]
    duration = time.time() - start
    rate = 100 * len(resolved) / len(final_results) if final_results else 0

    # Log resolution method breakdown
    api_resolved = sum(1 for r in final_results if r.method == "api")
    llm_resolved = len(resolved) - api_resolved

    if failed:
        failed_names = [r.raw_name for r in failed[:5]]
        logger.warning(
            "Failed to resolve %d entities: %s%s",
            len(failed), failed_names, "..." if len(failed) > 5 else ""
        )

    logger.info(
        "Completed entity_resolution in %.1fs — resolved=%d (api=%d, llm=%d), failed=%d (%.0f%%)",
        duration, len(resolved), api_resolved, llm_resolved, len(failed), rate
    )

    return {
        "resolved_entities": final_results,
        "errors": errors,
    }
