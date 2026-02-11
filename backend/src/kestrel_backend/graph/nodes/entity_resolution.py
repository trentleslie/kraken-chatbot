"""
Entity Resolution Node: Parallel entity name resolution using Kestrel KG.

This node resolves raw entity names to knowledge graph identifiers (CURIEs)
by querying the Kestrel MCP server in parallel batches.

Uses the Claude Agent SDK with McpStdioServerConfig for uvx-based MCP server.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

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

    Processes entities in parallel batches of BATCH_SIZE (6) to avoid
    overwhelming the Kestrel server while maintaining good throughput.

    Implements two-pass resolution:
    1. First pass: Standard resolution for all entities
    2. Second pass: Aggressive retry for failed entities with enhanced prompts

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

    all_results: list[EntityResolution] = []
    errors: list[str] = []

    # First pass: Standard resolution
    for batch in chunk(entities, BATCH_SIZE):
        batch_results = await asyncio.gather(
            *[resolve_single_entity(e) for e in batch],
            return_exceptions=True,
        )

        for entity, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                errors.append(f"Resolution failed for '{entity}': {str(result)}")
                all_results.append(EntityResolution(
                    raw_name=entity,
                    curie=None,
                    resolved_name=None,
                    category=None,
                    confidence=0.0,
                    method="failed",
                ))
            else:
                all_results.append(result)

    # Second pass: Retry failed entities with aggressive prompt
    failed_indices = [i for i, r in enumerate(all_results) if r.curie is None]

    if failed_indices and HAS_SDK:
        failed_entities = [all_results[i].raw_name for i in failed_indices]

        # Retry in smaller batches (more resources per entity)
        retry_batch_size = max(2, BATCH_SIZE // 2)
        retry_results = []

        for batch in chunk(failed_entities, retry_batch_size):
            batch_results = await asyncio.gather(
                *[resolve_single_entity(e, is_retry=True) for e in batch],
                return_exceptions=True,
            )
            retry_results.extend(batch_results)

        # Merge successful retries back into results
        for idx, retry_result in zip(failed_indices, retry_results):
            if isinstance(retry_result, EntityResolution) and retry_result.curie:
                all_results[idx] = retry_result

    # Calculate final stats
    resolved = [r for r in all_results if r.curie]
    failed = [r for r in all_results if not r.curie]
    duration = time.time() - start
    rate = 100 * len(resolved) / len(all_results) if all_results else 0

    if failed:
        failed_names = [r.raw_name for r in failed[:5]]
        logger.warning(
            "Failed to resolve %d entities: %s%s",
            len(failed), failed_names, "..." if len(failed) > 5 else ""
        )

    logger.info(
        "Completed entity_resolution in %.1fs — resolved=%d, failed=%d (%.0f%%)",
        duration, len(resolved), len(failed), rate
    )

    return {
        "resolved_entities": all_results,
        "errors": errors,
    }
