"""
Entity Resolution Node: Parallel entity name resolution using Kestrel KG.

This node resolves raw entity names to knowledge graph identifiers (CURIEs)
by querying the Kestrel MCP server in parallel batches.

Uses the Claude Agent SDK with McpSSEServerConfig for direct HTTP communication.
"""

import asyncio
import json
import re
from typing import Any

from ..state import DiscoveryState, EntityResolution

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpSSEServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Kestrel MCP server configuration
KESTREL_URL = "https://kestrel.nathanpricelab.com/mcp"

# Concise prompt for entity resolution
RESOLUTION_PROMPT = """You are an entity resolver for biomedical knowledge graphs.
Given an entity name, use hybrid_search to find it in the Kestrel knowledge graph.

Return ONLY a valid JSON object with this exact structure (no other text):
{"curie": "PREFIX:ID", "name": "Canonical Name", "category": "biolink:Category", "confidence": 0.95}

If the entity cannot be found, return:
{"curie": null, "name": null, "category": null, "confidence": 0.0}

Be extremely concise. No explanations or additional text."""

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


async def resolve_single_entity(entity: str) -> EntityResolution:
    """
    Resolve a single entity name to a CURIE using Claude Agent SDK.

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
        kestrel_config = McpSSEServerConfig(
            type="sse",
            url=KESTREL_URL,
        )

        options = ClaudeAgentOptions(
            system_prompt=RESOLUTION_PROMPT,
            allowed_tools=[
                "mcp__kestrel__hybrid_search",
                "mcp__kestrel__text_search",
                "mcp__kestrel__get_nodes",
            ],
            mcp_servers={"kestrel": kestrel_config},
            max_turns=2,
            permission_mode="bypassPermissions",
        )

        result_text_parts = []
        async for event in query(prompt=f"Resolve: {entity}", options=options):
            if hasattr(event, 'content'):
                for block in event.content:
                    if hasattr(block, 'text'):
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

    Returns resolved_entities list and any errors encountered.
    """
    entities = state.get("raw_entities", [])

    if not entities:
        return {
            "resolved_entities": [],
            "errors": [],
        }

    all_results: list[EntityResolution] = []
    errors: list[str] = []

    # Process in batches for controlled parallelism
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

    return {
        "resolved_entities": all_results,
        "errors": errors,
    }
