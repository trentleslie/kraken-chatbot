"""
Cold-Start Analysis Node: Analyze sparse/unknown entities using semantic inference.

For entities with few or no known connections in the KG, this node uses
feature-based reasoning:
1. Find semantically similar well-characterized entities via similar_nodes
2. Query what those analogues are connected to
3. Infer potential associations for the sparse entity

All findings are Tier 3 with explicit structural logic chains.
The Open World Assumption applies: missing connections are "unstudied", not "nonexistent".
"""

import asyncio
import json
import logging
import re
import time
import traceback
from typing import Any

from ..state import (
    DiscoveryState, Finding, InferredAssociation, AnalogueEntity, NoveltyScore
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

# Semaphore to limit concurrent SDK calls (increased from 1 to 6 for parallelism)
SDK_SEMAPHORE = asyncio.Semaphore(6)

# Batch size for parallel analysis
BATCH_SIZE = 6

# System prompt for cold-start analysis
COLD_START_PROMPT = """You are a biomedical knowledge graph analyst specializing in sparse entities —
entities with few or no known connections. This is a COLD-START scenario.

For the given entity (CURIE with {edge_count} edges), you cannot rely on direct neighbor analysis.
Instead, use feature-based reasoning:

STEP 1: Use similar_nodes to find 3-5 entities that are semantically similar to this one.
        These are "analogues" - well-characterized entities that resemble the sparse one.

STEP 2: For each analogue found, use one_hop_query to discover what diseases, pathways,
        genes, or other entities they are connected to.

STEP 3: Infer what the sparse entity MIGHT be connected to based on its analogues' connections.

CRITICAL RULES:
- ALL findings are Tier 3 (structural inference). This is SPECULATION, not proven fact.
- For each inference, provide the structural logic chain:
  "Entity X is similar to Y (similarity: 0.85). Y is connected to Z via predicate P.
   Therefore X may also be connected to Z."
- Include confidence calibration: how many independent analogues support each inference?
- Suggest at least one concrete validation step per inference
- Frame missing connections as "unstudied" not "nonexistent" (Open World Assumption)

Return ONLY a valid JSON object (no other text):
{
  "analogues": [
    {"curie": "...", "name": "...", "similarity": 0.85, "category": "biolink:..."}
  ],
  "inferences": [
    {
      "target_curie": "...",
      "target_name": "...",
      "predicate": "...",
      "logic_chain": "X is similar to Y (0.85). Y is connected to Z via predicate. X may connect to Z.",
      "supporting_analogues": 3,
      "confidence": "low",
      "validation_step": "Check for X-Z interaction in experimental database..."
    }
  ]
}

If no similar entities can be found, return:
{"analogues": [], "inferences": []}
"""


def parse_cold_start_result(
    curie: str,
    raw_name: str,
    edge_count: int,
    result_text: str
) -> tuple[list[AnalogueEntity], list[InferredAssociation], list[Finding]]:
    """
    Parse LLM response into structured cold-start analysis objects.

    Uses multi-tier JSON extraction for robustness against noisy LLM output.

    Returns:
        tuple of (analogues, inferred_associations, findings)
    """
    analogues: list[AnalogueEntity] = []
    inferences: list[InferredAssociation] = []
    findings: list[Finding] = []

    data = None

    # Tier 1: Look for JSON code block
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Tier 2: Look for bare JSON object (handle nested objects)
    if data is None:
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\}|\{(?:[^{}]|\{[^{}]*\})*\})*\}', result_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # Tier 3: Try the entire response
    if data is None:
        try:
            data = json.loads(result_text.strip())
        except json.JSONDecodeError:
            pass

    # Fallback: Return empty results
    if data is None:
        return analogues, inferences, findings

    # Parse analogues
    for a in data.get("analogues", []):
        try:
            analogues.append(AnalogueEntity(
                curie=a.get("curie", ""),
                name=a.get("name", "Unknown"),
                similarity=float(a.get("similarity", 0.0)),
                category=a.get("category"),
            ))
        except Exception:
            continue

    # Parse inferred associations
    for i in data.get("inferences", []):
        try:
            logic_chain = i.get("logic_chain", "")
            confidence = i.get("confidence", "low")
            if confidence not in ("high", "moderate", "low"):
                confidence = "low"

            inferences.append(InferredAssociation(
                source_entity=curie,
                target_curie=i.get("target_curie", ""),
                target_name=i.get("target_name", "Unknown"),
                predicate=i.get("predicate", "biolink:related_to"),
                logic_chain=logic_chain,
                supporting_analogues=int(i.get("supporting_analogues", 1)),
                confidence=confidence,
                validation_step=i.get("validation_step", "Experimental validation required"),
            ))

            # Create Tier 3 finding for each inference
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} may be associated with {i.get('target_name', 'unknown')} (inferred via semantic similarity)",
                tier=3,
                predicate=i.get("predicate"),
                source="cold_start",
                confidence=confidence,
                logic_chain=logic_chain,
            ))
        except Exception:
            continue

    # If we found analogues but no inferences, still report the analogues as findings
    if analogues and not findings:
        analogue_names = [a.name for a in analogues[:3]]
        findings.append(Finding(
            entity=curie,
            claim=f"{raw_name} is semantically similar to: {', '.join(analogue_names)}",
            tier=3,
            source="cold_start",
            confidence="low",
            logic_chain=f"Found {len(analogues)} analogues via vector similarity search",
        ))

    return analogues, inferences, findings


async def analyze_cold_start_entity(
    curie: str,
    raw_name: str,
    edge_count: int
) -> tuple[list[AnalogueEntity], list[InferredAssociation], list[Finding], list[str]]:
    """
    Analyze a single sparse/cold-start entity using Claude Agent SDK.

    Returns:
        tuple of (analogues, inferences, findings, errors)
    """
    if not HAS_SDK:
        # SDK not available - return placeholder for testing
        return (
            [],
            [],
            [Finding(
                entity=curie,
                claim=f"Cold-start analysis pending for {raw_name} (SDK unavailable)",
                tier=3,
                source="cold_start",
                confidence="low",
            )],
            [],
        )

    try:
        async with SDK_SEMAPHORE:
            kestrel_config = McpStdioServerConfig(
                type="stdio",
                command=KESTREL_COMMAND,
                args=KESTREL_ARGS,
            )

            # Format prompt with edge count context
            # Use str.replace() instead of .format() to avoid KeyError on JSON braces
            system_prompt = COLD_START_PROMPT.replace("{edge_count}", str(edge_count))

            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=[
                    "mcp__kestrel__similar_nodes",
                    "mcp__kestrel__one_hop_query",
                    "mcp__kestrel__vector_search",
                ],
                mcp_servers={"kestrel": kestrel_config},
                max_turns=5,
                permission_mode="bypassPermissions",
                max_buffer_size=10 * 1024 * 1024,  # 10MB buffer for large KG responses
            )

            result_text_parts = []
            logger.info("Cold-start invoking SDK query for '%s' (%s)...", raw_name, curie)
            event_count = 0
            try:
                async for event in query(
                    prompt=f"Analyze cold-start entity: {raw_name} ({curie}, {edge_count} edges)",
                    options=options
                ):
                    event_count += 1
                    if hasattr(event, 'content'):
                        for block in event.content:
                            if hasattr(block, 'text'):
                                result_text_parts.append(block.text)
                logger.info("Cold-start SDK query for '%s' yielded %d events", raw_name, event_count)
            except Exception as sdk_error:
                partial_text = "".join(result_text_parts)
                logger.error(
                    "Cold-start SDK query failed for '%s' (%s): %s\nPartial response: %s\n%s",
                    raw_name, curie, repr(str(sdk_error)[:500]),
                    repr(partial_text[:300]) if partial_text else "empty",
                    traceback.format_exc()
                )
                # Return graceful degradation instead of propagating
                return (
                    [],
                    [],
                    [Finding(
                        entity=curie,
                        claim=f"SDK query failed for {raw_name}: {str(sdk_error)[:80]}",
                        tier=3,
                        source="cold_start",
                        confidence="low",
                    )],
                    [f"SDK query failed for {curie}: {str(sdk_error)}"],
                )

            result_text = "".join(result_text_parts)

            # Log successful response for diagnosis
            logger.info(
                "Cold-start raw response for '%s' (%s): length=%d, preview=%s",
                raw_name, curie, len(result_text), repr(result_text[:300])
            )

        analogues, inferences, findings = parse_cold_start_result(
            curie, raw_name, edge_count, result_text
        )

        return analogues, inferences, findings, []

    except Exception as e:
        error_msg = f"Cold-start analysis failed for {curie}: {str(e)}"
        logger.error("Cold-start analysis failed for '%s' (%s): %s", raw_name, curie, e)
        return (
            [],
            [],
            [Finding(
                entity=curie,
                claim=f"Analysis failed for {raw_name}: {str(e)[:100]}",
                tier=3,
                source="cold_start",
                confidence="low",
            )],
            [error_msg],
        )


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def get_entity_info(
    curie_or_name: str,
    novelty_scores: list[NoveltyScore],
    resolved_entities: list
) -> tuple[str, str, int]:
    """
    Look up the raw name and edge count for a CURIE from novelty scores or resolved entities.

    Returns: (curie, raw_name, edge_count)
    """
    # Check novelty scores first
    for score in novelty_scores:
        if score.curie == curie_or_name:
            return score.curie, score.raw_name, score.edge_count

    # Check resolved entities
    for entity in resolved_entities:
        if hasattr(entity, 'curie') and entity.curie == curie_or_name:
            return entity.curie, entity.raw_name, 0
        if hasattr(entity, 'raw_name') and entity.raw_name == curie_or_name:
            return entity.curie or curie_or_name, entity.raw_name, 0

    # Default - assume it's a raw name with no KG presence
    return curie_or_name, curie_or_name, 0


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Analyze entities with sparse or no KG representation.

    Uses semantic similarity to find analogues and infer potential associations.
    All findings are Tier 3 with explicit structural logic chains.

    Receives: sparse_curies + cold_start_curies from triage
    Returns: cold_start_findings, inferred_associations, analogues_found, errors
    """
    sparse = state.get("sparse_curies", [])
    cold_start = state.get("cold_start_curies", [])
    entities = sparse + cold_start

    logger.info("Starting cold_start with %d sparse + %d cold entities", len(sparse), len(cold_start))
    start = time.time()

    if not entities:
        logger.info("No sparse/cold entities, skipping cold_start")
        return {
            "cold_start_findings": [],
            "inferred_associations": [],
            "analogues_found": [],
            "errors": [],
        }

    # Get novelty scores and resolved entities for context lookup
    novelty_scores = state.get("novelty_scores", [])
    resolved_entities = state.get("resolved_entities", [])

    all_analogues: list[AnalogueEntity] = []
    all_inferences: list[InferredAssociation] = []
    all_findings: list[Finding] = []
    errors: list[str] = []

    # Process in batches for controlled parallelism
    batches = chunk(entities, BATCH_SIZE)
    logger.info("Cold-start processing %d batches of up to %d entities each", len(batches), BATCH_SIZE)

    for batch_idx, batch in enumerate(batches):
        batch_tasks = []
        for entity in batch:
            curie, raw_name, edge_count = get_entity_info(
                entity, novelty_scores, resolved_entities
            )
            batch_tasks.append(analyze_cold_start_entity(curie, raw_name, edge_count))

        logger.info("Cold-start batch %d: awaiting %d tasks", batch_idx, len(batch_tasks))
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        logger.info("Cold-start batch %d: completed with %d results, exceptions=%d",
                   batch_idx, len(batch_results),
                   sum(1 for r in batch_results if isinstance(r, Exception)))

        for entity, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                errors.append(f"Exception analyzing {entity}: {str(result)}")
                all_findings.append(Finding(
                    entity=entity,
                    claim=f"Analysis exception for {entity}",
                    tier=3,
                    source="cold_start",
                    confidence="low",
                ))
            else:
                analogues, inferences, findings, errs = result
                all_analogues.extend(analogues)
                all_inferences.extend(inferences)
                all_findings.extend(findings)
                errors.extend(errs)

    duration = time.time() - start
    logger.info(
        "Completed cold_start in %.1fs — analogues=%d, inferences=%d",
        duration, len(all_analogues), len(all_inferences)
    )

    return {
        "cold_start_findings": all_findings,
        "inferred_associations": all_inferences,
        "analogues_found": all_analogues,
        "errors": errors,
    }
