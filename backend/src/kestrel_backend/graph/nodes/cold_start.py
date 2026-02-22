"""
Cold-Start Analysis Node: Analyze sparse/unknown entities using semantic inference.

For entities with few or no known connections in the KG, this node uses
feature-based reasoning:
1. Find semantically similar well-characterized entities via similar_nodes (HTTP API)
2. Query what those analogues are connected to via one_hop_query (HTTP API)
3. Pass real KG data to SDK for inference reasoning (no MCP tools needed)

All findings are Tier 3 with explicit structural logic chains.
The Open World Assumption applies: missing connections are "unstudied", not "nonexistent".

NOTE: This node uses direct HTTP API calls to Kestrel (via call_kestrel_tool) instead of
trying to expose tools via MCP. This is more reliable because:
- The mcp-client-kestrel package doesn't exist on PyPI
- Entity resolution already uses this pattern successfully
- Direct control over KG queries ensures deterministic data retrieval
- SDK agent reasons over REAL KG data instead of hallucinating from training knowledge
"""

import asyncio
import json
import logging
import re
import time
import traceback
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ..state import (
    DiscoveryState, Finding, InferredAssociation, AnalogueEntity, NoveltyScore
)

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# Semaphore to limit concurrent SDK calls
SDK_SEMAPHORE = asyncio.Semaphore(6)

# Batch size for parallel analysis
BATCH_SIZE = 3

# Timeout for SDK inference query per entity (2 minutes - reduced since no tool calls)
SDK_INFERENCE_TIMEOUT = 120

# Number of similar entities to retrieve for each sparse entity
ANALOGUE_LIMIT = 5


# System prompt for inference reasoning (no tools needed - KG data provided in prompt)
INFERENCE_PROMPT = """You are a biomedical knowledge graph analyst specializing in inference.

You will be given:
1. A sparse entity (few or no known KG connections)
2. Similar entities (analogues) found via vector similarity search
3. The connections each analogue has in the knowledge graph

Your task is to infer what the sparse entity MIGHT be connected to based on its analogues' connections.

CRITICAL RULES:
- Use ONLY the KG data provided in the prompt for inferences
- Each inference must cite which analogue connection supports it
- ALL findings are Tier 3 (structural inference) - SPECULATION requiring validation
- Include confidence calibration based on how many analogues support each inference
- Frame missing connections as "unstudied" not "nonexistent" (Open World Assumption)

Return ONLY a valid JSON object (no other text):
{
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

If no valid inferences can be made from the provided data, return:
{"inferences": []}
"""


async def get_similar_entities(curie: str, limit: int = ANALOGUE_LIMIT) -> list[dict]:
    """
    Query Kestrel similar_nodes via HTTP API.

    Returns list of similar entities with curie, name, similarity score, and category.
    """
    try:
        result = await call_kestrel_tool("similar_nodes", {
            "node_id": curie,
            "limit": limit,
        })

        if result.get("isError"):
            logger.warning("similar_nodes failed for %s: %s", curie, result)
            return []

        content = result.get("content", [])
        if not content:
            return []

        # Parse the JSON response
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse similar_nodes response for %s", curie)
            return []

        # Check for embedded error in response data
        if isinstance(data, dict) and data.get("error"):
            logger.warning(
                "similar_nodes API error for %s: %s",
                curie, data.get("message", "unknown error")
            )
            return []

        # Extract similar entities
        # Response format: {"CURIE": [...]} where key is the queried CURIE
        similar = []
        if isinstance(data, list):
            results_list = data
        elif isinstance(data, dict):
            # Try CURIE as key first (actual API response format), then fallback keys
            results_list = data.get(curie, data.get("results", data.get("similar_nodes", [])))
        else:
            results_list = []

        for item in results_list:
            if isinstance(item, dict):
                similar.append({
                    "curie": item.get("id") or item.get("curie", ""),
                    "name": item.get("name") or item.get("label", "Unknown"),
                    "similarity": float(item.get("similarity") or item.get("score", 0.0)),
                    "category": item.get("category") or (item.get("categories", [""])[0] if item.get("categories") else ""),
                })

        logger.info("similar_nodes for %s: found %d analogues", curie, len(similar))
        return similar

    except Exception as e:
        logger.error("Exception in get_similar_entities for %s: %s", curie, e)
        return []


async def get_entity_connections(curie: str) -> dict:
    """
    Query Kestrel one_hop_query via HTTP API.

    Returns dict with edges grouped by predicate and direction.
    """
    try:
        result = await call_kestrel_tool("one_hop_query", {
            "start_node_ids": curie,
        })

        if result.get("isError"):
            logger.warning("one_hop_query failed for %s: %s", curie, result)
            return {"edges": [], "summary": "Query failed"}

        content = result.get("content", [])
        if not content:
            return {"edges": [], "summary": "No content"}

        # Parse the JSON response
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse one_hop_query response for %s", curie)
            return {"edges": [], "summary": "Parse error"}

        # Check for embedded error in response data
        if isinstance(data, dict) and data.get("error"):
            error_msg = data.get("message", "unknown error")
            logger.warning("one_hop_query API error for %s: %s", curie, error_msg)
            return {"edges": [], "summary": f"API error: {error_msg}"}

        # Extract edges, handling various response formats
        edges = []
        if isinstance(data, list):
            edges = data
        elif isinstance(data, dict):
            edges = data.get("edges", data.get("results", []))

        # Summarize connections by predicate
        predicate_counts: dict[str, int] = {}
        for edge in edges:
            pred = edge.get("predicate", "unknown")
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

        summary = ", ".join(f"{pred}: {count}" for pred, count in sorted(predicate_counts.items(), key=lambda x: -x[1])[:5])

        logger.info("one_hop_query for %s: %d edges (%s)", curie, len(edges), summary[:100])
        return {
            "edges": edges[:50],  # Limit to 50 edges to keep prompt size manageable
            "summary": summary or "No edges found",
            "total_count": len(edges),
        }

    except Exception as e:
        logger.error("Exception in get_entity_connections for %s: %s", curie, e)
        return {"edges": [], "summary": f"Error: {str(e)}"}


def format_inference_prompt(
    curie: str,
    raw_name: str,
    edge_count: int,
    analogues: list[dict],
    analogue_connections: dict[str, dict]
) -> str:
    """
    Format prompt with actual KG data for inference reasoning.

    This ensures the SDK agent reasons over REAL knowledge graph data
    rather than hallucinating connections from training knowledge.
    """
    # Format analogues section
    analogues_text = ""
    for i, analogue in enumerate(analogues, 1):
        analogues_text += f"\n{i}. {analogue['name']} ({analogue['curie']})"
        analogues_text += f"\n   - Similarity: {analogue['similarity']:.2f}"
        analogues_text += f"\n   - Category: {analogue.get('category', 'Unknown')}"

    # Format connections section
    connections_text = ""
    for analogue_curie, conn_data in analogue_connections.items():
        analogue_name = next((a["name"] for a in analogues if a["curie"] == analogue_curie), analogue_curie)
        connections_text += f"\n\n### {analogue_name} ({analogue_curie})"
        connections_text += f"\nConnection Summary: {conn_data.get('summary', 'None')}"
        connections_text += f"\nTotal Edges: {conn_data.get('total_count', 0)}"

        # Show sample edges
        edges = conn_data.get("edges", [])[:10]  # Limit to 10 sample edges
        if edges:
            connections_text += "\nSample Connections:"
            for edge in edges:
                subject = edge.get("subject", {})
                obj = edge.get("object", {})
                pred = edge.get("predicate", "related_to")
                subj_name = subject.get("name", subject.get("id", "?"))
                obj_name = obj.get("name", obj.get("id", "?"))
                connections_text += f"\n  - {subj_name} --[{pred}]--> {obj_name}"

    return f"""## Entity to Analyze
**{raw_name}** ({curie})
- Known edges in KG: {edge_count} (sparse - qualifies for cold-start analysis)

## Similar Entities Found (via KG similar_nodes query)
{analogues_text if analogues_text else "No similar entities found."}

## Connections for Each Analogue (via KG one_hop_query)
{connections_text if connections_text else "No connection data available."}

## Your Task
Based on the REAL knowledge graph evidence above, infer what diseases, pathways, and biological roles {raw_name} might have.

Requirements:
- Use ONLY the KG data provided above for inferences
- Each inference must cite which analogue connection supports it
- All findings are Tier 3 (speculative) requiring validation
- Consider confidence based on how many analogues share similar connections
"""


def parse_inference_result(
    curie: str,
    raw_name: str,
    result_text: str
) -> tuple[list[InferredAssociation], list[Finding]]:
    """
    Parse LLM inference response into structured objects.

    Uses multi-tier JSON extraction for robustness against noisy LLM output.
    """
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
        logger.warning("Could not parse inference result for %s", curie)
        return inferences, findings

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

    return inferences, findings


async def analyze_cold_start_entity(
    curie: str,
    raw_name: str,
    edge_count: int
) -> tuple[list[AnalogueEntity], list[InferredAssociation], list[Finding], list[str]]:
    """
    Analyze a single sparse/cold-start entity using direct HTTP API + SDK inference.

    Three-step process:
    1. Get similar entities via direct Kestrel API (similar_nodes)
    2. Get connections for each analogue via direct Kestrel API (one_hop_query)
    3. Use SDK for inference reasoning over the real KG data (no MCP tools needed)

    Returns:
        tuple of (analogues, inferences, findings, errors)
    """
    errors: list[str] = []

    # Step 1: Get similar entities via direct HTTP API
    logger.info("Cold-start Step 1: Getting similar entities for '%s' (%s)...", raw_name, curie)
    similar_entities = await get_similar_entities(curie, limit=ANALOGUE_LIMIT)

    if not similar_entities:
        logger.info("Cold-start: No similar entities found for '%s' (%s)", raw_name, curie)
        return (
            [],
            [],
            [Finding(
                entity=curie,
                claim=f"No similar entities found for {raw_name} (cold-start analysis limited)",
                tier=3,
                source="cold_start",
                confidence="low",
                logic_chain="Vector similarity search returned no results",
            )],
            [],
        )

    # Convert to AnalogueEntity objects
    analogues = [
        AnalogueEntity(
            curie=s["curie"],
            name=s["name"],
            similarity=s["similarity"],
            category=s.get("category"),
        )
        for s in similar_entities
    ]

    # Step 2: Get connections for each analogue via direct HTTP API
    logger.info("Cold-start Step 2: Getting connections for %d analogues...", len(similar_entities))
    analogue_connections: dict[str, dict] = {}

    # Query connections in parallel for efficiency
    connection_tasks = [
        get_entity_connections(s["curie"])
        for s in similar_entities
    ]
    connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)

    for entity, result in zip(similar_entities, connection_results):
        if isinstance(result, Exception):
            logger.warning("Failed to get connections for %s: %s", entity["curie"], result)
            analogue_connections[entity["curie"]] = {"edges": [], "summary": f"Error: {result}"}
        else:
            analogue_connections[entity["curie"]] = result

    # Step 3: Use SDK for inference reasoning (no MCP tools needed)
    if not HAS_SDK:
        # SDK not available - return analogues only
        logger.info("Cold-start: SDK unavailable, returning %d analogues without inferences", len(analogues))
        analogue_names = [a.name for a in analogues[:3]]
        return (
            analogues,
            [],
            [Finding(
                entity=curie,
                claim=f"{raw_name} is semantically similar to: {', '.join(analogue_names)}",
                tier=3,
                source="cold_start",
                confidence="low",
                logic_chain=f"Found {len(analogues)} analogues via vector similarity search (SDK unavailable for inference)",
            )],
            [],
        )

    try:
        async with SDK_SEMAPHORE:
            # Format prompt with real KG data
            inference_prompt = format_inference_prompt(
                curie, raw_name, edge_count,
                similar_entities, analogue_connections
            )

            options = ClaudeAgentOptions(
                system_prompt=INFERENCE_PROMPT,
                allowed_tools=[],  # No tools needed - KG data provided in prompt
                max_turns=1,  # Single turn - just inference reasoning
                permission_mode="bypassPermissions",
            )

            result_text_parts: list[str] = []
            event_count = 0

            async def collect_events() -> None:
                """Collect SDK query events into result_text_parts."""
                nonlocal event_count
                async for event in query(
                    prompt=inference_prompt,
                    options=options
                ):
                    event_count += 1
                    if hasattr(event, 'content'):
                        for block in event.content:
                            if hasattr(block, 'text'):
                                result_text_parts.append(block.text)

            logger.info("Cold-start Step 3: Invoking SDK inference for '%s' (%s)...", raw_name, curie)
            try:
                await asyncio.wait_for(collect_events(), timeout=SDK_INFERENCE_TIMEOUT)
                logger.info("Cold-start SDK inference for '%s' completed with %d events", raw_name, event_count)
            except asyncio.TimeoutError:
                logger.error(
                    "Cold-start SDK inference TIMED OUT for '%s' (%s) after %ds",
                    raw_name, curie, SDK_INFERENCE_TIMEOUT
                )
                # Return analogues even if inference times out
                analogue_names = [a.name for a in analogues[:3]]
                return (
                    analogues,
                    [],
                    [Finding(
                        entity=curie,
                        claim=f"{raw_name} is semantically similar to: {', '.join(analogue_names)}",
                        tier=3,
                        source="cold_start",
                        confidence="low",
                        logic_chain=f"Found {len(analogues)} analogues (inference timed out)",
                    )],
                    [f"SDK inference timed out for {curie} after {SDK_INFERENCE_TIMEOUT}s"],
                )
            except Exception as sdk_error:
                logger.error(
                    "Cold-start SDK inference failed for '%s' (%s): %s\n%s",
                    raw_name, curie, repr(str(sdk_error)[:500]),
                    traceback.format_exc()
                )
                # Return analogues even if inference fails
                analogue_names = [a.name for a in analogues[:3]]
                return (
                    analogues,
                    [],
                    [Finding(
                        entity=curie,
                        claim=f"{raw_name} is semantically similar to: {', '.join(analogue_names)}",
                        tier=3,
                        source="cold_start",
                        confidence="low",
                        logic_chain=f"Found {len(analogues)} analogues (inference failed: {str(sdk_error)[:50]})",
                    )],
                    [f"SDK inference failed for {curie}: {str(sdk_error)}"],
                )

            result_text = "".join(result_text_parts)

            # Log successful response for diagnosis
            logger.info(
                "Cold-start inference response for '%s' (%s): length=%d, preview=%s",
                raw_name, curie, len(result_text), repr(result_text[:300])
            )

        # Parse inference results
        inferences, findings = parse_inference_result(
            curie, raw_name, result_text
        )

        # If no inferences were parsed, still report the analogues
        if not findings:
            analogue_names = [a.name for a in analogues[:3]]
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} is semantically similar to: {', '.join(analogue_names)}",
                tier=3,
                source="cold_start",
                confidence="low",
                logic_chain=f"Found {len(analogues)} analogues via vector similarity search",
            ))

        return analogues, inferences, findings, errors

    except Exception as e:
        error_msg = f"Cold-start analysis failed for {curie}: {str(e)}"
        logger.error("Cold-start analysis failed for '%s' (%s): %s", raw_name, curie, e)
        # Return analogues even on error
        analogue_names = [a.name for a in analogues[:3]] if analogues else []
        return (
            analogues,
            [],
            [Finding(
                entity=curie,
                claim=f"{raw_name} analysis partial: {', '.join(analogue_names) if analogue_names else 'no analogues'}",
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

        # Progress logging
        processed_count = min((batch_idx + 1) * BATCH_SIZE, len(entities))
        logger.info(
            "Cold-start progress: %d/%d entities processed (batch %d/%d)",
            processed_count, len(entities), batch_idx + 1, len(batches)
        )

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
