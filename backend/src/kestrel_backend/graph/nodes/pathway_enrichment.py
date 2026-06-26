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
import os
import re
import time
from typing import Any

from ..state import (
    DiscoveryState, SharedNeighbor, BiologicalTheme, Finding
)
from ...kestrel_client import multi_hop_query, parse_kestrel_response
from .cold_start import get_entity_connections
from ..sdk_utils import HAS_SDK, ClaudeAgentOptions, query_with_usage
from ..pipeline_config import get_pipeline_config
from ..state_contracts import validate_state, PathwayEnrichmentInput, PathwayEnrichmentOutput

logger = logging.getLogger(__name__)

_config = get_pipeline_config().pathway_enrichment

# Minimum edge count to include entity in pathway analysis
# Only moderate (20-199) and well-characterized (200+) entities have enough
# KG connections to produce meaningful shared neighbor analysis
MIN_EDGE_COUNT = 20

# Timeout for the Phase B SDK query. Configurable (PathwayEnrichmentConfig.sdk_query_timeout,
# default 480s tuned for Sonnet) with a runtime env override (KRAKEN_PATHWAY_SDK_TIMEOUT) so
# slower models like Opus 4.8 can be given more headroom on this multi-turn node.
SDK_QUERY_TIMEOUT = int(os.getenv("KRAKEN_PATHWAY_SDK_TIMEOUT") or _config.sdk_query_timeout)

# Limit concurrent SDK inference calls (issue #44 Stage 2; also closes the
# previously-missing semaphore gap for this node).
SDK_SEMAPHORE = asyncio.Semaphore(_config.sdk_semaphore)

# Inference system prompt (issue #44): the one-hop neighbors are pre-fetched via HTTP and
# embedded in the user prompt, so the model has NO tools — it reasons over the provided
# data and emits a strict shared_neighbors/themes JSON schema.
#
# NOTE: this node was migrated off the stdio Kestrel MCP server (`uvx mcp-client-kestrel`,
# which does not exist on PyPI — see cold_start.py) to direct HTTP. The same broken stdio
# path is still used by entity_resolution, direct_kg, integration, temporal, and triage;
# their SDK tiers are silently degraded by the same root cause and should be audited /
# migrated under a separate issue. The Stage 1 instrumentation (sdk_utils diagnostics)
# quantifies their exposure in production logs.
PATHWAY_INFERENCE_PROMPT = """You are a biomedical knowledge graph analyst finding shared biological context.

The one-hop neighbors of each input entity have already been retrieved from the Kestrel
knowledge graph and are provided to you below. You do NOT have tools — reason over the
provided data only.

TASK: Find neighbors that appear in 2+ input entities' neighbor lists (shared neighbors),
track which input entities connect to each, and group them by Biolink category into
biological themes.

Hub nodes (very high degree — e.g. GO:0005515 protein binding, GO:0005737 cytoplasm) connect
to everything and provide less insight. Mark them with "is_hub": true when you can tell.

Return ONLY a valid JSON object (no other text):
{
  "shared_neighbors": [
    {
      "curie": "GO:0006915",
      "name": "apoptotic process",
      "category": "biolink:BiologicalProcess",
      "degree": 245,
      "is_hub": false,
      "connected_inputs": ["CHEBI:28757", "CHEBI:4208"],
      "predicates": ["biolink:participates_in"]
    }
  ],
  "themes": [
    {
      "category": "biolink:BiologicalProcess",
      "members": ["GO:0006915"],
      "member_names": ["apoptotic process"],
      "input_coverage": 2,
      "top_non_hub": "GO:0006915"
    }
  ]
}

If no shared neighbors are found, return: {"shared_neighbors": [], "themes": []}
"""


def _build_inference_user_prompt(selected_entities: list, per_entity: dict[str, dict]) -> str:
    """Embed the prefetched one-hop neighbors into the inference prompt (issue #44, Stage 2)."""
    blocks = []
    for e in selected_entities:
        data = per_entity.get(e.curie, {})
        if data.get("errored"):
            continue  # skip entities whose HTTP prefetch failed — no useful data to embed
        neighbors = []
        for edge in data.get("edges", []):
            if isinstance(edge, dict):
                obj = edge.get("object", {})
                oid = obj.get("id") if isinstance(obj, dict) else None
                pred = edge.get("predicate", "")
                if oid:
                    neighbors.append(f"{oid} [{pred}]")
        name = e.resolved_name or e.raw_name
        neighbor_str = "; ".join(neighbors[:50]) if neighbors else "none"
        blocks.append(
            f"### {e.curie} ({name})\n"
            f"Predicate summary: {data.get('summary', '')}\n"
            f"Neighbors ({len(neighbors)}): {neighbor_str}"
        )
    body = "\n\n".join(blocks)
    return (
        f"Analyze these {len(selected_entities)} entities' one-hop neighbors and find "
        f"neighbors shared by 2+ entities:\n\n{body}\n\nReturn JSON only."
    )


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
                is_hub=sn.get("is_hub", False) or sn.get("degree", 0) > _config.hub_threshold,
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


async def find_two_hop_shared_neighbors(
    entity_curies: list[str],
    max_results_per_entity: int = 50,
) -> tuple[dict[str, int], list[str]]:
    """
    Find nodes that connect to 2+ input entities within 2 hops using multi_hop_query.

    This complements the LLM-based one-hop analysis by finding indirect connections.

    Args:
        entity_curies: List of input entity CURIEs
        max_results_per_entity: Limit per entity for result size control

    Returns:
        tuple of (neighbor_counts, errors)
        - neighbor_counts: dict mapping neighbor CURIE -> count of inputs connected
        - errors: list of error messages
    """
    neighbor_counts: dict[str, int] = {}
    errors: list[str] = []

    if len(entity_curies) < 2:
        return neighbor_counts, errors

    logger.info(
        "find_two_hop_shared_neighbors: analyzing %d entities with max_hops=2",
        len(entity_curies)
    )

    for curie in entity_curies:
        try:
            # Use singly-pinned multi_hop_query to explore 2 hops from each entity
            result = await multi_hop_query(
                start_node_ids=[curie],
                max_hops=2,
                limit=max_results_per_entity,
            )

            if result.get("isError"):
                error_text = result.get("content", [{}])[0].get("text", "Unknown error")
                errors.append(f"multi_hop_query error for {curie}: {error_text}")
                continue

            # Parse via the shared helper. Kestrel's multi-hop response is
            # {"results": [...], "nodes": {...}, "edges": {...}} — there is NO top-level
            # "paths" key; the helper reads result["end_node_id"] + result["paths"] (CURIE
            # lists) and fails loudly to empty on a bad shape. Collect every node this entity
            # can reach (end-nodes + intermediates, deduped per input), drop the start node,
            # then count it as ONE input connection so the >=2 filter below means "reachable
            # from 2+ distinct input entities" (not "appeared on 2+ paths").
            parsed = parse_kestrel_response(result)
            reached: set[str] = set(parsed["end_node_ids"])
            for path in parsed["paths"]:
                reached.update(path["curies"])

            reached.discard(curie)  # exclude the start node itself
            for node in reached:
                neighbor_counts[node] = neighbor_counts.get(node, 0) + 1

        except Exception as e:
            logger.error("Error in two-hop search for %s: %s", curie, str(e))
            errors.append(f"Exception for {curie}: {str(e)}")

    # Filter to only neighbors connected to 2+ inputs
    shared_neighbors = {
        curie: count
        for curie, count in neighbor_counts.items()
        if count >= 2
    }

    logger.info(
        "find_two_hop_shared_neighbors: found %d shared neighbors (from %d total)",
        len(shared_neighbors), len(neighbor_counts)
    )

    return shared_neighbors, errors


def _build_two_hop_findings(two_hop_neighbors: dict[str, int]) -> list[Finding]:
    """Build Phase A two-hop Finding objects.

    Computed independently of Phase B so they survive a Phase B degradation or
    exception (issue #44, R5).
    """
    if not two_hop_neighbors:
        return []
    top_two_hop = sorted(two_hop_neighbors.items(), key=lambda x: x[1], reverse=True)[:3]
    two_hop_summary = ", ".join([f"{curie} ({count} inputs)" for curie, count in top_two_hop])
    return [Finding(
        entity=", ".join([c for c, _ in top_two_hop]),
        claim=f"Two-hop connectivity: {two_hop_summary}",
        tier=2,
        source="pathway_enrichment_two_hop",
        confidence="moderate",
    )]


def _degraded_phase_b_result(
    two_hop_findings: list[Finding],
    base_errors: list[str],
    usage_record: Any,
    reason: str,
) -> dict[str, Any]:
    """Shared degraded-result builder (issue #44).

    Drops the unreliable SDK Phase B findings, keeps the real Phase A two-hop
    findings, and flags the run as degraded. Used for both the MCP-unavailable
    path (Unit 3) and the HTTP-prefetch-no-data path (Unit 5).
    """
    logger.warning(
        "pathway_enrichment degraded (reason=%s) — dropping SDK shared neighbors, keeping two-hop",
        reason,
    )
    result: dict[str, Any] = {
        "shared_neighbors": [],
        "biological_themes": [],
        "direct_findings": two_hop_findings,
        "pathway_enrichment_degraded": True,
        "errors": base_errors + [f"pathway_enrichment Phase B degraded: {reason}"],
    }
    if usage_record is not None:
        result["model_usages"] = [usage_record]
    return result


async def prefetch_one_hop_neighbors(entities: list) -> tuple[dict[str, dict], bool]:
    """Fetch one-hop neighbor data per entity via the HTTP Kestrel client (issue #44, Stage 2).

    Mirrors cold_start's ``get_entity_connections`` (which collapses every failure mode into
    an empty ``{"edges": []}`` with no ``total_count`` key — so absence of ``total_count``
    is our per-entity ``errored`` discriminator).

    Returns ``(per_entity, no_data)`` where ``per_entity`` maps curie -> ``{summary, edges,
    errored}`` and ``no_data`` is True when fewer than 2 entities returned real (non-errored,
    non-empty) neighbor data. ``no_data`` is the post-migration emptiness signal that keeps
    the in-prompt inference from running on a sparse, hallucination-prone prompt.
    """
    curies = [e.curie for e in entities if e.curie]
    results = await asyncio.gather(
        *[get_entity_connections(c) for c in curies], return_exceptions=True
    )
    per_entity: dict[str, dict] = {}
    populated = 0
    for curie, result in zip(curies, results):
        if isinstance(result, Exception):
            logger.warning("prefetch one_hop_query raised for %s: %s", curie, result)
            per_entity[curie] = {"summary": f"Error: {result}", "edges": [], "errored": True}
            continue
        errored = "total_count" not in result  # get_entity_connections drops it on every failure
        edges = result.get("edges", [])
        per_entity[curie] = {
            "summary": result.get("summary", ""),
            "edges": edges,
            "errored": errored,
        }
        if not errored and edges:
            populated += 1

    no_data = populated < 2  # shared-neighbor analysis is meaningless below 2 populated entities
    if no_data:
        logger.warning(
            "pathway_enrichment prefetch: only %d/%d entities returned real neighbor data (no-data signal)",
            populated, len(curies),
        )
    return per_entity, no_data


@validate_state(PathwayEnrichmentInput, PathwayEnrichmentOutput)
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

    # Get edge counts from novelty scores to filter sparse entities
    novelty_scores = state.get("novelty_scores", [])
    edge_count_map = {ns.curie: ns.edge_count for ns in novelty_scores}

    # Filter to entities with enough KG connections (moderate + well-characterized)
    # Sparse entities (<20 edges) rarely share neighbors and waste SDK turns
    selected_entities = [
        e for e in valid_entities
        if edge_count_map.get(e.curie, 0) >= MIN_EDGE_COUNT
    ]
    filtered_count = len(valid_entities) - len(selected_entities)

    # Log selection
    logger.info(
        "pathway_enrichment: selected %d/%d entities (edge_count >= %d): %s",
        len(selected_entities), len(valid_entities), MIN_EDGE_COUNT,
        [(e.curie, edge_count_map.get(e.curie, 0)) for e in selected_entities]
    )
    if filtered_count > 0:
        logger.info(
            "pathway_enrichment: filtered out %d sparse entities (edge_count < %d)",
            filtered_count, MIN_EDGE_COUNT
        )

    # Skip if fewer than 2 entities after filtering
    if len(selected_entities) < 2:
        logger.warning(
            "Skipped pathway_enrichment: need at least 2 entities with edge_count >= %d (have %d)",
            MIN_EDGE_COUNT, len(selected_entities)
        )
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": [f"Need at least 2 entities with edge_count >= {MIN_EDGE_COUNT} for pathway enrichment"],
        }

    # Check SDK availability
    if not HAS_SDK:
        # Return placeholder for testing without SDK
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "errors": ["Claude Agent SDK not available for pathway enrichment"],
        }

    # Phase A: Two-hop shared neighbor analysis via API (fast, no LLM)
    logger.info("Starting two-hop shared neighbor analysis via API...")
    entity_curies = [e.curie for e in selected_entities if e.curie]
    two_hop_neighbors, two_hop_errors = await find_two_hop_shared_neighbors(
        entity_curies=entity_curies,
        max_results_per_entity=50,
    )

    # Log two-hop results
    if two_hop_neighbors:
        logger.info(
            "Two-hop analysis found %d shared neighbors: %s",
            len(two_hop_neighbors),
            list(two_hop_neighbors.items())[:5]
        )

    # Stage Phase A two-hop findings OUTSIDE the Phase B try so they survive a
    # Phase B degradation or exception (issue #44, R5).
    two_hop_findings = _build_two_hop_findings(two_hop_neighbors)

    try:
        # Stage 2 (issue #44): fetch one-hop neighbors via HTTP, then reason over them
        # in-prompt with NO stdio MCP. Mirrors cold_start's data-in-prompt inference.
        per_entity, no_data = await prefetch_one_hop_neighbors(selected_entities)
        if no_data:
            # Fewer than 2 entities returned real neighbor data — running inference on a
            # sparse prompt would re-hallucinate, so degrade instead (issue #44).
            return _degraded_phase_b_result(
                two_hop_findings, two_hop_errors, None, "prefetch_no_data"
            )

        user_prompt = _build_inference_user_prompt(selected_entities, per_entity)

        options = ClaudeAgentOptions(
            system_prompt=PATHWAY_INFERENCE_PROMPT,
            allowed_tools=[],  # data-in-prompt inference — no MCP tools (issue #44)
            max_turns=2,
            permission_mode="bypassPermissions",
            max_buffer_size=10 * 1024 * 1024,  # 10MB buffer for embedded KG data
        )

        # Execute the inference under the shared SDK semaphore (closes the
        # previously-missing semaphore gap for this node).
        try:
            async with SDK_SEMAPHORE:
                result_text, usage_record = await asyncio.wait_for(
                    query_with_usage(
                        prompt=user_prompt,
                        options=options,
                        node_name="pathway_enrichment",
                    ),
                    timeout=SDK_QUERY_TIMEOUT,
                )
        except asyncio.TimeoutError:
            duration = time.time() - start
            logger.error(
                "pathway_enrichment SDK query TIMED OUT after %ds (%.1fs elapsed)",
                SDK_QUERY_TIMEOUT, duration
            )
            return {
                "shared_neighbors": [],
                "biological_themes": [],
                "direct_findings": two_hop_findings,  # preserve Phase A on timeout (issue #44, R5)
                "pathway_enrichment_degraded": True,  # disclose: Phase B produced nothing
                "errors": [f"SDK query timed out after {SDK_QUERY_TIMEOUT}s"],
            }

        # Log raw response for diagnosis
        logger.info(
            "pathway_enrichment raw response: length=%d, preview=%s",
            len(result_text), repr(result_text[:500]) if result_text else "empty"
        )

        # Parse the result
        shared_neighbors, themes, parse_errors = parse_enrichment_result(result_text)

        # Log parse errors if any
        if parse_errors:
            logger.warning("pathway_enrichment parse errors: %s", parse_errors)

        # Issue #44 (Stage 2): Phase B uses data-in-prompt inference (allowed_tools=[]),
        # so there is no MCP tier left to degrade — the MCP-degradation classifier was
        # removed here rather than left as inert/unreachable code. The active protection
        # against sparse-prompt re-hallucination is the prefetch no_data guard above
        # (see the _degraded_phase_b_result call at the prefetch stage). If MCP tools are
        # ever reintroduced into Phase B, re-add classify_mcp_degradation(expected_tools=[...])
        # plus the drop_findings_on_degraded branch here.
        phase_b_degraded = False

        # Create findings from top themes (SDK-derived — dropped on degradation)
        sdk_findings: list[Finding] = []
        for theme in themes[:5]:  # Top 5 themes
            if theme.input_coverage >= 2:
                non_hub_count = sum(
                    1 for sn in shared_neighbors
                    if sn.curie in theme.members and not sn.is_hub
                )
                category_name = theme.category.replace("biolink:", "")
                sdk_findings.append(Finding(
                    entity=", ".join(theme.members[:3]),
                    claim=f"Shared {category_name} context: {', '.join(theme.member_names[:3])} connects {theme.input_coverage} input entities",
                    tier=2,
                    source="pathway_enrichment",
                    confidence="high" if non_hub_count > 0 else "moderate",
                ))

        # Combine SDK theme findings with the staged Phase A two-hop findings.
        findings = sdk_findings + two_hop_findings

        duration = time.time() - start
        logger.info(
            "Completed pathway_enrichment in %.1fs — themes=%d, shared_neighbors=%d, two_hop=%d",
            duration, len(themes), len(shared_neighbors), len(two_hop_neighbors)
        )

        # Combine errors
        all_errors = parse_errors + two_hop_errors

        result_dict: dict[str, Any] = {
            "shared_neighbors": shared_neighbors,
            "biological_themes": themes,
            "direct_findings": findings,  # Add to direct_findings via reducer
            "pathway_enrichment_degraded": phase_b_degraded,
            "errors": all_errors,
        }
        if usage_record is not None:
            result_dict["model_usages"] = [usage_record]
        return result_dict

    except Exception as e:
        duration = time.time() - start
        logger.error("Pathway enrichment failed after %.1fs: %s", duration, str(e))
        # Preserve the staged Phase A two-hop findings even on a Phase B exception
        # (issue #44, R5): emit direct_findings so the operator.add reducer receives them.
        return {
            "shared_neighbors": [],
            "biological_themes": [],
            "direct_findings": two_hop_findings,
            "pathway_enrichment_degraded": True,  # disclose: Phase B produced nothing
            "errors": [f"Pathway enrichment failed: {str(e)}"],
        }
