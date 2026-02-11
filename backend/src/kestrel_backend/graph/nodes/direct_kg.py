"""
Direct KG Analysis Node: Analyze well-characterized entities using knowledge graph traversal.

This node uses a two-tier architecture for optimal performance:
- Tier 1 (API-first): Direct Kestrel API calls with structured JSON parsing (~1-2s per entity)
- Tier 2 (LLM fallback): Claude SDK for failed API calls or complex reasoning

Extracts:
- Disease associations via one_hop_query filtered by biolink:Disease
- Pathway memberships via one_hop_query filtered by biolink:BiologicalProcess
- Gene interactions for genes/proteins

Hub bias detection flags high-degree neighbors (>5000 edges) that could create
spurious associations.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ..state import (
    DiscoveryState, Finding, DiseaseAssociation, PathwayMembership, NoveltyScore
)

logger = logging.getLogger(__name__)

# Try to import Claude Agent SDK - graceful fallback if not available
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.types import McpStdioServerConfig
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


# =============================================================================
# Configuration Constants
# =============================================================================

# Discovery depth configuration
# "standard" = Tier 1 API only (fast, ~3s for 8 entities)
# "deep" = Tier 2 LLM for well_characterized entities (slow, ~20min, but finds novel associations)
DISCOVERY_DEPTH = os.getenv("DIRECT_KG_DEPTH", "standard")

# Tier 1 API settings
SLIM_LIMIT = 50   # edges per category filter in standard mode
DEEP_LIMIT = 200  # edges per category filter in deep mode (still API, no LLM)

# Hub detection threshold (uses edge_count from triage's novelty_scores)
HUB_THRESHOLD = 5000  # Entities with >5000 edges are flagged as hubs

# Kestrel MCP command for stdio-based server (used by Tier 2)
KESTREL_COMMAND = "uvx"
KESTREL_ARGS = ["mcp-client-kestrel"]

# Semaphore to limit concurrent SDK calls (Tier 2 only)
SDK_SEMAPHORE = asyncio.Semaphore(6)

# Batch size for parallel analysis
BATCH_SIZE = 6

# System prompt for direct KG analysis (Tier 2 LLM fallback)
DIRECT_KG_PROMPT = """You are a biomedical knowledge graph analyst. For the given entity (CURIE),
use one_hop_query to retrieve and analyze its relationships.

Query for THREE types of associations:
1. Disease associations - use one_hop_query with end_category filter for biolink:Disease
2. Pathway/biological process - filter for biolink:Pathway or biolink:BiologicalProcess
3. Protein interactions (for genes) - filter for biolink:Protein or biolink:Gene

For each association found, extract:
- The predicate type (e.g., biolink:treats, biolink:gene_associated_with_condition)
- The source database from edge provenance if available
- Any PMIDs in the edge data

CRITICAL: If a neighbor has very high connectivity (mentioned as having many edges or
appearing frequently), flag it as a potential hub. Hub nodes create spurious associations.

Return ONLY a valid JSON object (no other text):
{
  "diseases": [
    {"curie": "MONDO:...", "name": "...", "predicate": "...", "source": "...", "pmids": [], "is_hub": false}
  ],
  "pathways": [
    {"curie": "GO:...", "name": "...", "predicate": "...", "source": "..."}
  ],
  "interactions": [
    {"curie": "...", "name": "...", "predicate": "..."}
  ],
  "hub_flags": ["CURIE1", "CURIE2"]
}

If the query returns no results or fails, return:
{"diseases": [], "pathways": [], "interactions": [], "hub_flags": []}
"""


# =============================================================================
# Tier 1: API-First Analysis (Fast Path)
# =============================================================================

def parse_disease_edges(
    curie: str,
    raw_name: str,
    result: dict[str, Any]
) -> tuple[list[DiseaseAssociation], list[Finding], list[str]]:
    """
    Parse one_hop_query response into DiseaseAssociation objects and findings.

    Response structure (from actual API test):
    - content[0]["text"] contains JSON string with:
      - results[]: list of {end_node_id, edges: [[subj, pred, obj, qual, source, supporting, ...]], score}
      - nodes: {curie: {name, categories, ...}}
    - Edge indices per edge_schema: [0]=subject, [1]=predicate, [4]=primary_knowledge_source, [5]=supporting_sources

    Returns: (disease_associations, findings, hub_flags)
    """
    diseases: list[DiseaseAssociation] = []
    findings: list[Finding] = []
    hub_flags: list[str] = []

    if result.get("isError"):
        return diseases, findings, hub_flags

    content = result.get("content", [])
    if not content:
        return diseases, findings, hub_flags

    try:
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError, KeyError):
        return diseases, findings, hub_flags

    # Get nodes dict for name lookup
    nodes = data.get("nodes", {})

    # Process each result row
    for row in data.get("results", []):
        try:
            end_node_id = row.get("end_node_id", "")
            # Get name from nodes dict, fallback to CURIE
            end_node_info = nodes.get(end_node_id, {})
            end_node_name = end_node_info.get("name", end_node_id)

            # Extract edge info - take first edge for this end_node
            edges = row.get("edges", [])
            if not edges:
                continue

            edge = edges[0]  # Use first edge
            predicate = edge[1] if len(edge) > 1 else "biolink:related_to"
            source = edge[4] if len(edge) > 4 else "unknown"
            supporting = edge[5] if len(edge) > 5 else None

            # Determine evidence type from source
            evidence_type = "curated"
            if source and "gwas" in source.lower():
                evidence_type = "gwas"
            elif source and ("text" in source.lower() or "pubmed" in source.lower()):
                evidence_type = "text_mined"
            elif source and "predict" in source.lower():
                evidence_type = "predicted"

            # Extract PMIDs from supporting_sources if available
            pmids = []
            if supporting and isinstance(supporting, list):
                for s in supporting:
                    if isinstance(s, str) and s.startswith("PMID:"):
                        pmids.append(s)

            diseases.append(DiseaseAssociation(
                entity_curie=curie,
                disease_curie=end_node_id,
                disease_name=end_node_name,
                predicate=predicate,
                source=source or "unknown",
                pmids=pmids,
                evidence_type=evidence_type,
            ))

            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} is associated with {end_node_name} via {predicate}",
                tier=1,
                predicate=predicate,
                source="direct_kg",
                pmids=pmids,
                confidence="high" if pmids else "moderate",
            ))
        except Exception:
            continue

    # Hub detection: use edge_count from novelty_scores (passed separately), not from this response
    return diseases, findings, hub_flags


def parse_pathway_edges(
    curie: str,
    raw_name: str,
    result: dict[str, Any]
) -> tuple[list[PathwayMembership], list[Finding]]:
    """Parse one_hop_query response into PathwayMembership objects."""
    pathways: list[PathwayMembership] = []
    findings: list[Finding] = []

    if result.get("isError"):
        return pathways, findings

    content = result.get("content", [])
    if not content:
        return pathways, findings

    try:
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError, KeyError):
        return pathways, findings

    # Get nodes dict for name lookup
    nodes = data.get("nodes", {})

    for row in data.get("results", []):
        try:
            end_node_id = row.get("end_node_id", "")
            end_node_info = nodes.get(end_node_id, {})
            end_node_name = end_node_info.get("name", end_node_id)
            edges = row.get("edges", [])

            if not edges:
                continue

            edge = edges[0]
            predicate = edge[1] if len(edge) > 1 else "biolink:participates_in"
            source = edge[4] if len(edge) > 4 else "unknown"

            pathways.append(PathwayMembership(
                entity_curie=curie,
                pathway_curie=end_node_id,
                pathway_name=end_node_name,
                predicate=predicate,
                source=source or "unknown",
            ))

            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} participates in {end_node_name}",
                tier=1,
                predicate=predicate,
                source="direct_kg",
                confidence="high",
            ))
        except Exception:
            continue

    return pathways, findings


def parse_gene_edges(
    curie: str,
    raw_name: str,
    result: dict[str, Any]
) -> list[Finding]:
    """Parse one_hop_query response into Finding objects for gene interactions."""
    findings: list[Finding] = []

    if result.get("isError"):
        return findings

    content = result.get("content", [])
    if not content:
        return findings

    try:
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError, KeyError):
        return findings

    # Get nodes dict for name lookup
    nodes = data.get("nodes", {})

    for row in data.get("results", []):
        try:
            end_node_id = row.get("end_node_id", "")
            end_node_info = nodes.get(end_node_id, {})
            end_node_name = end_node_info.get("name", end_node_id)
            edges = row.get("edges", [])

            if not edges:
                continue

            edge = edges[0]
            predicate = edge[1] if len(edge) > 1 else "biolink:interacts_with"

            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} interacts with {end_node_name} ({end_node_id})",
                tier=2,
                predicate=predicate,
                source="direct_kg",
                confidence="moderate",
            ))
        except Exception:
            continue

    return findings


async def analyze_via_api(
    curie: str,
    raw_name: str,
    limit: int = SLIM_LIMIT
) -> tuple[list[DiseaseAssociation], list[PathwayMembership], list[Finding], list[str], list[str]] | None:
    """
    Tier 1: Analyze entity via direct Kestrel API calls.

    Makes 3 parallel filtered one_hop_query calls (Disease, BiologicalProcess, Gene)
    and parses the structured JSON responses directly.

    Args:
        curie: Entity CURIE to analyze
        raw_name: Human-readable name for findings
        limit: Max edges per category (SLIM_LIMIT for standard, DEEP_LIMIT for deep mode)

    Returns:
        Tuple of (diseases, pathways, findings, hub_flags, errors) or None if API fails.
    """
    try:
        # Make 3 parallel filtered queries
        disease_task = call_kestrel_tool("one_hop_query", {
            "start_node_ids": curie,
            "end_category_filter": "biolink:Disease",
            "mode": "slim",
            "limit": limit,
        })
        pathway_task = call_kestrel_tool("one_hop_query", {
            "start_node_ids": curie,
            "end_category_filter": "biolink:BiologicalProcess",
            "mode": "slim",
            "limit": limit,
        })
        gene_task = call_kestrel_tool("one_hop_query", {
            "start_node_ids": curie,
            "end_category_filter": "biolink:Gene",
            "mode": "slim",
            "limit": limit,
        })

        disease_result, pathway_result, gene_result = await asyncio.gather(
            disease_task, pathway_task, gene_task
        )

        # Parse results and build structured objects
        diseases, disease_findings, hub_flags = parse_disease_edges(curie, raw_name, disease_result)
        pathways, pathway_findings = parse_pathway_edges(curie, raw_name, pathway_result)
        gene_findings = parse_gene_edges(curie, raw_name, gene_result)

        all_findings = disease_findings + pathway_findings + gene_findings

        logger.info(
            "Tier 1 direct_kg '%s': diseases=%d, pathways=%d, genes=%d",
            curie, len(diseases), len(pathways), len(gene_findings)
        )

        return diseases, pathways, all_findings, hub_flags, []

    except Exception as e:
        logger.warning("Tier 1 direct_kg '%s': Exception - %s", curie, str(e))
        return None


# =============================================================================
# Tier 2: LLM Fallback (for API failures or complex reasoning)
# =============================================================================

def parse_direct_kg_result(
    curie: str,
    raw_name: str,
    result_text: str
) -> tuple[list[DiseaseAssociation], list[PathwayMembership], list[Finding], list[str]]:
    """
    Parse LLM response into structured analysis objects.

    Uses multi-tier JSON extraction for robustness against noisy LLM output.

    Returns:
        tuple of (disease_associations, pathway_memberships, findings, hub_flags)
    """
    diseases: list[DiseaseAssociation] = []
    pathways: list[PathwayMembership] = []
    findings: list[Finding] = []
    hub_flags: list[str] = []

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
        # Match outermost braces with nested content
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
        return diseases, pathways, findings, hub_flags

    # Parse disease associations
    for d in data.get("diseases", []):
        try:
            # Determine evidence type from source
            source = d.get("source", "unknown")
            evidence_type = "curated"
            if "gwas" in source.lower():
                evidence_type = "gwas"
            elif "text" in source.lower() or "pubmed" in source.lower():
                evidence_type = "text_mined"
            elif "predict" in source.lower():
                evidence_type = "predicted"

            diseases.append(DiseaseAssociation(
                entity_curie=curie,
                disease_curie=d.get("curie", ""),
                disease_name=d.get("name", "Unknown"),
                predicate=d.get("predicate", "biolink:related_to"),
                source=source,
                pmids=d.get("pmids", []),
                evidence_type=evidence_type,
            ))

            # Create Tier 1 finding for each disease association
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} is associated with {d.get('name', 'unknown disease')} via {d.get('predicate', 'unknown')}",
                tier=1,
                predicate=d.get("predicate"),
                source="direct_kg",
                pmids=d.get("pmids", []),
                confidence="high" if d.get("pmids") else "moderate",
            ))

            # Check for hub flag
            if d.get("is_hub", False):
                hub_flags.append(d.get("curie", ""))

        except Exception:
            continue

    # Parse pathway memberships
    for p in data.get("pathways", []):
        try:
            pathways.append(PathwayMembership(
                entity_curie=curie,
                pathway_curie=p.get("curie", ""),
                pathway_name=p.get("name", "Unknown"),
                predicate=p.get("predicate", "biolink:participates_in"),
                source=p.get("source", "unknown"),
            ))

            # Create Tier 1 finding for pathway
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} participates in {p.get('name', 'unknown pathway')}",
                tier=1,
                predicate=p.get("predicate"),
                source="direct_kg",
                confidence="high",
            ))
        except Exception:
            continue

    # Parse protein interactions (findings only, no separate model)
    for i in data.get("interactions", []):
        try:
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} interacts with {i.get('name', 'unknown')} ({i.get('curie', '')})",
                tier=2,
                predicate=i.get("predicate"),
                source="direct_kg",
                confidence="moderate",
            ))
        except Exception:
            continue

    # Collect hub flags
    hub_flags.extend(data.get("hub_flags", []))

    return diseases, pathways, findings, hub_flags


async def analyze_single_entity(
    curie: str,
    raw_name: str
) -> tuple[list[DiseaseAssociation], list[PathwayMembership], list[Finding], list[str], list[str]]:
    """
    Tier 2: Analyze a single entity using Claude Agent SDK with LLM reasoning.

    This is the fallback path when Tier 1 API calls fail or for deep analysis.

    Returns:
        tuple of (diseases, pathways, findings, hub_flags, errors)
    """
    if not HAS_SDK:
        # SDK not available - return placeholder for testing
        return (
            [],
            [],
            [Finding(
                entity=curie,
                claim=f"Direct KG analysis pending for {raw_name} (SDK unavailable)",
                tier=1,
                source="direct_kg",
                confidence="low",
            )],
            [],
            [],
        )

    try:
        async with SDK_SEMAPHORE:
            kestrel_config = McpStdioServerConfig(
                type="stdio",
                command=KESTREL_COMMAND,
                args=KESTREL_ARGS,
            )

            options = ClaudeAgentOptions(
                system_prompt=DIRECT_KG_PROMPT,
                allowed_tools=[
                    "mcp__kestrel__one_hop_query",
                    "mcp__kestrel__get_nodes",
                    "mcp__kestrel__get_edges",
                ],
                mcp_servers={"kestrel": kestrel_config},
                max_turns=4,
                permission_mode="bypassPermissions",
                max_buffer_size=10 * 1024 * 1024,  # 10MB buffer for large KG responses
            )

            result_text_parts = []
            async for event in query(prompt=f"Analyze entity: {raw_name} ({curie})", options=options):
                if hasattr(event, 'content'):
                    for block in event.content:
                        if hasattr(block, 'text'):
                            result_text_parts.append(block.text)

            result_text = "".join(result_text_parts)
        diseases, pathways, findings, hub_flags = parse_direct_kg_result(curie, raw_name, result_text)

        logger.info(
            "Tier 2 direct_kg '%s': diseases=%d, pathways=%d, findings=%d",
            curie, len(diseases), len(pathways), len(findings)
        )

        return diseases, pathways, findings, hub_flags, []

    except Exception as e:
        error_msg = f"Tier 2 direct_kg failed for {curie}: {str(e)}"
        logger.error(error_msg)
        return (
            [],
            [],
            [Finding(
                entity=curie,
                claim=f"Analysis failed for {raw_name}: {str(e)[:100]}",
                tier=1,
                source="direct_kg",
                confidence="low",
            )],
            [],
            [error_msg],
        )


# =============================================================================
# Helper Functions
# =============================================================================

def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of specified size."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def get_raw_name_for_curie(curie: str, novelty_scores: list[NoveltyScore], resolved_entities: list) -> str:
    """Look up the raw name for a CURIE from novelty scores or resolved entities."""
    # Check novelty scores first
    for score in novelty_scores:
        if score.curie == curie:
            return score.raw_name

    # Fall back to resolved entities
    for entity in resolved_entities:
        if hasattr(entity, 'curie') and entity.curie == curie:
            return entity.raw_name

    # Default to CURIE if no name found
    return curie


# =============================================================================
# Main Run Function
# =============================================================================

async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Two-tier analysis of well-characterized entities with configurable depth.

    Tier 1 (API-first): Direct Kestrel API calls with structured JSON parsing
    Tier 2 (LLM fallback): Claude SDK for failed API calls or deep analysis mode

    Receives: well_characterized_curies + moderate_curies from triage
    Returns: direct_findings, disease_associations, pathway_memberships, hub_flags, errors
    """
    well_char = state.get("well_characterized_curies", [])
    moderate = state.get("moderate_curies", [])
    curies = well_char + moderate

    # Get discovery depth from state or use default from environment
    depth = state.get("discovery_depth", DISCOVERY_DEPTH)

    # Configure based on depth mode
    if depth == "deep":
        limit = DEEP_LIMIT
        use_llm_for_well_char = True
    else:
        limit = SLIM_LIMIT
        use_llm_for_well_char = False

    logger.info("Starting direct_kg with %d entities (depth=%s, limit=%d)",
                len(curies), depth, limit)
    start = time.time()

    if not curies:
        logger.info("No entities to analyze, skipping direct_kg")
        return {
            "direct_findings": [],
            "disease_associations": [],
            "pathway_memberships": [],
            "hub_flags": [],
            "errors": [],
        }

    # Get novelty scores and resolved entities for name lookup
    novelty_scores = state.get("novelty_scores", [])
    resolved_entities = state.get("resolved_entities", [])

    # Prepare result containers with indices
    all_results: list[tuple | None] = [None] * len(curies)
    errors: list[str] = []

    # ========== TIER 1: API Analysis ==========
    tier1_start = time.time()
    logger.info("Tier 1 (API): Analyzing %d entities", len(curies))

    tier1_tasks = []
    for curie in curies:
        raw_name = get_raw_name_for_curie(curie, novelty_scores, resolved_entities)
        tier1_tasks.append(analyze_via_api(curie, raw_name, limit=limit))

    tier1_results = await asyncio.gather(*tier1_tasks, return_exceptions=True)

    tier1_success = 0
    tier2_needed = []

    for i, result in enumerate(tier1_results):
        if isinstance(result, Exception):
            logger.warning("Tier 1 exception for %s: %s", curies[i], str(result))
            tier2_needed.append(i)  # API exception
        elif result is None:
            tier2_needed.append(i)  # API failed
        else:
            all_results[i] = result
            tier1_success += 1
            # In deep mode: also use LLM for well_characterized entities
            if use_llm_for_well_char and curies[i] in well_char:
                tier2_needed.append(i)  # LLM enrichment for novel associations

    tier1_duration = time.time() - tier1_start
    logger.info("Tier 1 (API) analyzed %d/%d entities in %.1fs",
                tier1_success, len(curies), tier1_duration)

    # ========== TIER 2: LLM Fallback / Enrichment ==========
    if tier2_needed and HAS_SDK:
        tier2_start = time.time()
        tier2_reason = "failed Tier 1" if not use_llm_for_well_char else "failed Tier 1 + well_characterized enrichment"
        logger.info("Tier 2 (LLM): Processing %d entities (%s)",
                    len(tier2_needed), tier2_reason)

        tier2_curies = [curies[i] for i in tier2_needed]
        tier2_indices = tier2_needed.copy()

        for batch in chunk(tier2_curies, BATCH_SIZE):
            batch_tasks = []
            for curie in batch:
                raw_name = get_raw_name_for_curie(curie, novelty_scores, resolved_entities)
                batch_tasks.append(analyze_single_entity(curie, raw_name))

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            batch_indices = tier2_indices[:len(batch)]
            tier2_indices = tier2_indices[len(batch):]

            for idx, result in zip(batch_indices, batch_results):
                if isinstance(result, Exception):
                    errors.append(f"Tier 2 failed for {curies[idx]}: {str(result)}")
                elif result is not None:
                    # In deep mode, merge with existing API results
                    if all_results[idx] is not None:
                        # Merge: combine diseases, pathways, findings from both tiers
                        existing = all_results[idx]
                        diseases, pathways, findings, hub_flags, errs = result
                        all_results[idx] = (
                            existing[0] + diseases,
                            existing[1] + pathways,
                            existing[2] + findings,
                            existing[3] + hub_flags,
                            existing[4] + errs,
                        )
                    else:
                        all_results[idx] = result

        tier2_duration = time.time() - tier2_start
        logger.info("Tier 2 (LLM) completed in %.1fs", tier2_duration)

    # Aggregate results
    all_diseases: list[DiseaseAssociation] = []
    all_pathways: list[PathwayMembership] = []
    all_findings: list[Finding] = []
    all_hub_flags: list[str] = []

    for result in all_results:
        if result:
            diseases, pathways, findings, hub_flags, errs = result
            all_diseases.extend(diseases)
            all_pathways.extend(pathways)
            all_findings.extend(findings)
            all_hub_flags.extend(hub_flags)
            errors.extend(errs)

    # Flag hubs based on novelty_scores edge_count
    for score in novelty_scores:
        if score.edge_count > HUB_THRESHOLD:
            if score.curie not in all_hub_flags:
                all_hub_flags.append(score.curie)
                logger.info("Flagged hub entity: %s (edges=%d)", score.curie, score.edge_count)

    duration = time.time() - start
    logger.info("Completed direct_kg in %.1fs — findings=%d, diseases=%d, pathways=%d (depth=%s)",
                duration, len(all_findings), len(all_diseases), len(all_pathways), depth)

    return {
        "direct_findings": all_findings,
        "disease_associations": all_diseases,
        "pathway_memberships": all_pathways,
        "hub_flags": all_hub_flags,
        "errors": errors,
    }
