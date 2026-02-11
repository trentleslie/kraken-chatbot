"""
Direct KG Analysis Node: Analyze well-characterized entities using knowledge graph traversal.

This node queries the Kestrel KG for entities with good coverage (well_characterized or moderate)
to extract:
- Disease associations via one_hop_query filtered by biolink:Disease
- Pathway memberships via one_hop_query filtered by biolink:Pathway / biolink:BiologicalProcess
- Protein interactions for genes/proteins

Hub bias detection flags high-degree neighbors (>1000 edges) that could create
spurious associations.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

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


# Kestrel MCP command for stdio-based server (same as entity_resolution)
KESTREL_COMMAND = "uvx"
KESTREL_ARGS = ["mcp-client-kestrel"]

# Semaphore to limit concurrent SDK calls (increased from 1 to 6 for parallelism)
SDK_SEMAPHORE = asyncio.Semaphore(6)

# Batch size for parallel analysis
BATCH_SIZE = 6

# System prompt for direct KG analysis
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
    Analyze a single well-characterized entity using Claude Agent SDK.

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

        return diseases, pathways, findings, hub_flags, []

    except Exception as e:
        error_msg = f"Direct KG analysis failed for {curie}: {str(e)}"
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


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Analyze entities with strong KG representation.

    Queries the Kestrel KG for disease associations, pathway memberships,
    and protein interactions. Flags high-degree hub nodes.

    Receives: well_characterized_curies + moderate_curies from triage
    Returns: direct_findings, disease_associations, pathway_memberships, hub_flags, errors
    """
    well_char = state.get("well_characterized_curies", [])
    moderate = state.get("moderate_curies", [])
    curies = well_char + moderate

    logger.info("Starting direct_kg with %d entities", len(curies))
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

    all_diseases: list[DiseaseAssociation] = []
    all_pathways: list[PathwayMembership] = []
    all_findings: list[Finding] = []
    all_hub_flags: list[str] = []
    errors: list[str] = []

    # Process in batches for controlled parallelism
    for batch in chunk(curies, BATCH_SIZE):
        batch_tasks = []
        for curie in batch:
            raw_name = get_raw_name_for_curie(curie, novelty_scores, resolved_entities)
            batch_tasks.append(analyze_single_entity(curie, raw_name))

        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for curie, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                errors.append(f"Exception analyzing {curie}: {str(result)}")
                all_findings.append(Finding(
                    entity=curie,
                    claim=f"Analysis exception for {curie}",
                    tier=1,
                    source="direct_kg",
                    confidence="low",
                ))
            else:
                diseases, pathways, findings, hub_flags, errs = result
                all_diseases.extend(diseases)
                all_pathways.extend(pathways)
                all_findings.extend(findings)
                all_hub_flags.extend(hub_flags)
                errors.extend(errs)

    duration = time.time() - start
    logger.info(
        "Completed direct_kg in %.1fs — findings=%d, diseases=%d, pathways=%d",
        duration, len(all_findings), len(all_diseases), len(all_pathways)
    )

    return {
        "direct_findings": all_findings,
        "disease_associations": all_diseases,
        "pathway_memberships": all_pathways,
        "hub_flags": all_hub_flags,
        "errors": errors,
    }
