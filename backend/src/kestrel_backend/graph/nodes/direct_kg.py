"""
Direct KG Analysis Node: Analyze entities using knowledge graph traversal with ranking presets.

This node uses a two-tier architecture with Kestrel's ranking presets:
- Tier 1 (API-first): 6 parallel API calls (3 categories × 2 presets) per entity
- Tier 2 (LLM fallback): Claude SDK for failed API calls only

Ranking Presets:
- "established": Well-characterized, high-evidence connections
- "hidden_gems": Novel/less-studied connections that may reveal new biology

# Future: Add "frontier" and "speculative" presets for deeper exploration

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

# Edges per preset per category (6 calls × 25 = 150 max per entity)
PRESET_LIMIT = 25

# Ranking presets to use (Kestrel API supports these)
PRESETS = ["established", "hidden_gems"]

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
# Deduplication and Merging Helpers
# =============================================================================

def _merge_into_deduped(
    deduped: dict[str, dict],
    results: list[dict],
    preset: str,
    nodes: dict[str, dict]
) -> None:
    """
    Merge API results into deduplicated dict, tracking which presets found each.

    Args:
        deduped: Dict keyed by end_node_id, value contains edge info and presets
        results: List of result rows from API response
        preset: Which preset found these results ("established" or "hidden_gems")
        nodes: Nodes dict from API response for name lookup
    """
    for row in results:
        end_node_id = row.get("end_node_id", "")
        if not end_node_id:
            continue

        # Get node info
        node_info = nodes.get(end_node_id, {})
        node_name = node_info.get("name", end_node_id)

        # Get edge info
        edges = row.get("edges", [])
        if not edges:
            continue

        edge = edges[0]
        predicate = edge[1] if len(edge) > 1 else "biolink:related_to"
        source = edge[4] if len(edge) > 4 else "unknown"
        supporting = edge[5] if len(edge) > 5 else None

        # Extract PMIDs from supporting_sources
        pmids = []
        if supporting and isinstance(supporting, list):
            for s in supporting:
                if isinstance(s, str) and s.startswith("PMID:"):
                    pmids.append(s)

        if end_node_id in deduped:
            # Already seen - track that both presets found it
            existing = deduped[end_node_id]
            if existing["preset"] != preset:
                existing["preset"] = "both"
            # Merge PMIDs
            existing["pmids"] = list(set(existing["pmids"] + pmids))
        else:
            # New entry
            deduped[end_node_id] = {
                "end_node_id": end_node_id,
                "name": node_name,
                "predicate": predicate,
                "source": source,
                "pmids": pmids,
                "preset": preset,
            }


def _determine_evidence_type(source: str) -> str:
    """Determine evidence type from source string."""
    if not source:
        return "curated"
    source_lower = source.lower()
    if "gwas" in source_lower:
        return "gwas"
    elif "text" in source_lower or "pubmed" in source_lower:
        return "text_mined"
    elif "predict" in source_lower:
        return "predicted"
    return "curated"


# =============================================================================
# Tier 1: API-First Analysis with Ranking Presets
# =============================================================================

def parse_deduped_diseases(
    curie: str,
    raw_name: str,
    deduped: dict[str, dict]
) -> tuple[list[DiseaseAssociation], list[Finding]]:
    """Convert deduplicated disease results into structured objects."""
    diseases: list[DiseaseAssociation] = []
    findings: list[Finding] = []

    for end_node_id, data in deduped.items():
        evidence_type = _determine_evidence_type(data["source"])
        preset = data["preset"]

        diseases.append(DiseaseAssociation(
            entity_curie=curie,
            disease_curie=end_node_id,
            disease_name=data["name"],
            predicate=data["predicate"],
            source=data["source"] or "unknown",
            pmids=data["pmids"],
            evidence_type=evidence_type,
            discovery_preset=preset,
        ))

        # Highlight hidden_gems in finding claim
        novelty_note = ""
        if preset == "hidden_gems":
            novelty_note = " [novel connection]"
        elif preset == "both":
            novelty_note = " [established + novel]"

        findings.append(Finding(
            entity=curie,
            claim=f"{raw_name} is associated with {data['name']}{novelty_note} via {data['predicate']}",
            tier=1,
            predicate=data["predicate"],
            source=f"direct_kg:{preset}",
            pmids=data["pmids"],
            confidence="high" if data["pmids"] else "moderate",
        ))

    return diseases, findings


def parse_deduped_pathways(
    curie: str,
    raw_name: str,
    deduped: dict[str, dict]
) -> tuple[list[PathwayMembership], list[Finding]]:
    """Convert deduplicated pathway results into structured objects."""
    pathways: list[PathwayMembership] = []
    findings: list[Finding] = []

    for end_node_id, data in deduped.items():
        preset = data["preset"]

        pathways.append(PathwayMembership(
            entity_curie=curie,
            pathway_curie=end_node_id,
            pathway_name=data["name"],
            predicate=data["predicate"],
            source=data["source"] or "unknown",
            discovery_preset=preset,
        ))

        novelty_note = ""
        if preset == "hidden_gems":
            novelty_note = " [novel connection]"
        elif preset == "both":
            novelty_note = " [established + novel]"

        findings.append(Finding(
            entity=curie,
            claim=f"{raw_name} participates in {data['name']}{novelty_note}",
            tier=1,
            predicate=data["predicate"],
            source=f"direct_kg:{preset}",
            confidence="high",
        ))

    return pathways, findings


def parse_deduped_genes(
    curie: str,
    raw_name: str,
    deduped: dict[str, dict]
) -> list[Finding]:
    """Convert deduplicated gene interaction results into findings."""
    findings: list[Finding] = []

    for end_node_id, data in deduped.items():
        preset = data["preset"]

        novelty_note = ""
        if preset == "hidden_gems":
            novelty_note = " [novel connection]"
        elif preset == "both":
            novelty_note = " [established + novel]"

        findings.append(Finding(
            entity=curie,
            claim=f"{raw_name} interacts with {data['name']} ({end_node_id}){novelty_note}",
            tier=2,
            predicate=data["predicate"],
            source=f"direct_kg:{preset}",
            confidence="moderate",
        ))

    return findings


async def analyze_via_api(
    curie: str,
    raw_name: str
) -> tuple[list[DiseaseAssociation], list[PathwayMembership], list[Finding], list[str], list[str]] | None:
    """
    Tier 1: Analyze entity via direct Kestrel API calls with ranking presets.

    Makes 6 parallel calls: 3 categories × 2 presets (established + hidden_gems)
    Deduplicates results by end_node_id, tracking which presets found each.

    Args:
        curie: Entity CURIE to analyze
        raw_name: Human-readable name for findings

    Returns:
        Tuple of (diseases, pathways, findings, hub_flags, errors) or None if API fails.
    """
    try:
        categories = [
            ("biolink:Disease", "disease"),
            ("biolink:BiologicalProcess", "pathway"),
            ("biolink:Gene", "gene"),
        ]

        # Build 6 parallel tasks
        tasks = []
        task_labels = []

        for cat_filter, cat_key in categories:
            for preset in PRESETS:
                tasks.append(call_kestrel_tool("one_hop_query", {
                    "start_node_ids": curie,
                    "end_node_category": cat_filter,
                    "ranking": preset,
                    "mode": "slim",
                    "limit": PRESET_LIMIT,
                }))
                task_labels.append((cat_key, preset))

        # Execute all 6 calls in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by category, deduplicate within each
        disease_deduped: dict[str, dict] = {}
        pathway_deduped: dict[str, dict] = {}
        gene_deduped: dict[str, dict] = {}

        for i, result in enumerate(results):
            cat_key, preset = task_labels[i]

            if isinstance(result, Exception):
                logger.warning("API call failed for %s/%s/%s: %s",
                              curie, cat_key, preset, str(result))
                continue

            if result.get("isError"):
                continue

            content = result.get("content", [])
            if not content:
                continue

            try:
                text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                data = json.loads(text)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

            nodes = data.get("nodes", {})
            api_results = data.get("results", [])

            if cat_key == "disease":
                _merge_into_deduped(disease_deduped, api_results, preset, nodes)
            elif cat_key == "pathway":
                _merge_into_deduped(pathway_deduped, api_results, preset, nodes)
            elif cat_key == "gene":
                _merge_into_deduped(gene_deduped, api_results, preset, nodes)

        # Convert deduplicated results to structured objects
        diseases, disease_findings = parse_deduped_diseases(curie, raw_name, disease_deduped)
        pathways, pathway_findings = parse_deduped_pathways(curie, raw_name, pathway_deduped)
        gene_findings = parse_deduped_genes(curie, raw_name, gene_deduped)

        all_findings = disease_findings + pathway_findings + gene_findings

        # Count by preset for logging
        established_count = sum(1 for d in disease_deduped.values() if d["preset"] == "established")
        hidden_gems_count = sum(1 for d in disease_deduped.values() if d["preset"] == "hidden_gems")
        both_count = sum(1 for d in disease_deduped.values() if d["preset"] == "both")

        logger.info(
            "Tier 1 direct_kg '%s': diseases=%d (est=%d, hg=%d, both=%d), pathways=%d, genes=%d",
            curie, len(diseases), established_count, hidden_gems_count, both_count,
            len(pathways), len(gene_findings)
        )

        return diseases, pathways, all_findings, [], []  # Hub flags detected later

    except Exception as e:
        logger.warning("Tier 1 direct_kg '%s': Exception - %s", curie, str(e))
        return None


# =============================================================================
# Tier 2: LLM Fallback (for API failures)
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
        # Match outermost braces with nested content - simplified pattern
        start_idx = result_text.find('{')
        if start_idx != -1:
            # Find matching closing brace by counting
            depth = 0
            end_idx = start_idx
            for i, c in enumerate(result_text[start_idx:], start_idx):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > start_idx:
                try:
                    data = json.loads(result_text[start_idx:end_idx])
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
            source = d.get("source", "unknown")
            evidence_type = _determine_evidence_type(source)

            diseases.append(DiseaseAssociation(
                entity_curie=curie,
                disease_curie=d.get("curie", ""),
                disease_name=d.get("name", "Unknown"),
                predicate=d.get("predicate", "biolink:related_to"),
                source=source,
                pmids=d.get("pmids", []),
                evidence_type=evidence_type,
                discovery_preset="llm_fallback",
            ))

            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} is associated with {d.get('name', 'unknown disease')} via {d.get('predicate', 'unknown')}",
                tier=1,
                predicate=d.get("predicate"),
                source="direct_kg:llm_fallback",
                pmids=d.get("pmids", []),
                confidence="high" if d.get("pmids") else "moderate",
            ))

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
                discovery_preset="llm_fallback",
            ))

            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} participates in {p.get('name', 'unknown pathway')}",
                tier=1,
                predicate=p.get("predicate"),
                source="direct_kg:llm_fallback",
                confidence="high",
            ))
        except Exception:
            continue

    # Parse protein interactions
    for i in data.get("interactions", []):
        try:
            findings.append(Finding(
                entity=curie,
                claim=f"{raw_name} interacts with {i.get('name', 'unknown')} ({i.get('curie', '')})",
                tier=2,
                predicate=i.get("predicate"),
                source="direct_kg:llm_fallback",
                confidence="moderate",
            ))
        except Exception:
            continue

    hub_flags.extend(data.get("hub_flags", []))

    return diseases, pathways, findings, hub_flags


async def analyze_single_entity(
    curie: str,
    raw_name: str
) -> tuple[list[DiseaseAssociation], list[PathwayMembership], list[Finding], list[str], list[str]]:
    """
    Tier 2: Analyze a single entity using Claude Agent SDK with LLM reasoning.

    This is the fallback path when Tier 1 API calls fail.

    Returns:
        tuple of (diseases, pathways, findings, hub_flags, errors)
    """
    if not HAS_SDK:
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
                max_buffer_size=10 * 1024 * 1024,
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
    for score in novelty_scores:
        if score.curie == curie:
            return score.raw_name

    for entity in resolved_entities:
        if hasattr(entity, 'curie') and entity.curie == curie:
            return entity.raw_name

    return curie


# =============================================================================
# Main Run Function
# =============================================================================

async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Analyze entities using Kestrel ranking presets for established vs novel connections.

    Tier 1 (API-first): 6 parallel API calls per entity (3 categories × 2 presets)
    Tier 2 (LLM fallback): Claude SDK for failed API calls only

    Receives: well_characterized_curies + moderate_curies from triage
    Returns: direct_findings, disease_associations, pathway_memberships, hub_flags, errors
    """
    well_char = state.get("well_characterized_curies", [])
    moderate = state.get("moderate_curies", [])
    curies = well_char + moderate

    logger.info("Starting direct_kg with %d entities (presets=%s)",
                len(curies), PRESETS)
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

    # Deduplicate curies BEFORE creating tasks
    unique_curies: list[str] = []
    curie_to_indices: dict[str, list[int]] = {}

    for i, curie in enumerate(curies):
        if curie not in curie_to_indices:
            curie_to_indices[curie] = []
            unique_curies.append(curie)
        curie_to_indices[curie].append(i)

    # Prepare result containers
    all_results: list[tuple | None] = [None] * len(curies)
    errors: list[str] = []

    # ========== TIER 1: API Analysis with Ranking Presets ==========
    tier1_start = time.time()
    logger.info("Tier 1 (API): Analyzing %d unique entities (%d total with duplicates)",
                len(unique_curies), len(curies))

    tier1_tasks = []
    for curie in unique_curies:
        raw_name = get_raw_name_for_curie(curie, novelty_scores, resolved_entities)
        tier1_tasks.append(analyze_via_api(curie, raw_name))

    tier1_results = await asyncio.gather(*tier1_tasks, return_exceptions=True)

    tier1_success = 0
    tier2_needed_curies: list[str] = []

    for curie, result in zip(unique_curies, tier1_results):
        indices = curie_to_indices[curie]

        if isinstance(result, Exception):
            logger.warning("Tier 1 exception for %s: %s", curie, str(result))
            tier2_needed_curies.append(curie)
        elif result is None:
            tier2_needed_curies.append(curie)
        else:
            # Copy result to all indices where this curie appears
            for idx in indices:
                all_results[idx] = result
            tier1_success += 1

    tier1_duration = time.time() - tier1_start
    logger.info("Tier 1 (API) analyzed %d/%d unique entities in %.1fs",
                tier1_success, len(unique_curies), tier1_duration)

    # ========== TIER 2: LLM Fallback ==========
    if tier2_needed_curies and HAS_SDK:
        tier2_start = time.time()
        logger.info("Tier 2 (LLM): Processing %d unique entities that failed Tier 1",
                    len(tier2_needed_curies))

        for batch in chunk(tier2_needed_curies, BATCH_SIZE):
            batch_tasks = []
            for curie in batch:
                raw_name = get_raw_name_for_curie(curie, novelty_scores, resolved_entities)
                batch_tasks.append(analyze_single_entity(curie, raw_name))

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for curie, result in zip(batch, batch_results):
                indices = curie_to_indices[curie]
                if isinstance(result, Exception):
                    errors.append(f"Tier 2 failed for {curie}: {str(result)}")
                elif result is not None:
                    for idx in indices:
                        all_results[idx] = result

        tier2_duration = time.time() - tier2_start
        logger.info("Tier 2 (LLM) completed in %.1fs", tier2_duration)

    # Aggregate results
    all_diseases: list[DiseaseAssociation] = []
    all_pathways: list[PathwayMembership] = []
    all_findings: list[Finding] = []
    all_hub_flags: list[str] = []

    for result in all_results:
        if result and isinstance(result, tuple) and len(result) >= 5:
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

    # Count findings by preset for logging
    established_findings = sum(1 for f in all_findings if "established" in (f.source or ""))
    hidden_gems_findings = sum(1 for f in all_findings if "hidden_gems" in (f.source or ""))

    duration = time.time() - start
    logger.info(
        "Completed direct_kg in %.1fs — findings=%d (established=%d, hidden_gems=%d), diseases=%d, pathways=%d",
        duration, len(all_findings), established_findings, hidden_gems_findings,
        len(all_diseases), len(all_pathways)
    )

    return {
        "direct_findings": all_findings,
        "disease_associations": all_diseases,
        "pathway_memberships": all_pathways,
        "hub_flags": all_hub_flags,
        "errors": errors,
    }
