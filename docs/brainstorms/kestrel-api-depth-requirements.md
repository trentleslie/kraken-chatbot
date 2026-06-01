---
date: 2026-05-18
topic: kestrel-api-depth
---

# Kestrel API Depth: Method-Aware Discovery Pipeline

## Problem Frame

The KRAKEN discovery pipeline uses ~8 of the 22 Kestrel REST API endpoints. After spiking the actual API inventory (OpenAPI spec at `/api/docs`), we confirmed 5 real endpoints go unused (`subgraph`, `vector-search`, `canonicalize`, `traversal-options`, `metagraph`) and 4 of 6 ranking presets are untouched. The pipeline also underuses `multi-hop` (only in Integration, not Direct KG).

Separately, the 5 "advanced" MCP tools (`guilt_by_association`, `missing_edge_prediction`, etc.) declared in Classic mode's `ALLOWED_TOOLS` are agent-side Claude SDK prompts — not Kestrel server endpoints. They are out of scope for this work.

A systematic review of 135+ biomedical link prediction studies ([PDF resource](../references/biomedical-link-prediction-methods-and-evidence.pdf)) provides directional guidance: different tools work better depending on entity sparsity. The Kestrel API's ranking presets (`established` through `long_shot`) and search modes (`one-hop` vs `vector-search` vs `hybrid-search`) naturally map to this spectrum — dense entities benefit from topological queries, sparse entities from embedding-based search.

**Who is affected:** Researchers using KRAKEN for discovery queries get shallower analysis than the KG supports.

**What changes:** Triage outputs a per-entity tool strategy that downstream nodes use to select API endpoints and ranking presets, producing deeper analysis tailored to each entity's position on the sparsity spectrum.

## Requirements

**Per-Entity Tool Strategy**

- R1. Triage must output a `tool_strategies: dict[str, ToolStrategy]` (keyed by CURIE) alongside the existing sparsity classification. Each strategy specifies: recommended ranking presets, whether to use multi-hop, and preferred search mode (topological vs hybrid vs embedding-based).
- R2. Downstream nodes (Direct KG, Cold-Start, Pathway Enrichment, Integration) must read the tool strategy for each entity and adapt their Kestrel API calls accordingly, rather than using a fixed call pattern for all entities.

**Ranking Preset Expansion**

- R3. Direct KG must expand from 2 presets (`established`, `hidden_gems`) to use presets informed by R1's tool strategy. Dense entities get `established`/`hidden_gems`; moderate entities add `frontier`; sparse entities get `speculative`/`long_shot`. The `deep_dive` preset applies to all entity types for thorough neighborhood exploration.
- R4. All 6 presets are confirmed to exist on the server and apply to `/one-hop`, `/multi-hop`, and `/subgraph`. Validation during planning: run the same well-characterized entity through all 6 presets and compare result differentiation before wiring all of them in.

**Multi-Hop in Direct KG**

- R5. Direct KG must use `/multi-hop` for well-characterized entities to detect mechanistic chains (Drug -> Gene -> Pathway -> Disease). The endpoint and a `multi_hop_query()` wrapper already exist in `kestrel_client.py` — this is wiring it into a new node, not building new infrastructure.

**Subgraph Extraction**

- R6. Integration must use the `/subgraph` endpoint to find connecting subgraphs between entities, augmenting the existing multi-hop bridge detection. Path structure from subgraph results feeds into Synthesis as evidence for hypothesis generation.

**Activate Unused Endpoints**

- R7. Entity Resolution must call `/canonicalize` after initial CURIE resolution to normalize identifiers before they propagate through the pipeline.
- R8. Pipeline initialization must query `/traversal-options` and `/metagraph` once at startup and cache the results. Triage and downstream nodes use this metadata to construct valid queries (available constraint fields, operators, entity categories, predicates) instead of hardcoding them.

**Validation Tier Metadata**

- R9. Hypotheses must carry a new `validation_tier` field (separate from the existing `tier: Literal[1, 2, 3]`, which is unchanged) reflecting where the hypothesis sits on the review's validation hierarchy:
  - Validation Tier 1: Computational prediction only
  - Validation Tier 2: Literature/database corroboration
  - Validation Tier 3: Wet-lab testable (validation steps are concrete and feasible)
  - Validation Tier 4: Clinical investigation relevance
- R10. Synthesis must present tier-specific attrition context from the review (100% computational -> ~60% literature -> ~30% wet-lab -> ~18% clinical) as interpretive framing, not precision scores. These are literature-derived baselines across diverse domains, not calibrated for Kestrel specifically.

## Success Criteria

- A cold-start entity and a well-characterized entity in the same query produce visibly different pipeline logs: different API endpoints called, different ranking presets used, different search strategies
- Multi-hop mechanistic chains appear in synthesis reports for well-characterized entities (currently only one-hop associations)
- Subgraph connections appear in Integration bridge analysis
- CURIEs are consistent across pipeline nodes (no identifier drift after Entity Resolution)
- Pipeline execution time is profiled before and after; regressions are documented against analysis quality gains

## Scope Boundaries

- **In scope:** Wiring verified REST API endpoints into the pipeline via `kestrel_client.py`; per-entity triage routing; validation tier metadata on hypotheses
- **Out of scope:** Classic mode changes; the 5 agent-side "discovery" tools (`guilt_by_association`, `missing_edge_prediction`, `gap_analysis`, `novelty_score`, `pathway_enrichment`) — these are Claude SDK prompts, not API endpoints; training ML models; new Kestrel API endpoints; frontend UI changes; literature grounding changes
- **Not changing:** Pipeline topology (10 nodes), WebSocket protocol, existing `tier: Literal[1, 2, 3]` semantics

## Key Decisions

- **Direct HTTP via `kestrel_client.py`, not MCP:** The pipeline uses direct HTTP calls. New integrations follow this pattern. Classic mode's MCP tool declarations are a separate concern.
- **Per-entity strategies, not per-query:** Different entities in the same query get different API calls based on sparsity.
- **Additive throughout:** All existing analysis stays. New endpoints augment, never replace.
- **Review as directional guidance, not engineering spec:** The systematic review's method selection matrix informs which tools to try for which entity types. The mapping is approximate — validate empirically during implementation.

## Dependencies / Assumptions

- Kestrel REST API at `https://kestrel.nathanpricelab.com/api` remains stable. OpenAPI spec is authoritative.
- The existing `multi_hop_query()` wrapper in `kestrel_client.py` works for Direct KG use cases (currently only used in Integration).
- `/subgraph`, `/canonicalize`, `/traversal-options`, `/metagraph` are functional (confirmed present in OpenAPI spec but never called — may have edge cases).

## Outstanding Questions

### Resolve Before Planning

(none — V1 spike completed, tool inventory confirmed)

### Deferred to Planning

- [Affects R3-R4][Needs research] Run all 6 ranking presets on a representative entity and compare results. Drop any preset that doesn't differentiate.
- [Affects R5][Technical] Profile multi-hop query latency per entity to determine how many calls are feasible.
- [Affects R6][Technical] Test `/subgraph` with representative entity pairs — confirm it returns connecting paths, not just shared neighbors.
- [Affects R8][Technical] Determine cache invalidation strategy for `/traversal-options` and `/metagraph` — per-session cache is likely sufficient.

## Next Steps

-> `/ce:plan` for structured implementation planning
