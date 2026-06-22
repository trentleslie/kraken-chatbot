"""Pipeline configuration for the discovery workflow.

Centralizes all per-node thresholds, semaphore values, entity limits, and batch
sizes into a single Pydantic model with documented rationale for each default.

Values are testable/overridable per-run via get_pipeline_config.cache_clear().
Not sourced from environment variables — these are pipeline tuning parameters,
not deployment configuration.
"""

import asyncio
from functools import lru_cache

from pydantic import BaseModel, Field


class DirectKGConfig(BaseModel):
    """Configuration for the direct_kg node."""

    hub_threshold: int = Field(
        default=5000,
        description="Entities with more edges than this are flagged as hubs. "
        "Set higher than pathway_enrichment (1000) because direct_kg detects "
        "entity-level hub bias, not shared-neighbor filtering.",
    )
    sdk_semaphore: int = Field(
        default=6,
        description="Concurrent SDK calls for tier-2 analysis. Higher than "
        "entity_resolution/triage (1) because direct_kg runs batch analysis.",
    )
    batch_size: int = Field(
        default=6,
        description="Number of entities to analyze in parallel per batch.",
    )
    preset_limit: int = Field(
        default=25,
        description="Maximum edges per preset per category in KG queries.",
    )
    multi_hop_enabled: bool = Field(
        default=False,
        description="Demo-slice flag: when True, well-characterized entities also get a "
        "multi_hop_query for mechanistic chains. Default False keeps the pipeline inert "
        "until the demo run flips it (docs/plans/2026-05-30-001-feat-discovery-depth-demo-slice-plan.md).",
    )
    multi_hop_max_hops: int = Field(
        default=2,
        description="Max path length for direct_kg multi-hop queries.",
    )
    multi_hop_limit: int = Field(
        default=10,
        description="Max paths returned per multi-hop query.",
    )
    multi_hop_semaphore: int = Field(
        default=6,
        description="Concurrent multi-hop API calls against the shared Kestrel server. "
        "Kept independent of batch_size (Tier-2 SDK batching) so tuning one does not "
        "silently change the other.",
    )


class PathwayEnrichmentConfig(BaseModel):
    """Configuration for the pathway_enrichment node."""

    hub_threshold: int = Field(
        default=1000,
        description="Nodes with more edges than this are flagged as hubs in "
        "shared-neighbor filtering. Lower than direct_kg (5000) because this "
        "filters neighbors, not entities — a different purpose.",
    )
    drop_findings_on_degraded: bool = Field(
        default=True,
        description="When Phase B is detected as degraded (MCP tools unavailable or "
        "HTTP prefetch returned no data), drop the unreliable SDK shared-neighbor "
        "findings instead of letting hallucinated output reach synthesis (issue #44).",
    )
    sdk_semaphore: int = Field(
        default=4,
        description="Max concurrent SDK calls for the Phase B data-in-prompt inference "
        "(issue #44 Stage 2). Mirrors the other SDK nodes' per-node semaphore.",
    )


class BiomapperConfig(BaseModel):
    """Configuration for the Biomapper pre-resolver in entity_resolution.

    Default-off feature flag plus the namespace/species policy knobs the pre-resolver
    needs. Secrets (BIOMAPPER_API_KEY / BIOMAPPER_BASE_URL) live in config.Settings, not
    here — this model is never sourced from environment variables. See
    docs/plans/2026-06-11-001-feat-biomapper-entity-resolution-plan.md (Unit 1).
    """

    enabled: bool = Field(
        default=False,
        description="Default-off flag: when True, entity_resolution resolves each hinted "
        "entity via Biomapper first (namespace/species-correct CURIE), confirms it in Kestrel, "
        "and falls back to the Kestrel Tier 1/1.5/2 path on any miss. Flag-off behavior is "
        "byte-identical to today. Mirrors the multi_hop_enabled / subgraph_enabled demo-slice "
        "flags; a follow-up PR flips it after the gold-set eval confirms improvement.",
    )
    namespace_preference: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "gene": ["HGNC", "NCBIGene", "UniProtKB"],
            "protein": ["HGNC", "NCBIGene", "UniProtKB"],
            "metabolite": ["CHEBI", "HMDB", "RM", "LM"],
        },
        description="Per-class ordered namespace preference for reconciling a Biomapper CURIE "
        "against the Kestrel KG (R5: explicit config, not implicit ranking). Genes/proteins "
        "anchor on HGNC (human-only by construction; Kestrel normalizes HGNC/UniProt/NCBIGene to "
        "the same human node and keys edges on NCBIGene — verified 2026-06-15). Metabolite order "
        "(CHEBI-first) is the starting hypothesis; confirm which namespace Kestrel keys metabolite "
        "edges on at impl.",
    )
    species_default: str = Field(
        default="human",
        description="Default species for resolution (R5: explicit). For genes/proteins, "
        "humanness is enforced at confirmation time by the HGNC-marker gate in Unit 3 (an HGNC "
        "equivalent id is required), not by an implicit Biomapper default — so this field is "
        "documentation/metabolite-relevant.",
    )
    http_concurrency: int = Field(
        default=8,
        description="Max concurrent Biomapper HTTP calls (its own semaphore, independent of the "
        "entity_resolution SDK semaphore=1) — bounds fan-out without serializing, avoiding the "
        "throttling-as-no-match gotcha.",
    )
    node_timeout_seconds: float = Field(
        default=30.0,
        description="Per-entity asyncio.wait_for timeout around each Biomapper map_entity call. "
        "A stalled call falls back to Kestrel for that entity only; already-resolved entities are "
        "retained (caps added latency at ~http_concurrency × this).",
    )


class EntityResolutionConfig(BaseModel):
    """Configuration for the entity_resolution node."""

    sdk_semaphore: int = Field(
        default=1,
        description="Serialized SDK calls (semaphore=1) to prevent concurrent "
        "CLI spawn conflicts during entity resolution.",
    )
    batch_size: int = Field(
        default=6,
        description="Number of entities to resolve in parallel per batch.",
    )
    tier1_min_score: float = Field(
        default=0.6,
        description="Minimum API resolution score to accept a tier-1 match.",
    )
    tier1_fallback_confidence: float = Field(
        default=0.5,
        description="Confidence assigned when a category-constrained hybrid_search errors and "
        "resolution falls back to the unconstrained Kestrel result. Honesty signal only — the value "
        "is not compared against any threshold. (Returning a non-None result here does suppress "
        "Tier 2 for that entity, but that is driven by the fallback succeeding, not by this number.)",
    )
    biomapper: BiomapperConfig = Field(
        default_factory=BiomapperConfig,
        description="Biomapper pre-resolver config (default-off flag + namespace/species policy).",
    )


class TriageConfig(BaseModel):
    """Configuration for the triage node."""

    sdk_semaphore: int = Field(
        default=1,
        description="Serialized SDK calls (semaphore=1) to prevent concurrent "
        "CLI spawn conflicts during triage edge counting.",
    )
    batch_size: int = Field(
        default=6,
        description="Number of entities to process in parallel per batch.",
    )


class ColdStartConfig(BaseModel):
    """Configuration for the cold_start node."""

    sdk_semaphore: int = Field(
        default=8,
        description="Concurrent SDK calls for batch parallelism. Higher than "
        "other nodes because cold-start analysis benefits from parallel "
        "inference across multiple sparse entities.",
    )
    batch_size: int = Field(
        default=5,
        description="Number of entities to analyze in parallel per batch.",
    )
    inference_timeout: int = Field(
        default=60,
        description="Per-entity timeout in seconds for SDK inference.",
    )
    analogue_limit: int = Field(
        default=3,
        description="Maximum number of similar entities to retrieve per "
        "cold-start entity for analogue-based inference.",
    )


class LiteratureGroundingConfig(BaseModel):
    """Configuration for the literature_grounding node."""

    papers_per_hypothesis: int = Field(
        default=3,
        description="Maximum papers to attach per hypothesis.",
    )
    max_hypotheses: int = Field(
        default=15,
        description="Maximum hypotheses to ground (prioritize by tier).",
    )
    parallel_fetch_limit: int = Field(
        default=6,
        description="Fetch multiplier — retrieve 2x papers per source for "
        "deduplication, then trim to papers_per_hypothesis.",
    )
    use_llm_classifier: bool = Field(
        default=False,
        description="Enable LLM-based relationship classification (R14). "
        "Defaults to False (current 'supporting' behavior). Flag flip is a "
        "separate follow-up PR after quality measurement confirms improvement.",
    )
    overall_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="R13 latency ceiling: wall-clock cap on the whole grounding body "
        "(4-provider fan-out + EFetch backfill). On expiry, asyncio.timeout raises "
        "TimeoutError, caught by the node's degrade boundary, which returns the upstream "
        "hypotheses ungrounded so synthesis still runs. Value tuned vs a representative "
        "high-bridge run; existence + mechanism are decided, not deferred.",
    )


class HypothesisExtractionConfig(BaseModel):
    """Configuration for the hypothesis_extraction node."""

    validate_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description="R13 latency ceiling: wall-clock cap on the serial validate_bridge_hypotheses "
        "loop (each doubly-pinned multi_hop_query can run up to the ~60s Kestrel client timeout). "
        "On expiry, asyncio.timeout raises TimeoutError, caught by the node's degrade boundary, "
        "which returns {bridges: upstream, hypotheses: []} so synthesis still runs. A grounding-only "
        "timeout cannot bound this loop — it lives in a different node. Value tuned vs a "
        "representative high-bridge run; existence + mechanism are decided, not deferred.",
    )


class IntegrationConfig(BaseModel):
    """Configuration for the integration node."""

    hub_threshold: int = Field(
        default=5000,
        description="Subgraph traversal is constrained to nodes below this degree "
        "(in-query hub guard), matching direct_kg's threshold.",
    )
    subgraph_enabled: bool = Field(
        default=False,
        description="Demo-slice flag: when True, integration also runs a subgraph_query "
        "to surface connecting structure between resolved entities. Default False keeps "
        "the pipeline inert until the demo run flips it "
        "(docs/plans/2026-05-30-001-feat-discovery-depth-demo-slice-plan.md).",
    )
    max_subgraph_nodes: int = Field(
        default=5,
        description="Max resolved-entity CURIEs passed as node_ids to a subgraph_query.",
    )


class BridgeGroundingConfig(BaseModel):
    """Configuration for the bridge_grounding node (deterministic evidence-provenance labeler)."""

    enabled: bool = Field(
        default=True,
        description="Enabled (L4 flip, 2026-06-18). The L4 validation eval over real bridges passed: "
        "no all-`none` collapse and a 33% curated-causal leg fraction (>= the ~23% baseline) under "
        "correct Tier 1 + Tier 2 resolution — see assessment_data/bridge_grounding_eval.py and "
        "docs/plans/2026-06-17-002-feat-bridge-evidence-provenance-labeler-plan.md. When False the "
        "node is a no-op (no Kestrel calls).",
    )
    max_scored_bridges: int = Field(
        default=20,
        description="Cap on ordered 3-node bridges labeled per run. Each costs up to 2 one_hop_query "
        "full calls, so this bounds the added Kestrel load. Per-CURIE fetches are cached within a "
        "run, so the actual call count is the number of DISTINCT leg endpoints, often well below "
        "2 * max_scored_bridges. (The per-leg edge limit is fixed at L1's leg_tier constant.)",
    )
    concurrency: int = Field(
        default=8,
        ge=1,
        description="Max concurrent leg-edge one_hop_query calls. Bridges are labeled in parallel "
        "and per-CURIE fetches are deduplicated within a run (cached_leg_fetcher), so a module whose "
        "bridges share hub endpoints fetches each hub once instead of once per bridge. Bounds Kestrel "
        "load while cutting the node's wall-clock from sequential ~O(2*bridges) fetches.",
    )


class SynthesisConfig(BaseModel):
    """Configuration for the synthesis node's context-assembly caps.

    At module scale the assembled LLM context overflowed the model's ~200K-token input
    window (48-analyte Brown run, 2026-06-22: ~882K chars ~= 230K tokens; findings 58%,
    disease 21%, pathway 17% of the dump). These caps bound the context against a
    token-derived budget and switch multi-entity ("module") queries to cross-entity
    aggregation + a per-member table instead of per-entity dumps. See
    docs/plans/2026-06-22-001-feat-module-aware-synthesis-context-plan.md.
    """

    max_findings_per_tier: int = Field(
        default=50,
        ge=1,
        description="Max findings rendered per tier (1/2/3), ranked by confidence "
        "high->moderate->low; the rest are elided as '... and N more (tier T)'. Findings are "
        "the dominant context section (58% of the 882K dump, 4,099 entries), so this is the "
        "load-bearing cut. 50/tier (<=150 total) is ample for a narrative and ~27x below 4,099; "
        "tune against the R7 run.",
    )
    max_aggregated_diseases: int = Field(
        default=30,
        ge=1,
        description="Max diseases in the Module-Level Disease Recurrence section (ranked by "
        "distinct-member count, then evidence strength). Replaces the per-entity disease dump "
        "(21% of the overflow) for module queries.",
    )
    max_aggregated_pathways: int = Field(
        default=30,
        ge=1,
        description="Max pathways in the Module-Level Pathway Recurrence section. Replaces the "
        "per-entity pathway dump (17% of the overflow) for module queries.",
    )
    module_mode_min_entities: int = Field(
        default=5,
        ge=2,
        description="Resolved-entity count at/above which assembly switches to module-aware mode "
        "(aggregation + member table) instead of per-entity sections. Defaults to 5 (>2) so genuine "
        "single/pair/triple queries keep the per-entity report shape (R5); operators may lower it to 2 "
        "to treat pairs as modules. Distinct from min_members_for_recurrence: this gates *whether* "
        "module mode engages, not *which* diseases/pathways qualify for the recurrence lists.",
    )
    min_members_for_recurrence: int = Field(
        default=2,
        ge=2,
        description="Minimum distinct member entities sharing a disease/pathway for it to appear in "
        "the Module-Level Recurrence sections. Inclusive (2 = shared by any pair). Distinct from "
        "module_mode_min_entities: a low value keeps recurrence inclusive without forcing small "
        "queries into module mode.",
    )
    max_member_table_rows: int = Field(
        default=50,
        ge=1,
        description="Max rows in the per-member prioritization table (top-N by edge_count, rest "
        "elided). A no-op at 48 analytes; bounds the table at the 217-analyte target where a full "
        "table (~217 rows) would itself become a dump.",
    )
    max_context_chars: int = Field(
        default=350_000,
        ge=1,
        description="Char budget for the assembled synthesis context — a PROXY for the model's "
        "~200K-token input window (the real ceiling). ~350K chars ~= 100K tokens at the measured "
        "3.5-3.8 chars/token for this CURIE-dense content, leaving ~100K-token headroom for the "
        "system prompt + output. ~2.5x below the 882K/230K that crashed. A backstop logs a WARNING "
        "(with an estimated token count) if assembly exceeds this; the per-section caps should "
        "prevent reaching it. Tune downward if R7 shows headroom is tight.",
    )


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration with per-node sub-models.

    Each sub-model documents the rationale for its defaults via Field(description=...).
    Use get_pipeline_config() to access the singleton instance.
    Override for testing with get_pipeline_config.cache_clear() + direct instantiation.
    """

    direct_kg: DirectKGConfig = Field(default_factory=DirectKGConfig)
    pathway_enrichment: PathwayEnrichmentConfig = Field(default_factory=PathwayEnrichmentConfig)
    entity_resolution: EntityResolutionConfig = Field(default_factory=EntityResolutionConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)
    literature_grounding: LiteratureGroundingConfig = Field(default_factory=LiteratureGroundingConfig)
    hypothesis_extraction: HypothesisExtractionConfig = Field(default_factory=HypothesisExtractionConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    bridge_grounding: BridgeGroundingConfig = Field(default_factory=BridgeGroundingConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)


@lru_cache(maxsize=1)
def get_pipeline_config() -> PipelineConfig:
    """Get the pipeline configuration singleton.

    Returns the default configuration. To override for testing:
        get_pipeline_config.cache_clear()
        # Use PipelineConfig(...) directly with custom values
    """
    return PipelineConfig()


def get_semaphore(node_name: str) -> asyncio.Semaphore:
    """Get the asyncio.Semaphore for a specific node.

    Convenience function that creates a Semaphore from the config value.
    """
    config = get_pipeline_config()
    semaphore_map = {
        "direct_kg": config.direct_kg.sdk_semaphore,
        "entity_resolution": config.entity_resolution.sdk_semaphore,
        "triage": config.triage.sdk_semaphore,
        "cold_start": config.cold_start.sdk_semaphore,
    }
    value = semaphore_map.get(node_name)
    if value is None:
        raise ValueError(f"No semaphore configured for node: {node_name}")
    return asyncio.Semaphore(value)
