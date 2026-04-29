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


class PathwayEnrichmentConfig(BaseModel):
    """Configuration for the pathway_enrichment node."""

    hub_threshold: int = Field(
        default=1000,
        description="Nodes with more edges than this are flagged as hubs in "
        "shared-neighbor filtering. Lower than direct_kg (5000) because this "
        "filters neighbors, not entities — a different purpose.",
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
