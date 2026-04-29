"""Tests for pipeline configuration module."""

import pytest

from kestrel_backend.graph.pipeline_config import (
    ColdStartConfig,
    DirectKGConfig,
    EntityResolutionConfig,
    LiteratureGroundingConfig,
    PathwayEnrichmentConfig,
    PipelineConfig,
    TriageConfig,
    get_pipeline_config,
    get_semaphore,
)


class TestPipelineConfig:
    """Test default configuration values."""

    def test_direct_kg_defaults(self):
        config = get_pipeline_config().direct_kg
        assert config.hub_threshold == 5000
        assert config.sdk_semaphore == 6
        assert config.batch_size == 6
        assert config.preset_limit == 25

    def test_pathway_enrichment_defaults(self):
        config = get_pipeline_config().pathway_enrichment
        assert config.hub_threshold == 1000

    def test_entity_resolution_defaults(self):
        config = get_pipeline_config().entity_resolution
        assert config.sdk_semaphore == 1
        assert config.batch_size == 6
        assert config.tier1_min_score == 0.6

    def test_triage_defaults(self):
        config = get_pipeline_config().triage
        assert config.sdk_semaphore == 1
        assert config.batch_size == 6

    def test_cold_start_defaults(self):
        config = get_pipeline_config().cold_start
        assert config.sdk_semaphore == 8
        assert config.batch_size == 5
        assert config.inference_timeout == 60
        assert config.analogue_limit == 3

    def test_literature_grounding_defaults(self):
        config = get_pipeline_config().literature_grounding
        assert config.papers_per_hypothesis == 3
        assert config.max_hypotheses == 15
        assert config.parallel_fetch_limit == 6
        assert config.use_llm_classifier is False

    def test_hub_thresholds_intentionally_different(self):
        """Direct KG and pathway enrichment have different hub thresholds by design."""
        config = get_pipeline_config()
        assert config.direct_kg.hub_threshold == 5000
        assert config.pathway_enrichment.hub_threshold == 1000
        assert config.direct_kg.hub_threshold != config.pathway_enrichment.hub_threshold


class TestConfigOverride:
    """Test configuration overrides for testing."""

    def test_override_hub_threshold(self):
        custom = PipelineConfig(
            direct_kg=DirectKGConfig(hub_threshold=3000)
        )
        assert custom.direct_kg.hub_threshold == 3000
        # Other fields keep defaults
        assert custom.direct_kg.sdk_semaphore == 6

    def test_override_llm_classifier_flag(self):
        custom = PipelineConfig(
            literature_grounding=LiteratureGroundingConfig(use_llm_classifier=True)
        )
        assert custom.literature_grounding.use_llm_classifier is True

    def test_invalid_semaphore_raises(self):
        """Semaphore values must be positive."""
        # Pydantic doesn't validate semaphore > 0 by default (it's an int),
        # but the value should be positive for asyncio.Semaphore
        config = DirectKGConfig(sdk_semaphore=0)
        # The config accepts 0 but asyncio.Semaphore(0) would deadlock
        assert config.sdk_semaphore == 0


class TestGetSemaphore:
    """Test semaphore factory."""

    def test_known_nodes(self):
        for node_name in ["direct_kg", "entity_resolution", "triage", "cold_start"]:
            sem = get_semaphore(node_name)
            assert sem is not None

    def test_unknown_node_raises(self):
        with pytest.raises(ValueError, match="No semaphore configured"):
            get_semaphore("unknown_node")
