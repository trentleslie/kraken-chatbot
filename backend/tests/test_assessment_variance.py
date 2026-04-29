"""Tests for variance computation and tolerance band generation."""

import json
from pathlib import Path

import pytest

from kestrel_backend.assessment.variance import (
    CV_WARNING_THRESHOLD,
    extract_metrics,
    compute_tolerance_bands,
    select_canonical_run,
)


class TestExtractMetrics:
    """Test metric extraction from pipeline state."""

    def test_extract_from_populated_state(self):
        state = {
            "resolved_entities": [
                {"raw_name": "NAD+", "curie": "CHEBI:15422"},
                {"raw_name": "SIRT1", "curie": "HGNC:14929"},
            ],
            "direct_findings": [{"text": "finding1"}, {"text": "finding2"}],
            "cold_start_findings": [],
            "hypotheses": [{"claim": "h1"}],
            "shared_neighbors": [{"entity": "n1"}, {"entity": "n2"}, {"entity": "n3"}],
            "bridges": [{"source": "a", "target": "b"}],
            "errors": [],
            "raw_entities": ["NAD+", "SIRT1"],
            "novelty_scores": [{"curie": "CHEBI:15422"}],
            "synthesis_report": "report text",
        }
        metrics = extract_metrics(state)

        assert metrics["resolved_entity_count"] == 2
        assert metrics["direct_finding_count"] == 2
        assert metrics["cold_start_finding_count"] == 0
        assert metrics["hypothesis_count"] == 1
        assert metrics["shared_neighbor_count"] == 3
        assert metrics["bridge_count"] == 1
        assert metrics["error_count"] == 0
        assert metrics["resolved_curies"] == ["CHEBI:15422", "HGNC:14929"]
        assert "intake" in metrics["node_execution_set"]
        assert "entity_resolution" in metrics["node_execution_set"]
        assert "synthesis" in metrics["node_execution_set"]

    def test_extract_from_empty_state(self):
        metrics = extract_metrics({})

        assert metrics["resolved_entity_count"] == 0
        assert metrics["hypothesis_count"] == 0
        assert metrics["resolved_curies"] == []
        assert metrics["node_execution_set"] == []

    def test_extract_curies_sorted(self):
        state = {
            "resolved_entities": [
                {"raw_name": "B", "curie": "HGNC:999"},
                {"raw_name": "A", "curie": "CHEBI:111"},
            ],
        }
        metrics = extract_metrics(state)
        assert metrics["resolved_curies"] == ["CHEBI:111", "HGNC:999"]

    def test_extract_skips_none_curies(self):
        state = {
            "resolved_entities": [
                {"raw_name": "NAD+", "curie": "CHEBI:15422"},
                {"raw_name": "unknown", "curie": None},
            ],
        }
        metrics = extract_metrics(state)
        assert metrics["resolved_curies"] == ["CHEBI:15422"]


class TestComputeToleranceBands:
    """Test tolerance band computation."""

    def test_basic_tolerance_bands(self):
        runs = [
            {"resolved_entity_count": 3, "direct_finding_count": 10, "cold_start_finding_count": 0,
             "hypothesis_count": 5, "shared_neighbor_count": 8, "bridge_count": 2, "error_count": 0,
             "resolved_curies": ["A", "B", "C"], "node_execution_set": ["intake", "synthesis"]},
            {"resolved_entity_count": 3, "direct_finding_count": 12, "cold_start_finding_count": 0,
             "hypothesis_count": 4, "shared_neighbor_count": 7, "bridge_count": 3, "error_count": 0,
             "resolved_curies": ["A", "B", "C"], "node_execution_set": ["intake", "synthesis"]},
            {"resolved_entity_count": 3, "direct_finding_count": 11, "cold_start_finding_count": 0,
             "hypothesis_count": 5, "shared_neighbor_count": 9, "bridge_count": 2, "error_count": 0,
             "resolved_curies": ["A", "B", "C"], "node_execution_set": ["intake", "synthesis"]},
        ]
        bands = compute_tolerance_bands(runs)

        assert bands["run_count"] == 3
        assert bands["metric_bands"]["resolved_entity_count"]["mean"] == 3.0
        assert bands["metric_bands"]["resolved_entity_count"]["stddev"] == 0.0
        assert bands["metric_bands"]["direct_finding_count"]["mean"] == 11.0
        assert bands["metric_bands"]["direct_finding_count"]["status"] == "ok"

        # Deterministic fields should be consistent
        assert bands["metric_bands"]["resolved_curies"]["consistent"] is True
        assert bands["metric_bands"]["node_execution_set"]["consistent"] is True

    def test_high_variance_warning(self):
        # CV > 0.5 triggers warning
        runs = [
            {"resolved_entity_count": 2, "direct_finding_count": 5, "cold_start_finding_count": 0,
             "hypothesis_count": 1, "shared_neighbor_count": 0, "bridge_count": 0, "error_count": 0,
             "resolved_curies": [], "node_execution_set": []},
            {"resolved_entity_count": 2, "direct_finding_count": 20, "cold_start_finding_count": 0,
             "hypothesis_count": 8, "shared_neighbor_count": 0, "bridge_count": 0, "error_count": 0,
             "resolved_curies": [], "node_execution_set": []},
        ]
        bands = compute_tolerance_bands(runs)

        # direct_finding_count has high variance (5 vs 20)
        dfc = bands["metric_bands"]["direct_finding_count"]
        assert dfc["cv"] > CV_WARNING_THRESHOLD
        assert dfc["status"] == "warning"
        assert any("direct_finding_count" in w for w in bands["warnings"])

    def test_single_run_insufficient_data(self):
        runs = [
            {"resolved_entity_count": 5, "direct_finding_count": 10, "cold_start_finding_count": 0,
             "hypothesis_count": 3, "shared_neighbor_count": 4, "bridge_count": 1, "error_count": 0,
             "resolved_curies": [], "node_execution_set": []},
        ]
        bands = compute_tolerance_bands(runs)
        assert bands["metric_bands"]["resolved_entity_count"]["status"] == "insufficient_data"

    def test_empty_runs(self):
        bands = compute_tolerance_bands([])
        assert "error" in bands

    def test_inconsistent_deterministic_fields(self):
        runs = [
            {"resolved_entity_count": 2, "direct_finding_count": 5, "cold_start_finding_count": 0,
             "hypothesis_count": 1, "shared_neighbor_count": 0, "bridge_count": 0, "error_count": 0,
             "resolved_curies": ["A", "B"], "node_execution_set": ["intake"]},
            {"resolved_entity_count": 2, "direct_finding_count": 5, "cold_start_finding_count": 0,
             "hypothesis_count": 1, "shared_neighbor_count": 0, "bridge_count": 0, "error_count": 0,
             "resolved_curies": ["A", "C"], "node_execution_set": ["intake"]},
        ]
        bands = compute_tolerance_bands(runs)
        assert bands["metric_bands"]["resolved_curies"]["consistent"] is False
        assert any("resolved_curies" in w for w in bands["warnings"])

    def test_query_metadata_included(self):
        runs = [
            {"resolved_entity_count": 3, "direct_finding_count": 10, "cold_start_finding_count": 0,
             "hypothesis_count": 5, "shared_neighbor_count": 8, "bridge_count": 2, "error_count": 0,
             "resolved_curies": [], "node_execution_set": []},
        ] * 3
        metadata = {"path_type": "well-characterized", "query": "test"}
        bands = compute_tolerance_bands(runs, query_metadata=metadata)
        assert bands["query_metadata"]["path_type"] == "well-characterized"


class TestSelectCanonicalRun:
    """Test canonical run selection."""

    def test_selects_median_finding_count(self):
        runs = [
            {"direct_finding_count": 5, "cold_start_finding_count": 0},
            {"direct_finding_count": 10, "cold_start_finding_count": 0},
            {"direct_finding_count": 15, "cold_start_finding_count": 0},
            {"direct_finding_count": 8, "cold_start_finding_count": 0},
            {"direct_finding_count": 12, "cold_start_finding_count": 0},
        ]
        outputs = [{"run": i} for i in range(5)]

        idx, output = select_canonical_run(runs, outputs)
        # Median is 10, which is index 1
        selected_count = runs[idx]["direct_finding_count"]
        assert selected_count == 10

    def test_includes_cold_start_findings(self):
        runs = [
            {"direct_finding_count": 0, "cold_start_finding_count": 5},
            {"direct_finding_count": 0, "cold_start_finding_count": 10},
            {"direct_finding_count": 0, "cold_start_finding_count": 7},
        ]
        outputs = [{"run": i} for i in range(3)]

        idx, output = select_canonical_run(runs, outputs)
        total = runs[idx]["direct_finding_count"] + runs[idx]["cold_start_finding_count"]
        assert total == 7  # Median of [5, 10, 7] = 7

    def test_single_run(self):
        runs = [{"direct_finding_count": 5, "cold_start_finding_count": 2}]
        outputs = [{"run": 0}]

        idx, output = select_canonical_run(runs, outputs)
        assert idx == 0
        assert output == {"run": 0}
