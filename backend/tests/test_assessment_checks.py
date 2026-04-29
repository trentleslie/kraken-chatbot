"""Tests for structural checks and assessment report."""

import json
from pathlib import Path

import pytest

from kestrel_backend.assessment.checks import (
    CheckResult,
    check_entity_resolution_recall,
    check_finding_count_stability,
    check_hypothesis_completeness,
    check_pipeline_completion,
    check_schema_conformance,
    run_all_checks,
)
from kestrel_backend.assessment.report import (
    AssessmentReport,
    generate_report,
    save_report,
)


# === Shared test state ===

COMPLETE_STATE = {
    "raw_query": "NAD+ and aging",
    "raw_entities": ["NAD+", "aging"],
    "resolved_entities": [
        {"raw_name": "NAD+", "curie": "CHEBI:15422", "confidence": 0.95},
        {"raw_name": "aging", "curie": "HP:0011462", "confidence": 0.8},
    ],
    "novelty_scores": [{"curie": "CHEBI:15422", "edge_count": 500}],
    "direct_findings": [
        {"text": "NAD+ is involved in energy metabolism", "tier": "tier1"},
        {"text": "NAD+ declines with age", "tier": "tier1"},
    ],
    "cold_start_findings": [],
    "synthesis_report": "# Analysis\n\nNAD+ metabolism and aging...",
    "hypotheses": [
        {"claim": "NAD+ decline drives aging", "evidence": ["finding1"], "novelty_score": 0.7},
        {"claim": "NMN restores NAD+", "evidence": ["finding2"], "novelty_score": 0.5},
    ],
    "errors": [],
}


class TestCheckPipelineCompletion:
    def test_all_nodes_present(self):
        result = check_pipeline_completion(COMPLETE_STATE)
        assert result.passed
        assert result.status == "pass"

    def test_missing_synthesis(self):
        state = {**COMPLETE_STATE, "synthesis_report": ""}
        result = check_pipeline_completion(state)
        assert not result.passed
        assert "synthesis" in result.message

    def test_missing_entities(self):
        state = {**COMPLETE_STATE, "raw_entities": []}
        result = check_pipeline_completion(state)
        assert not result.passed
        assert "intake" in result.message

    def test_empty_state(self):
        result = check_pipeline_completion({})
        assert not result.passed


class TestCheckSchemaConformance:
    def test_valid_types(self):
        result = check_schema_conformance(COMPLETE_STATE)
        assert result.passed

    def test_wrong_type_synthesis_report(self):
        state = {**COMPLETE_STATE, "synthesis_report": 42}
        result = check_schema_conformance(state)
        assert not result.passed
        assert "synthesis_report" in result.message

    def test_none_fields_accepted(self):
        """None values should not trigger type violations."""
        state = {**COMPLETE_STATE, "hypotheses": None}
        result = check_schema_conformance(state)
        assert result.passed


class TestCheckEntityResolutionRecall:
    def test_all_expected_found(self):
        result = check_entity_resolution_recall(
            COMPLETE_STATE, expected_curies=["CHEBI:15422", "HP:0011462"]
        )
        assert result.passed
        assert result.status == "pass"

    def test_missing_expected_curie(self):
        result = check_entity_resolution_recall(
            COMPLETE_STATE, expected_curies=["CHEBI:15422", "MISSING:999"]
        )
        assert not result.passed
        assert "MISSING:999" in result.message

    def test_extra_curies_accepted(self):
        """Extra resolved CURIEs beyond expected should pass (recall, not exact match)."""
        result = check_entity_resolution_recall(
            COMPLETE_STATE, expected_curies=["CHEBI:15422"]
        )
        assert result.passed

    def test_no_expected_curies_skipped(self):
        result = check_entity_resolution_recall(COMPLETE_STATE, expected_curies=None)
        assert result.passed
        assert "skipping" in result.message.lower()


class TestCheckFindingCountStability:
    def test_within_tolerance_band(self):
        baseline = {
            "metric_bands": {
                "direct_finding_count": {
                    "mean": 2.0, "stddev": 0.5,
                    "lower_bound": 1.0, "upper_bound": 3.0,
                    "cv": 0.25, "status": "ok",
                }
            }
        }
        result = check_finding_count_stability(COMPLETE_STATE, baseline)
        assert result.passed
        assert result.status == "pass"

    def test_outside_tolerance_band_fails(self):
        baseline = {
            "metric_bands": {
                "direct_finding_count": {
                    "mean": 10.0, "stddev": 0.5,
                    "lower_bound": 9.0, "upper_bound": 11.0,
                    "cv": 0.05, "status": "ok",
                }
            }
        }
        result = check_finding_count_stability(COMPLETE_STATE, baseline)
        assert not result.passed
        assert result.status == "fail"

    def test_high_variance_produces_warning(self):
        """CV > 0.5 should produce warning, not fail."""
        baseline = {
            "metric_bands": {
                "direct_finding_count": {
                    "mean": 10.0, "stddev": 6.0,
                    "lower_bound": 0.0, "upper_bound": 22.0,
                    "cv": 0.6, "status": "warning",
                }
            }
        }
        # State has 2 direct findings, which is outside [0, 22] — wait no, 2 is in [0, 22]
        # Let me make it actually outside
        state = {**COMPLETE_STATE, "direct_findings": [{"text": f"f{i}"} for i in range(25)]}
        result = check_finding_count_stability(state, baseline)
        assert result.passed  # warnings count as pass
        assert result.status == "warning"

    def test_no_baseline_skipped(self):
        result = check_finding_count_stability(COMPLETE_STATE, None)
        assert result.passed

    def test_zero_findings_detected(self):
        state = {**COMPLETE_STATE, "direct_findings": []}
        baseline = {
            "metric_bands": {
                "direct_finding_count": {
                    "mean": 5.0, "stddev": 1.0,
                    "lower_bound": 3.0, "upper_bound": 7.0,
                    "cv": 0.2, "status": "ok",
                }
            }
        }
        result = check_finding_count_stability(state, baseline)
        assert not result.passed


class TestCheckHypothesisCompleteness:
    def test_complete_hypotheses(self):
        result = check_hypothesis_completeness(COMPLETE_STATE)
        assert result.passed

    def test_missing_claim_field(self):
        state = {
            **COMPLETE_STATE,
            "hypotheses": [{"evidence": ["e1"], "novelty_score": 0.5}],
        }
        result = check_hypothesis_completeness(state)
        assert not result.passed
        assert "claim" in result.message

    def test_no_hypotheses(self):
        state = {**COMPLETE_STATE, "hypotheses": []}
        result = check_hypothesis_completeness(state)
        assert result.passed


class TestRunAllChecks:
    def test_returns_five_results(self):
        results = run_all_checks(COMPLETE_STATE)
        assert len(results) == 5
        assert all(isinstance(r, CheckResult) for r in results)

    def test_all_pass_on_complete_state(self):
        results = run_all_checks(COMPLETE_STATE)
        assert all(r.passed for r in results)


class TestAssessmentReport:
    def test_generate_report(self):
        assessment_results = {
            "summary": {"mode": "replay"},
            "results": [
                {
                    "query": "NAD+ and aging",
                    "query_hash": "abc123",
                    "state": COMPLETE_STATE,
                    "metadata": {"path_type": "well-characterized"},
                    "error": None,
                },
            ],
        }
        report = generate_report(assessment_results)
        assert report.schema_version == "v1"
        assert len(report.query_reports) == 1
        assert report.query_reports[0].passed
        assert report.aggregate["passed"] == 1
        assert report.aggregate["failed"] == 0

    def test_report_includes_human_judgment_placeholders(self):
        assessment_results = {
            "summary": {"mode": "replay"},
            "results": [
                {
                    "query": "test",
                    "query_hash": "def456",
                    "state": COMPLETE_STATE,
                    "metadata": {},
                    "error": None,
                },
            ],
        }
        report = generate_report(assessment_results)
        qr = report.query_reports[0]
        # Should have one judgment placeholder per hypothesis
        assert len(qr.human_judgment) == len(COMPLETE_STATE["hypotheses"])
        assert all(j.override_plausibility is None for j in qr.human_judgment)

    def test_skipped_queries_excluded(self):
        assessment_results = {
            "summary": {"mode": "replay"},
            "results": [
                {"query": "failed", "query_hash": "x", "state": None, "metadata": {}, "error": "API down"},
            ],
        }
        report = generate_report(assessment_results)
        assert len(report.query_reports) == 0

    def test_save_report(self, tmp_path: Path):
        report = AssessmentReport(mode="test", aggregate={"total": 0})
        output_path = tmp_path / "report.json"
        save_report(report, output_path)
        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert loaded["schema_version"] == "v1"

    def test_report_json_serializable(self):
        assessment_results = {
            "summary": {"mode": "replay"},
            "results": [
                {
                    "query": "test",
                    "query_hash": "abc",
                    "state": COMPLETE_STATE,
                    "metadata": {},
                    "error": None,
                },
            ],
        }
        report = generate_report(assessment_results)
        # Should serialize without errors
        json_str = report.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "v1"
        assert len(parsed["query_reports"]) == 1
