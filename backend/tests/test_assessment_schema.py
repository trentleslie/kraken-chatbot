"""Tests for assessment dataset format and JSON schema."""

import json
from pathlib import Path

import pytest

from kestrel_backend.assessment.report import (
    AssessmentReport,
    HypothesisJudgment,
    QueryReport,
)
from kestrel_backend.assessment.checks import CheckResult


class TestSchemaGeneration:
    """Test that Pydantic model generates valid JSON schema."""

    def test_schema_is_valid_json(self):
        schema = AssessmentReport.model_json_schema()
        # Should serialize without errors
        json_str = json.dumps(schema, indent=2)
        parsed = json.loads(json_str)
        assert "properties" in parsed

    def test_schema_has_version_field(self):
        schema = AssessmentReport.model_json_schema()
        assert "schema_version" in schema["properties"]

    def test_schema_includes_human_judgment(self):
        schema = AssessmentReport.model_json_schema()
        # human_judgment should be in QueryReport's definition
        defs = schema.get("$defs", {})
        assert "HypothesisJudgment" in defs
        hj_props = defs["HypothesisJudgment"]["properties"]
        assert "override_plausibility" in hj_props
        assert "override_relevance" in hj_props
        assert "override_novelty" in hj_props
        assert "notes" in hj_props


class TestExampleValidation:
    """Test that the example file validates against the schema."""

    def test_example_file_exists(self):
        example_path = Path(__file__).parent.parent / "assessment_data" / "schema" / "example.json"
        assert example_path.exists(), f"Example file not found at {example_path}"

    def test_example_validates_against_model(self):
        example_path = Path(__file__).parent.parent / "assessment_data" / "schema" / "example.json"
        data = json.loads(example_path.read_text())
        report = AssessmentReport.model_validate(data)
        assert report.schema_version == "v1"
        assert len(report.query_reports) == 1
        assert report.query_reports[0].passed

    def test_example_has_human_judgment_nulls(self):
        example_path = Path(__file__).parent.parent / "assessment_data" / "schema" / "example.json"
        data = json.loads(example_path.read_text())
        report = AssessmentReport.model_validate(data)
        for judgment in report.query_reports[0].human_judgment:
            assert judgment.override_plausibility is None
            assert judgment.override_relevance is None
            assert judgment.override_novelty is None


class TestHypothesisJudgment:
    """Test human judgment placeholder behavior."""

    def test_all_nulls_valid(self):
        judgment = HypothesisJudgment()
        assert judgment.override_plausibility is None
        assert judgment.notes is None

    def test_partial_fill_valid(self):
        judgment = HypothesisJudgment(
            override_plausibility=8,
            notes="Reviewed by domain expert",
        )
        assert judgment.override_plausibility == 8
        assert judgment.override_relevance is None

    def test_schema_version_forward_compat(self):
        """Report with different schema version should still parse."""
        report = AssessmentReport(
            schema_version="v2",
            mode="test",
            aggregate={},
        )
        assert report.schema_version == "v2"
