"""Tests for state validation contracts.

Tests the @validate_state decorator and per-node input/output Pydantic models,
including path-conditional field handling and OR-semantics validation.
"""

import pytest

from kestrel_backend.graph.state_contracts import (
    ColdStartInput,
    ColdStartOutput,
    DirectKGInput,
    DirectKGOutput,
    EntityResolutionInput,
    EntityResolutionOutput,
    IntakeInput,
    IntakeOutput,
    IntegrationInput,
    IntegrationOutput,
    LiteratureGroundingInput,
    LiteratureGroundingOutput,
    PathwayEnrichmentInput,
    PathwayEnrichmentOutput,
    StateValidationError,
    SynthesisInput,
    SynthesisOutput,
    TemporalInput,
    TemporalOutput,
    TriageInput,
    TriageOutput,
    validate_state,
)


class TestIntakeContract:
    """Test intake node input/output validation."""

    def test_valid_input(self):
        model = IntakeInput.model_validate({"raw_query": "NAD+ and aging"})
        assert model.raw_query == "NAD+ and aging"

    def test_missing_raw_query_raises(self):
        with pytest.raises(Exception):
            IntakeInput.model_validate({})

    def test_valid_output(self):
        model = IntakeOutput.model_validate({
            "raw_entities": ["NAD+", "aging"],
            "query_type": "discovery",
        })
        assert model.raw_entities == ["NAD+", "aging"]

    def test_extra_fields_ignored(self):
        """State dict has many fields — extras should be silently ignored."""
        model = IntakeInput.model_validate({
            "raw_query": "test",
            "conversation_history": [],
            "unrelated_field": "value",
        })
        assert model.raw_query == "test"


class TestEntityResolutionContract:
    def test_valid_input(self):
        model = EntityResolutionInput.model_validate({"raw_entities": ["NAD+"]})
        assert model.raw_entities == ["NAD+"]

    def test_missing_raw_entities_raises(self):
        with pytest.raises(Exception):
            EntityResolutionInput.model_validate({})

    def test_valid_output(self):
        model = EntityResolutionOutput.model_validate({
            "resolved_entities": [{"raw_name": "NAD+", "curie": "CHEBI:15422"}],
        })
        assert len(model.resolved_entities) == 1


class TestTriageContract:
    def test_valid_input(self):
        model = TriageInput.model_validate({
            "resolved_entities": [{"raw_name": "NAD+", "curie": "CHEBI:15422"}],
        })
        assert len(model.resolved_entities) == 1

    def test_valid_output(self):
        model = TriageOutput.model_validate({
            "novelty_scores": [{"score": 0.8}],
            "well_characterized_curies": ["CHEBI:15422"],
            "moderate_curies": [],
            "sparse_curies": ["CHEBI:99999"],
            "cold_start_curies": [],
        })
        assert len(model.novelty_scores) == 1
        assert model.well_characterized_curies == ["CHEBI:15422"]
        assert model.sparse_curies == ["CHEBI:99999"]

    def test_missing_classification_buckets_raises(self):
        """Triage output must include all 4 classification bucket lists."""
        with pytest.raises(Exception):
            TriageOutput.model_validate({"novelty_scores": [{"score": 0.8}]})


class TestDirectKGContract:
    def test_valid_input(self):
        model = DirectKGInput.model_validate({
            "well_characterized_curies": ["CHEBI:15422"],
            "moderate_curies": [],
        })
        assert model.well_characterized_curies == ["CHEBI:15422"]

    def test_both_curie_lists_empty_raises(self):
        """Direct KG requires at least one of well_characterized or moderate CURIEs."""
        with pytest.raises(ValueError, match="at least one of well_characterized_curies or moderate_curies"):
            DirectKGInput.model_validate({})

    def test_moderate_only_valid(self):
        model = DirectKGInput.model_validate({
            "well_characterized_curies": [],
            "moderate_curies": ["CHEBI:12345"],
        })
        assert model.moderate_curies == ["CHEBI:12345"]

    def test_valid_output(self):
        model = DirectKGOutput.model_validate({
            "direct_findings": [{"text": "finding"}],
        })
        assert len(model.direct_findings) == 1


class TestColdStartContract:
    def test_valid_input(self):
        model = ColdStartInput.model_validate({
            "sparse_curies": ["CHEBI:99999"],
            "cold_start_curies": [],
        })
        assert model.sparse_curies == ["CHEBI:99999"]

    def test_both_curie_lists_empty_raises(self):
        """Cold start requires at least one of sparse or cold_start CURIEs."""
        with pytest.raises(ValueError, match="at least one of sparse_curies or cold_start_curies"):
            ColdStartInput.model_validate({})

    def test_cold_start_only_valid(self):
        model = ColdStartInput.model_validate({
            "sparse_curies": [],
            "cold_start_curies": ["UNKNOWN:1"],
        })
        assert model.cold_start_curies == ["UNKNOWN:1"]

    def test_valid_output(self):
        model = ColdStartOutput.model_validate({
            "cold_start_findings": [{"text": "inferred finding"}],
        })
        assert len(model.cold_start_findings) == 1


class TestIntegrationContract:
    """Test integration node's OR-semantics validation."""

    def test_valid_with_direct_findings_only(self):
        """Direct-KG-only path: cold_start_findings is None/empty."""
        model = IntegrationInput.model_validate({
            "direct_findings": [{"text": "finding"}],
            "cold_start_findings": None,
        })
        assert len(model.direct_findings) == 1

    def test_valid_with_cold_start_findings_only(self):
        """Cold-start-only path: direct_findings is None/empty."""
        model = IntegrationInput.model_validate({
            "direct_findings": None,
            "cold_start_findings": [{"text": "inferred"}],
        })
        assert len(model.cold_start_findings) == 1

    def test_valid_with_both_branches(self):
        """Both branches produced findings."""
        model = IntegrationInput.model_validate({
            "direct_findings": [{"text": "direct"}],
            "cold_start_findings": [{"text": "cold"}],
        })
        assert len(model.direct_findings) == 1
        assert len(model.cold_start_findings) == 1

    def test_both_empty_raises(self):
        """Neither branch produced findings — should raise."""
        with pytest.raises(ValueError, match="at least one of direct_findings or cold_start_findings"):
            IntegrationInput.model_validate({
                "direct_findings": [],
                "cold_start_findings": [],
            })

    def test_both_none_raises(self):
        """Neither branch ran — should raise."""
        with pytest.raises(ValueError, match="at least one of direct_findings or cold_start_findings"):
            IntegrationInput.model_validate({
                "direct_findings": None,
                "cold_start_findings": None,
            })

    def test_valid_output(self):
        model = IntegrationOutput.model_validate({
            "bridges": [{"source": "A", "target": "B"}],
            "gap_entities": [],
        })
        assert len(model.bridges) == 1


class TestSynthesisContract:
    """Test synthesis node's OR-semantics (same as integration)."""

    def test_valid_with_direct_findings(self):
        model = SynthesisInput.model_validate({
            "direct_findings": [{"text": "finding"}],
        })
        assert len(model.direct_findings) == 1

    def test_both_empty_raises(self):
        with pytest.raises(ValueError, match="at least one of direct_findings or cold_start_findings"):
            SynthesisInput.model_validate({
                "direct_findings": [],
                "cold_start_findings": [],
            })

    def test_valid_output(self):
        model = SynthesisOutput.model_validate({
            "synthesis_report": "# Report\n\nFindings...",
            "hypotheses": [{"claim": "test"}],
        })
        assert model.synthesis_report.startswith("# Report")


class TestTemporalContract:
    def test_valid_input_longitudinal(self):
        model = TemporalInput.model_validate({
            "is_longitudinal": True,
            "direct_findings": [{"text": "finding"}],
        })
        assert model.is_longitudinal is True

    def test_valid_input_non_longitudinal(self):
        model = TemporalInput.model_validate({"is_longitudinal": False})
        assert model.is_longitudinal is False

    def test_valid_output(self):
        model = TemporalOutput.model_validate({
            "temporal_classifications": [{"entity": "NAD+", "category": "upstream_cause"}],
        })
        assert len(model.temporal_classifications) == 1


class TestLiteratureGroundingContract:
    def test_valid_input(self):
        model = LiteratureGroundingInput.model_validate({
            "hypotheses": [{"claim": "test hypothesis"}],
        })
        assert len(model.hypotheses) == 1

    def test_missing_hypotheses_raises(self):
        with pytest.raises(Exception):
            LiteratureGroundingInput.model_validate({})


class TestValidateStateDecorator:
    """Test the @validate_state decorator."""

    async def test_valid_input_and_output(self):
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return {
                "raw_entities": ["NAD+"],
                "query_type": "discovery",
            }

        result = await run({"raw_query": "NAD+ and aging"})
        assert result["raw_entities"] == ["NAD+"]

    async def test_invalid_input_raises_state_validation_error(self):
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return {"raw_entities": [], "query_type": "discovery"}

        with pytest.raises(StateValidationError, match="input"):
            await run({})  # Missing raw_query

    async def test_invalid_output_raises_state_validation_error(self):
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return {"missing_field": True}  # Missing raw_entities and query_type

        with pytest.raises(StateValidationError, match="output"):
            await run({"raw_query": "test"})

    async def test_error_message_includes_node_name(self):
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return {"raw_entities": [], "query_type": "discovery"}

        try:
            await run({})
        except StateValidationError as e:
            assert "run" in e.node_name or "test" in e.node_name
            assert e.direction == "input"
            assert len(e.validation_errors) > 0

    async def test_extra_state_fields_accepted(self):
        """Decorator should not reject state dicts with extra fields."""
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return {"raw_entities": ["x"], "query_type": "discovery"}

        result = await run({
            "raw_query": "test",
            "conversation_history": [],
            "some_extra_field": "value",
        })
        assert result["raw_entities"] == ["x"]

    async def test_integration_or_semantics(self):
        """Integration decorator should enforce at least one findings branch."""
        @validate_state(IntegrationInput, IntegrationOutput)
        async def run(state):
            return {"bridges": [], "gap_entities": []}

        # Should fail: both branches empty
        with pytest.raises(StateValidationError, match="input"):
            await run({"direct_findings": [], "cold_start_findings": []})

        # Should pass: one branch has data
        result = await run({
            "direct_findings": [{"text": "finding"}],
            "cold_start_findings": [],
        })
        assert result["bridges"] == []

    async def test_none_return_accepted(self):
        """Some nodes may return None in edge cases — decorator should handle gracefully."""
        @validate_state(IntakeInput, IntakeOutput)
        async def run(state):
            return None

        result = await run({"raw_query": "test"})
        assert result is None
