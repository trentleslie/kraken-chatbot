"""Tests for state validation contracts and state models.

Tests the @validate_state decorator and per-node input/output Pydantic models,
including path-conditional field handling and OR-semantics validation.
Also tests ModelUsageRecord and the model_usages reducer field.
"""

import operator
import pytest

from kestrel_backend.graph.state import ModelUsageRecord
from kestrel_backend.graph.state_contracts import (
    ColdStartInput,
    ColdStartOutput,
    DirectKGInput,
    DirectKGOutput,
    EntityResolutionInput,
    EntityResolutionOutput,
    HypothesisExtractionInput,
    HypothesisExtractionOutput,
    IntakeInput,
    IntakeOutput,
    IntegrationInput,
    IntegrationOutput,
    LiteratureGroundingInput,
    LiteratureGroundingOutput,
    NODE_CONTRACTS,
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

    def test_valid_output_report_only(self):
        """SynthesisOutput requires only synthesis_report now — hypotheses are produced
        upstream (hypothesis_extraction → literature_grounding), so synthesis stops emitting them."""
        model = SynthesisOutput.model_validate({
            "synthesis_report": "# Report\n\nFindings...",
        })
        assert model.synthesis_report.startswith("# Report")
        assert "hypotheses" not in SynthesisOutput.model_fields

    def test_input_accepts_hypotheses_optional(self):
        """SynthesisInput exposes hypotheses as an available Optional input (present and absent)."""
        with_hyps = SynthesisInput.model_validate({
            "direct_findings": [{"text": "finding"}],
            "hypotheses": [{"claim": "grounded"}],
        })
        assert len(with_hyps.hypotheses) == 1
        without_hyps = SynthesisInput.model_validate({
            "direct_findings": [{"text": "finding"}],
        })
        assert without_hyps.hypotheses is None


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

    def test_missing_hypotheses_degrades_not_raises(self):
        """hypotheses is Optional now (R13): grounding runs upstream of synthesis, so a missing
        key must degrade to grounding's empty no-op, NOT raise StateValidationError at input."""
        model = LiteratureGroundingInput.model_validate({})
        assert model.hypotheses is None

    def test_empty_hypotheses_does_not_raise(self):
        """hypotheses: [] is the well-characterized-only / degrade case — must validate cleanly."""
        model = LiteratureGroundingInput.model_validate({"hypotheses": []})
        assert model.hypotheses == []

    def test_output_carries_literature_errors_first_class(self):
        """literature_errors is a declared output field, not an extra='ignore' passthrough."""
        assert "literature_errors" in LiteratureGroundingOutput.model_fields
        model = LiteratureGroundingOutput.model_validate({
            "hypotheses": [{"claim": "grounded"}],
            "literature_errors": ["S2 rate limited (429)"],
        })
        assert model.literature_errors == ["S2 rate limited (429)"]

    def test_output_literature_errors_defaults_empty(self):
        """A clean grounding run with no errors still validates (default [])."""
        model = LiteratureGroundingOutput.model_validate({"hypotheses": []})
        assert model.literature_errors == []


class TestHypothesisExtractionContract:
    """Unit 1 — new node that validates bridges then extracts hypotheses (moved out of synthesis)."""

    def test_valid_input_direct_findings_and_bridges(self):
        """Happy path: non-empty direct_findings with bridges present."""
        model = HypothesisExtractionInput.model_validate({
            "direct_findings": [{"text": "finding"}],
            "bridges": [{"source": "A", "target": "B"}],
        })
        assert len(model.direct_findings) == 1
        assert len(model.bridges) == 1

    def test_valid_input_cold_start_only(self):
        """Edge case: OR-semantics — cold_start_findings only, no direct, no bridges."""
        model = HypothesisExtractionInput.model_validate({
            "direct_findings": None,
            "cold_start_findings": [{"text": "inferred"}],
        })
        assert len(model.cold_start_findings) == 1

    def test_both_findings_empty_raises(self):
        """Error path: neither findings branch produced results (mirrors IntegrationInput)."""
        with pytest.raises(ValueError, match="at least one of direct_findings or cold_start_findings"):
            HypothesisExtractionInput.model_validate({
                "direct_findings": [],
                "cold_start_findings": None,
            })

    def test_valid_output_bridges_and_hypotheses(self):
        """Output requires both bridges (re-emitted validated) and hypotheses (may be empty)."""
        model = HypothesisExtractionOutput.model_validate({
            "bridges": [{"source": "A", "target": "B"}],
            "hypotheses": [],
        })
        assert len(model.bridges) == 1
        assert model.hypotheses == []

    def test_output_missing_bridges_raises(self):
        with pytest.raises(Exception):
            HypothesisExtractionOutput.model_validate({"hypotheses": []})

    def test_output_missing_hypotheses_raises(self):
        with pytest.raises(Exception):
            HypothesisExtractionOutput.model_validate({"bridges": []})

    def test_registered_in_node_contracts(self):
        """Integration: NODE_CONTRACTS resolves the new node to its input/output pair."""
        in_model, out_model = NODE_CONTRACTS["hypothesis_extraction"]
        assert in_model is HypothesisExtractionInput
        assert out_model is HypothesisExtractionOutput


class TestConcatListFields:
    """R12 — dropping the bridges operator.add reducer must auto-remove it from CONCAT_LIST_FIELDS."""

    def test_bridges_not_in_concat_fields(self):
        from kestrel_backend.main import CONCAT_LIST_FIELDS
        assert "bridges" not in CONCAT_LIST_FIELDS
        # gap_entities keeps its reducer — sanity-check the derivation still finds real ones.
        assert "gap_entities" in CONCAT_LIST_FIELDS


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


class TestModelUsageRecord:
    """Test ModelUsageRecord Pydantic model for cost tracking."""

    def test_instantiate_with_all_fields(self):
        record = ModelUsageRecord(
            model_name="anthropic/claude-sonnet-4-20250514",
            node_name="synthesis",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_creation_tokens=5,
        )
        assert record.model_name == "anthropic/claude-sonnet-4-20250514"
        assert record.node_name == "synthesis"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cache_read_tokens == 10
        assert record.cache_creation_tokens == 5

    def test_serializes_to_dict(self):
        record = ModelUsageRecord(
            model_name="anthropic/claude-sonnet-4-20250514",
            node_name="triage",
            input_tokens=200,
            output_tokens=100,
        )
        d = record.model_dump()
        assert d["model_name"] == "anthropic/claude-sonnet-4-20250514"
        assert d["node_name"] == "triage"
        assert d["input_tokens"] == 200
        assert d["output_tokens"] == 100
        assert d["cache_read_tokens"] == 0
        assert d["cache_creation_tokens"] == 0

    def test_default_token_values_are_zero(self):
        record = ModelUsageRecord(
            model_name="test-model",
            node_name="test-node",
        )
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.cache_read_tokens == 0
        assert record.cache_creation_tokens == 0

    def test_negative_tokens_rejected(self):
        with pytest.raises(Exception):
            ModelUsageRecord(
                model_name="test-model",
                node_name="test-node",
                input_tokens=-1,
            )

    def test_frozen_rejects_mutation(self):
        record = ModelUsageRecord(
            model_name="test-model",
            node_name="test-node",
            input_tokens=100,
        )
        with pytest.raises(Exception):
            record.input_tokens = 200

    def test_operator_add_reducer_merges_lists(self):
        """Verify that operator.add correctly merges model_usages lists."""
        list_a = [
            ModelUsageRecord(model_name="m", node_name="direct_kg", input_tokens=100),
        ]
        list_b = [
            ModelUsageRecord(model_name="m", node_name="cold_start", input_tokens=200),
        ]
        merged = operator.add(list_a, list_b)
        assert len(merged) == 2
        assert merged[0].node_name == "direct_kg"
        assert merged[1].node_name == "cold_start"
