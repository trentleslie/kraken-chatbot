"""L2 — BridgeGrounding / LegSummary provenance models, Bridge.grounding field, and contracts.

The provenance label is attached to a frozen Bridge via model_copy and written to a separate
`grounded_bridges` state key (NOT `bridges`, whose operator.add reducer would duplicate). The
output contract defaults every field so a partial/early return still validates (never-throws).

Run with: uv run python -m pytest tests/test_bridge_grounding_models.py -v
"""

import pytest
from pydantic import ValidationError

from src.kestrel_backend.graph.state import Bridge, BridgeGrounding, LegSummary
from src.kestrel_backend.graph.state_contracts import (
    BridgeGroundingInput,
    BridgeGroundingOutput,
    NODE_CONTRACTS,
)


def _leg(from_curie="A", to_curie="B", evidence_tier="curated-causal"):
    return LegSummary(from_curie=from_curie, to_curie=to_curie, evidence_tier=evidence_tier)


def _bridge():
    return Bridge(
        path_description="metabolite → gene → disease",
        entities=["CHEBI:1", "HGNC:2", "MONDO:3"],
        entity_names=["a", "b", "c"],
        predicates=["biolink:affects", "biolink:related_to"],
        predicate_directions=[True, False],
        tier=2,
        novelty="known",
        significance="why",
    )


# --- LegSummary (minimal provenance shape) ---------------------------------------------

def test_leg_summary_minimal_fields():
    leg = LegSummary(from_curie="CHEBI:1", to_curie="HGNC:2", evidence_tier="text-mined")
    assert leg.from_curie == "CHEBI:1"
    assert leg.to_curie == "HGNC:2"
    assert leg.evidence_tier == "text-mined"


def test_leg_summary_is_frozen():
    leg = _leg()
    with pytest.raises(ValidationError):
        leg.evidence_tier = "none"  # frozen


def test_leg_summary_dropped_v1_tally_fields():
    # The v1 co-occurrence tally fields are gone (not merely optional).
    for gone in ("support", "refute", "neither", "off_topic", "pool_size", "dropped_co_mention"):
        assert gone not in LegSummary.model_fields


# --- BridgeGrounding (legs + chain label) ----------------------------------------------

def test_grounding_carries_legs_and_label():
    g = BridgeGrounding(
        legs=[_leg(evidence_tier="curated-causal"), _leg(from_curie="B", to_curie="C",
              evidence_tier="text-mined")],
        label="weakest leg text-mined",
    )
    assert g.label == "weakest leg text-mined"
    assert len(g.legs) == 2 and g.legs[0].evidence_tier == "curated-causal"


def test_grounding_legs_default_empty():
    g = BridgeGrounding(label="no KG edge")
    assert g.legs == []


def test_grounding_is_frozen():
    g = BridgeGrounding(label="both legs curated-causal")
    with pytest.raises(ValidationError):
        g.label = "changed"  # frozen


def test_grounding_dropped_v1_score_fields():
    # The v1 score/verdict fields are gone (not just optional) — the score is abandoned.
    for gone in ("support_fraction", "decision", "strong_leg_fraction", "strong_leg_n",
                 "ci_low", "ci_high", "rationale", "chain_pmids"):
        assert gone not in BridgeGrounding.model_fields


# --- Bridge.grounding field ------------------------------------------------------------

def test_bridge_grounding_defaults_none_backward_compatible():
    b = _bridge()
    assert b.grounding is None  # additive, optional — existing bridges deserialize unchanged


def test_attach_grounding_via_model_copy_does_not_mutate_original():
    b = _bridge()
    g = BridgeGrounding(legs=[_leg()], label="both legs curated-causal")
    grounded = b.model_copy(update={"grounding": g})
    assert grounded.grounding is g
    assert b.grounding is None  # frozen original untouched
    assert grounded.entities == b.entities  # everything else preserved


def test_bridge_roundtrips_with_grounding_serialized():
    b = _bridge().model_copy(update={"grounding": BridgeGrounding(
        legs=[_leg(evidence_tier="curated-associative")], label="weakest leg curated-associative")})
    restored = Bridge.model_validate(b.model_dump())
    assert restored.grounding is not None
    assert restored.grounding.label == "weakest leg curated-associative"
    assert restored.grounding.legs[0].evidence_tier == "curated-associative"


def test_grounded_bridges_join_back_by_entities_tuple():
    # grounded_bridges is a subset; consumers join to the original list by the entities tuple.
    b = _bridge()
    grounded = b.model_copy(update={"grounding": BridgeGrounding(
        legs=[_leg()], label="both legs curated-causal")})
    by_entities = {tuple(x.entities): x for x in [grounded]}
    assert by_entities[tuple(b.entities)].grounding.label == "both legs curated-causal"


# --- Contracts -------------------------------------------------------------------------

def test_output_contract_validates_empty_partial_return():
    # A partial/early return (no keys) must validate via defaults — never-throws guarantee.
    out = BridgeGroundingOutput()
    assert out.model_dump() == {
        "grounded_bridges": [], "bridge_grounding_errors": [], "model_usages": []}


def test_input_contract_requires_bridges():
    with pytest.raises(ValidationError):
        BridgeGroundingInput()
    assert BridgeGroundingInput(bridges=[]).bridges == []


def test_node_registered_in_contracts():
    assert NODE_CONTRACTS["bridge_grounding"] == (BridgeGroundingInput, BridgeGroundingOutput)
