"""U1 — BridgeGrounding / LegSummary models, Bridge.grounding field, and contracts.

The grounding signal is attached to a frozen Bridge via model_copy and written to a separate
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


def _leg(support=2, refute=0, neither=1, off_topic=0):
    return LegSummary(
        from_curie="A", to_curie="B", pool_size=10, abstracts_with_bodies=8,
        support=support, refute=refute, neither=neither, off_topic=off_topic,
    )


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


# --- BridgeGrounding model -------------------------------------------------------------

def test_grounding_minimal_defaults():
    g = BridgeGrounding(support_fraction=0.67, decision="grounded")
    assert g.support_fraction == 0.67
    assert g.decision == "grounded"
    # v2-only and secondary fields default to None / empty in v1.
    assert g.ci_low is None and g.ci_high is None
    assert g.strong_leg_fraction is None and g.strong_leg_n is None
    assert g.legs == [] and g.chain_pmids == [] and g.rationale == ""


def test_grounding_is_frozen():
    g = BridgeGrounding(support_fraction=0.5, decision="ungrounded")
    with pytest.raises(ValidationError):
        g.support_fraction = 0.9  # frozen


def test_support_fraction_bounds_enforced():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            BridgeGrounding(support_fraction=bad, decision="grounded")


def test_decision_enum_constrained():
    with pytest.raises(ValidationError):
        BridgeGrounding(support_fraction=0.5, decision="totally_grounded")


def test_grounding_carries_legs_and_strong_leg():
    g = BridgeGrounding(
        support_fraction=0.55, strong_leg_fraction=0.95, strong_leg_n=20,
        decision="grounded", legs=[_leg(), _leg(support=19, neither=1)],
        rationale="A affects B which relates to C", chain_pmids=["1", "2"],
    )
    assert g.strong_leg_fraction == 0.95 and g.strong_leg_n == 20
    assert len(g.legs) == 2 and g.legs[0].support == 2


# --- LegSummary ------------------------------------------------------------------------

def test_leg_summary_defaults():
    leg = LegSummary(from_curie="A", to_curie="B")
    assert leg.pool_size == 0 and leg.support == 0 and leg.dropped_co_mention == 0


# --- Bridge.grounding field ------------------------------------------------------------

def test_bridge_grounding_defaults_none_backward_compatible():
    b = _bridge()
    assert b.grounding is None  # additive, optional — existing bridges deserialize unchanged


def test_attach_grounding_via_model_copy_does_not_mutate_original():
    b = _bridge()
    g = BridgeGrounding(support_fraction=0.67, decision="grounded")
    grounded = b.model_copy(update={"grounding": g})
    assert grounded.grounding is g
    assert b.grounding is None  # frozen original untouched
    assert grounded.entities == b.entities  # everything else preserved


def test_bridge_roundtrips_with_grounding_serialized():
    b = _bridge().model_copy(update={"grounding": BridgeGrounding(
        support_fraction=0.5, decision="ungrounded")})
    restored = Bridge.model_validate(b.model_dump())
    assert restored.grounding is not None
    assert restored.grounding.decision == "ungrounded"


def test_grounded_bridges_join_back_by_entities_tuple():
    # grounded_bridges is a subset; consumers join to the original list by the entities tuple.
    b = _bridge()
    grounded = b.model_copy(update={"grounding": BridgeGrounding(
        support_fraction=0.8, decision="grounded")})
    by_entities = {tuple(x.entities): x for x in [grounded]}
    assert by_entities[tuple(b.entities)].grounding.support_fraction == 0.8


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
