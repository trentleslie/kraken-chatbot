"""Axis E Unit 2: bridge-specificity rendering (consume axis-C `specificity_by_bridge`).

Plan: docs/plans/2026-07-16-001-feat-tier3-falsifier-direction-awareness-plan.md

Specificity lives in a state side-map keyed by `tuple(entities)` (mirroring grounded_bridges), NOT
on the frozen Bridge. `generic` bridges are down-weighted and name their generic intermediates,
framed as genericity — never as low mechanism confidence. Absent entry ⇒ the line is omitted and the
bridge section stays byte-identical to today.
"""

from types import SimpleNamespace

from kestrel_backend.graph.nodes.synthesis import (
    format_bridges,
    render_bridge_specificity,
    specificity_by_bridge_from_state,
)
from kestrel_backend.graph.state import Bridge


def _bridge(entities, tier=2):
    return Bridge(
        path_description="A → B → C", entities=entities, entity_names=[], predicates=["p1", "p2"],
        tier=tier, novelty="inferred", significance="matters",
    )


def _spec(label, generic_intermediates=None):
    return SimpleNamespace(
        score=0.5, label=label, intermediate_curies=["CHEBI:1"],
        intermediate_degrees=[42], generic_intermediates=generic_intermediates or [],
    )


# --- render_bridge_specificity --------------------------------------------------------


def test_specific_and_moderate_render_plain_label():
    assert render_bridge_specificity(_spec("specific")) == "**Bridge specificity**: specific"
    assert render_bridge_specificity(_spec("moderate")) == "**Bridge specificity**: moderate"


def test_generic_is_downweighted_and_names_intermediates():
    line = render_bridge_specificity(_spec("generic", generic_intermediates=["CHEBI:9999"]))
    assert "generic" in line
    assert "CHEBI:9999" in line
    assert "down-weight" in line
    assert "low confidence" not in line.lower()  # framed as genericity, not low mechanism confidence


def test_generic_without_named_intermediates_still_renders_generic():
    line = render_bridge_specificity(_spec("generic", generic_intermediates=[]))
    assert line.startswith("**Bridge specificity**: generic")


def test_absent_or_labelless_spec_omits_line():
    assert render_bridge_specificity(None) is None
    assert render_bridge_specificity(_spec("")) is None


# --- specificity_by_bridge_from_state -------------------------------------------------


def test_reader_normalizes_list_keys_to_tuples():
    state = {"specificity_by_bridge": {("A", "B", "C"): _spec("specific")}}
    m = specificity_by_bridge_from_state(state)
    assert ("A", "B", "C") in m


def test_reader_absent_seam_is_empty():
    assert specificity_by_bridge_from_state({}) == {}


def test_reader_bad_shape_is_empty_not_crash():
    assert specificity_by_bridge_from_state({"specificity_by_bridge": "not-a-map"}) == {}


# --- format_bridges integration -------------------------------------------------------


def test_format_bridges_renders_specificity_when_present():
    bridges = [_bridge(["A", "B", "C"], tier=2)]
    spec_map = {("A", "B", "C"): _spec("generic", generic_intermediates=["CHEBI:9999"])}
    out = format_bridges(bridges, None, spec_map)
    assert "Bridge specificity" in out
    assert "CHEBI:9999" in out


def test_format_bridges_identical_without_specificity():
    bridges = [_bridge(["A", "B", "C"], tier=2)]
    without = format_bridges(bridges)
    with_absent = format_bridges(bridges, None, {("X", "Y"): _spec("specific")})  # no matching key
    assert without == with_absent
    assert "Bridge specificity" not in without
