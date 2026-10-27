"""Axis E Unit 1: deterministic Tier-3 direction helper + entity→weight join + small-n cap.

Plan: docs/plans/2026-07-16-001-feat-tier3-falsifier-direction-awareness-plan.md

The helper joins a Tier-3 hypothesis to axis A's signed module weights (`module_spine`) and computes
a deterministic ↑/↓/indeterminate label with a capped confidence tier. It reads the seam defensively
(seam absent ⇒ omit; present-but-conflicting ⇒ `indeterminate`) and never silently picks a sign.
The `module_spine` seam is built here from SimpleNamespaces to prove E does not depend on axis A's
concrete pydantic models.
"""

from types import SimpleNamespace

from kestrel_backend.graph.nodes.synthesis import (
    DirectionResult,
    compute_tier3_direction,
    render_direction_value,
    tier3_direction_map,
)
from kestrel_backend.graph.pipeline_config import SynthesisConfig
from kestrel_backend.graph.state import EntityResolution, Hypothesis


# --- seam + fixture builders ----------------------------------------------------------


def _member(kme, kim=None):
    return SimpleNamespace(kme=kme, kim=kim)


def _spine(members, corr=None):
    """Axis-A ModuleSpine shape: members keyed by canonical (lowercased) name, optional direction."""
    direction = SimpleNamespace(eigengene_trait_correlation=corr) if corr is not None else None
    return SimpleNamespace(members=members, direction=direction)


def _entity(curie, name):
    return EntityResolution(
        raw_name=name, curie=curie, resolved_name=name, category="biolink:Gene",
        confidence=0.9, method="exact",
    )


def _hyp(title, curies, tier=3):
    return Hypothesis(
        title=title, tier=tier, claim="c", supporting_entities=curies,
        structural_logic="logic", validation_steps=["step"],
    )


def _cfg(**kw):
    return SynthesisConfig(**kw)


def _state(module_spine=None, resolved=None):
    st = {"resolved_entities": resolved or [_entity("NCBIGene:1", "GeneX")]}
    if module_spine is not None:
        st["module_spine"] = module_spine
    return st


# --- happy paths ----------------------------------------------------------------------


def test_kme_pos_direction_pos_is_up():
    state = _state({"modA": _spine({"genex": _member(0.8)}, corr=0.5)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert res.direction == "up"
    assert res.computable is True and res.seam_present is True
    assert render_direction_value(res) == "↑ (confidence: high)"


def test_kme_pos_direction_neg_is_down():
    state = _state({"modA": _spine({"genex": _member(0.8)}, corr=-0.5)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert res.direction == "down"
    assert render_direction_value(res).startswith("↓")


def test_kme_neg_direction_pos_is_down():
    state = _state({"modA": _spine({"genex": _member(-0.8)}, corr=0.5)})
    assert compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg()).direction == "down"


# --- indeterminate / seam-gating ------------------------------------------------------


def test_missing_outcome_direction_is_indeterminate():
    # module present, member present, but no eigengene→trait correlation ⇒ unsignable
    state = _state({"modA": _spine({"genex": _member(0.8)}, corr=None)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert res.direction == "indeterminate"
    assert res.computable is False and res.seam_present is True
    assert render_direction_value(res) == "indeterminate"


def test_seam_absent_is_indeterminate_and_omitted():
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), _state(module_spine=None), _cfg())
    assert res.direction == "indeterminate"
    assert res.seam_present is False and res.computable is False
    assert render_direction_value(res) is None  # line omitted entirely


def test_empty_module_spine_is_seam_absent():
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), _state({}), _cfg())
    assert res.seam_present is False
    assert render_direction_value(res) is None


def test_conflicting_signs_is_indeterminate_but_computable():
    # same entity weighted in two oppositely-signed modules ⇒ never a silently-picked sign
    state = _state({
        "modA": _spine({"genex": _member(0.8)}, corr=0.5),   # +
        "modB": _spine({"genex": _member(0.8)}, corr=-0.5),  # -
    })
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert res.direction == "indeterminate"
    assert res.computable is True  # signed records existed, they just disagree


# --- join-rate ------------------------------------------------------------------------


def test_join_resolves_records_on_matching_names():
    state = _state({"modA": _spine({"genex": _member(0.8)}, corr=0.5)})
    # name normalization: EntityResolution.raw_name "GeneX" → canonical "genex" matches member key
    assert compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg()).computable is True


def test_mismatched_name_space_yields_zero_computable_not_crash():
    # module_spine present but keyed on a different name ⇒ seam present, no signed record
    state = _state({"modA": _spine({"someotherprotein": _member(0.8)}, corr=0.5)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert res.seam_present is True
    assert res.computable is False
    assert res.direction == "indeterminate"


# --- small-n cap + confidence banding -------------------------------------------------


def test_confidence_bands_by_abs_kme():
    def _tier(kme):
        state = _state({"modA": _spine({"genex": _member(kme)}, corr=0.5)})
        return compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg()).confidence
    assert _tier(0.9) == "high"
    assert _tier(0.5) == "moderate"
    assert _tier(0.2) == "low"


def test_small_n_caps_high_confidence():
    state = _state({"modA": _spine({"genex": _member(0.95)}, corr=0.5)})  # |kME| high
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg(direction_small_n_cap=15), derivation_n=14)
    assert res.direction == "up"
    assert res.confidence == "moderate"  # n=14 < 15 ⇒ cannot reach top tier


def test_kim_floor_caps_high_confidence():
    state = _state({"modA": _spine({"genex": _member(0.95, kim=0.1)}, corr=0.5)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg(direction_kim_floor=0.5))
    assert res.confidence == "moderate"  # weakly-connected member cannot anchor top confidence


def test_malformed_record_does_not_raise():
    # member with a None kME and a member missing kme attr ⇒ helper still returns, no exception
    state = _state({"modA": _spine({"genex": _member(None), "genex2": SimpleNamespace()}, corr=0.5)})
    res = compute_tier3_direction(_hyp("H", ["NCBIGene:1"]), state, _cfg())
    assert isinstance(res, DirectionResult)
    assert res.direction == "indeterminate"


# --- tier3_direction_map --------------------------------------------------------------


def test_map_covers_only_tier3_keyed_by_title():
    state = _state({"modA": _spine({"genex": _member(0.8)}, corr=0.5)})
    state["hypotheses"] = [
        _hyp("T3", ["NCBIGene:1"], tier=3),
        _hyp("T1", ["NCBIGene:1"], tier=1),  # excluded
    ]
    dmap = tier3_direction_map(state, _cfg())
    assert set(dmap) == {"T3"}
    assert dmap["T3"].direction == "up"
