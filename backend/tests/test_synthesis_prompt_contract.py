"""Axis E Unit 3: prompt contract + direction-hint injection.

Plan: docs/plans/2026-07-16-001-feat-tier3-falsifier-direction-awareness-plan.md

The synthesis prompt must mandate one title-anchored prediction block per Tier-3 hypothesis, a
substance-guarded falsifier (measurable observable + threshold + null example), a once-per-section
calibration preamble, and the "Direction is stamped — narrate only" clause. The assembled context
must inject a title-anchored direction hint per Tier-3 hypothesis when axis A's seam is present, and
none when it is absent (seam-gating, no indeterminate-everywhere scaffolding).
"""

from types import SimpleNamespace

from kestrel_backend.graph.nodes.synthesis import (
    SYNTHESIS_PROMPT,
    assemble_synthesis_context,
    format_direction_hints,
)
from kestrel_backend.graph.pipeline_config import SynthesisConfig
from kestrel_backend.graph.state import EntityResolution, Hypothesis


def _member(kme, kim=None):
    return SimpleNamespace(kme=kme, kim=kim)


def _spine(members, corr=None):
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


# --- prompt contract ------------------------------------------------------------------


def test_prompt_mandates_per_hypothesis_blocks_and_falsifier():
    assert "one prediction block per Tier-3 hypothesis" in SYNTHESIS_PROMPT
    assert "**Falsifier:**" in SYNTHESIS_PROMPT
    assert "null-result example" in SYNTHESIS_PROMPT
    # substance guard: a validation restatement is NOT a falsifier
    assert "restatement of the validation step is NOT a falsifier" in SYNTHESIS_PROMPT


def test_prompt_marks_direction_as_stamped_not_invented():
    assert "stamped" in SYNTHESIS_PROMPT
    assert "do not invent" in SYNTHESIS_PROMPT


def test_prompt_instructs_bridge_specificity_use():
    # The Cross-Type Bridges context carries a "**Bridge specificity:**" signal (axis C); the prompt
    # must tell the LLM to use it — down-weight generic bridges, foreground specific ones — else the
    # signal reaches the context but never the report (observed at module scale).
    assert "Bridge specificity: generic" in SYNTHESIS_PROMPT
    assert "DOWN-WEIGHT" in SYNTHESIS_PROMPT
    assert "specific" in SYNTHESIS_PROMPT


def test_prompt_calibration_preamble_has_all_caveats():
    assert "~18%" in SYNTHESIS_PROMPT
    assert "n≈13–15" in SYNTHESIS_PROMPT  # Discovery-1 suggestive
    assert "956" in SYNTHESIS_PROMPT       # Discovery-2 null
    assert "Arivale" in SYNTHESIS_PROMPT   # wellness-cohort caveat
    assert "never a probability" in SYNTHESIS_PROMPT


# --- direction hint injection ---------------------------------------------------------


def _state_with_seam():
    return {
        "resolved_entities": [_entity("NCBIGene:1", "GeneX")],
        "hypotheses": [_hyp("MyPrediction", ["NCBIGene:1"])],
        "module_spine": {"modA": _spine({"genex": _member(0.8)}, corr=0.5)},
    }


def test_direction_hints_injected_when_seam_present():
    hints = format_direction_hints(_state_with_seam(), SynthesisConfig())
    assert "Direction Hints" in hints
    assert "MyPrediction" in hints
    assert "↑" in hints


def test_no_hints_when_seam_absent():
    state = _state_with_seam()
    del state["module_spine"]
    assert format_direction_hints(state, SynthesisConfig()) == ""


def test_assemble_context_includes_hints_when_seam_present():
    ctx = assemble_synthesis_context(_state_with_seam())
    assert "Tier-3 Direction Hints" in ctx
    assert "MyPrediction" in ctx


def test_assemble_context_omits_hints_when_seam_absent():
    state = _state_with_seam()
    del state["module_spine"]
    ctx = assemble_synthesis_context(state)
    assert "Direction Hints" not in ctx
