"""Axis E Unit 5: tier3_prediction_stats telemetry (deterministic + compliance) + fallback parity.

Plan: docs/plans/2026-07-16-001-feat-tier3-falsifier-direction-awareness-plan.md

The block reports a DETERMINISTIC coverage metric (`direction_computable_pct`, from state and
independent of report text) explicitly separated from COMPLIANCE metrics (`*_rendered_pct`, regex
over the report), so the compliance markers are never mistaken for falsifiability/coverage of the
property. The fallback path renders Direction from state and emits stats with `falsifier_rendered=0`.
"""

from types import SimpleNamespace

import pytest

from kestrel_backend.graph.nodes import synthesis
from kestrel_backend.graph.nodes.synthesis import (
    _compute_tier3_stats,
    _count_wellformed_falsifiers,
)
from kestrel_backend.graph.pipeline_config import SynthesisConfig
from kestrel_backend.graph.state import EntityResolution, Finding, Hypothesis


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


def _state_2of3():
    """3 Tier-3 hypotheses; module_spine gives signed records for exactly 2 of them."""
    return {
        "resolved_entities": [
            _entity("NCBIGene:1", "GeneX"),
            _entity("NCBIGene:2", "GeneY"),
            _entity("NCBIGene:3", "GeneZ"),
        ],
        "hypotheses": [
            _hyp("H1", ["NCBIGene:1"]),
            _hyp("H2", ["NCBIGene:2"]),
            _hyp("H3", ["NCBIGene:3"]),  # not in module_spine ⇒ not computable
        ],
        "module_spine": {
            "modA": _spine({"genex": _member(0.8), "geney": _member(0.6)}, corr=0.5),
        },
        # SynthesisInput requires a non-empty findings channel; irrelevant to the metrics under test.
        "direct_findings": [
            Finding(entity="GeneX", claim="c", tier=1, predicate=None, source="direct_kg",
                    pmids=[], confidence="high", logic_chain=None),
        ],
    }


# --- deterministic coverage metric ----------------------------------------------------


def test_direction_computable_pct_from_state_ignores_report_text():
    cfg = SynthesisConfig()
    state = _state_2of3()
    a = _compute_tier3_stats(state, report="garbage with no markers", cfg=cfg)
    b = _compute_tier3_stats(state, report="#### H1\n**Direction:** ↑ (confidence: high)\n", cfg=cfg)
    assert a["total_tier3"] == 3
    assert a["direction_computable"] == 2
    assert a["direction_computable_pct"] == pytest.approx(66.7, abs=0.1)
    # deterministic metric is identical regardless of report content
    assert a["direction_computable_pct"] == b["direction_computable_pct"]
    assert a["metric_kind"]["direction_computable_pct"] == "deterministic"
    assert a["metric_kind"]["falsifier_rendered_pct"] == "compliance"


def test_compliance_metrics_count_report_markers():
    cfg = SynthesisConfig()
    report = (
        "#### H1\n**Direction:** ↑ (confidence: high)\n"
        "**Falsifier:** metabolite X does not decrease by >20% in an independent cohort\n"
        "#### H2\n**Direction:** ↓ (confidence: moderate)\n"
        "**Falsifier:** module eigengene correlation with trait is ≥ 0 in validation set\n"
        "#### H3\n"
        "**Falsifier:** taxon abundance shows no change (null) across all timepoints\n"
    )
    stats = _compute_tier3_stats(_state_2of3(), report, cfg)
    assert stats["direction_rendered"] == 2
    assert stats["falsifier_rendered"] == 3
    assert stats["direction_rendered_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["falsifier_rendered_pct"] == pytest.approx(100.0, abs=0.1)


def test_indeterminate_tracked_separately():
    # entity present in module but no module direction ⇒ seam present, indeterminate, not computable
    state = {
        "resolved_entities": [_entity("NCBIGene:1", "GeneX")],
        "hypotheses": [_hyp("H1", ["NCBIGene:1"])],
        "module_spine": {"modA": _spine({"genex": _member(0.8)}, corr=None)},
    }
    stats = _compute_tier3_stats(state, report="", cfg=SynthesisConfig())
    assert stats["seam_present"] is True
    assert stats["direction_computable"] == 0
    assert stats["indeterminate_direction"] == 1


def test_zero_tier3_no_division_by_zero():
    stats = _compute_tier3_stats({"hypotheses": []}, report="", cfg=SynthesisConfig())
    assert stats["total_tier3"] == 0
    assert stats["direction_computable_pct"] == 0.0
    assert stats["falsifier_rendered_pct"] == 0.0


# --- well-formed falsifier heuristic --------------------------------------------------


def test_validation_restatement_not_counted_wellformed():
    # a vacuous restatement with no measurable observable / threshold / direction word
    report = "**Falsifier:** Run the same validation experiment and see what happens.\n"
    assert _count_wellformed_falsifiers(report) == 0


def test_measurable_falsifier_counted():
    report = "**Falsifier:** if metabolite X does not decrease by >20% in cases, reject the prediction\n"
    assert _count_wellformed_falsifiers(report) == 1


# --- run() emits the field on both paths ----------------------------------------------


@pytest.mark.asyncio
async def test_run_emits_tier3_stats_on_fallback_path(monkeypatch):
    monkeypatch.setattr(synthesis, "HAS_SDK", False)  # deterministic fallback, no LLM
    result = await synthesis.run(_state_2of3())
    assert "tier3_prediction_stats" in result
    stats = result["tier3_prediction_stats"]
    assert stats["total_tier3"] == 3
    assert stats["direction_computable"] == 2
    # R6: fallback renders Direction from state, but authors no falsifiers
    assert "Tier-3 Prediction Directions" in result["synthesis_report"]
    assert stats["direction_rendered"] >= 2
    assert stats["falsifier_rendered"] == 0


def test_field_is_not_a_concat_field():
    from kestrel_backend.main import _get_concat_fields
    assert "tier3_prediction_stats" not in _get_concat_fields()  # plain dict → last-write-wins
