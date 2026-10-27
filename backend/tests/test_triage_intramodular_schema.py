"""Axis B, Unit 1 — NoveltyScore hub fields + TriageConfig centrality knobs (schema/contract).

The hub verdict rides a boolean field (NOT a new classification Literal — a 5th value would KeyError
in synthesis.py's by_class dict), plus kme/kim for downstream visibility. Config knobs are default-off.
"""

from kestrel_backend.graph.pipeline_config import PipelineConfig, TriageConfig
from kestrel_backend.graph.state import NoveltyScore


def test_novelty_score_defaults_are_backward_compatible():
    # Existing construction (no hub fields) must still work; hub-ness defaults off.
    s = NoveltyScore(curie="NCBIGene:1", raw_name="A", edge_count=500, classification="well_characterized")
    assert s.is_intramodular_hub is False
    assert s.kme is None
    assert s.kim is None


def test_novelty_score_carries_hub_fields():
    s = NoveltyScore(
        curie="NCBIGene:1", raw_name="A", edge_count=500, classification="well_characterized",
        is_intramodular_hub=True, kme=-0.82, kim=14.3,
    )
    assert s.is_intramodular_hub is True
    assert s.kme == -0.82
    assert s.kim == 14.3


def test_classification_literal_unchanged_no_fifth_value():
    # A 5th classification value would break synthesis.py by_class — hub rides the boolean instead.
    import pydantic
    try:
        NoveltyScore(curie="X:1", raw_name="A", edge_count=1, classification="intramodular_hub")
    except pydantic.ValidationError:
        return
    raise AssertionError("classification must stay a 4-value Literal (no 'intramodular_hub')")


def test_kim_floor_ge_zero_on_novelty_score():
    import pydantic
    try:
        NoveltyScore(curie="X:1", raw_name="A", edge_count=1, classification="sparse", kim=-1.0)
    except pydantic.ValidationError:
        return
    raise AssertionError("kim must carry a ge=0 floor (raw connectivity is non-negative)")


def test_novelty_score_frozen():
    s = NoveltyScore(curie="X:1", raw_name="A", edge_count=1, classification="sparse")
    try:
        s.is_intramodular_hub = True  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("NoveltyScore must stay frozen")


def test_triage_config_centrality_knobs_default_off():
    cfg = TriageConfig()
    assert cfg.intramodular_centrality_enabled is False
    assert cfg.intramodular_hub_top_k_pct == 10.0
    assert cfg.intramodular_kim_floor is None


def test_triage_config_top_k_pct_bounds():
    import pydantic
    for bad in (0.0, -5.0, 150.0):
        try:
            TriageConfig(intramodular_hub_top_k_pct=bad)
        except pydantic.ValidationError:
            continue
        raise AssertionError(f"top_k_pct={bad} should be rejected (must be in (0, 100])")


def test_pipeline_config_exposes_triage_knobs():
    assert PipelineConfig().triage.intramodular_centrality_enabled is False
