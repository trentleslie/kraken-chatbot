"""L3 — bridge_grounding node: labeling, default-off no-op, filtering, error isolation,
graph wiring (both study-type paths), and synthesis rendering of the provenance label.
"""

from kestrel_backend.graph.builder import build_discovery_graph
from kestrel_backend.graph.nodes import bridge_grounding
from kestrel_backend.graph.nodes.synthesis import format_bridges, grounding_labels_from_state
from kestrel_backend.graph.pipeline_config import BridgeGroundingConfig, PipelineConfig
from kestrel_backend.graph.state import Bridge, BridgeGrounding


def _bridge(entities=("CHEBI:1", "HGNC:2", "MONDO:3"), tier=2, predicate_directions=(True, False)):
    return Bridge(
        path_description="metabolite → gene → disease",
        entities=list(entities),
        entity_names=[],
        predicates=["biolink:affects", "biolink:causes"],
        predicate_directions=list(predicate_directions),
        tier=tier,
        novelty="inferred",
        significance="why it matters",
    )


def _enable(monkeypatch, enabled=True, max_scored_bridges=20):
    cfg = PipelineConfig(bridge_grounding=BridgeGroundingConfig(
        enabled=enabled, max_scored_bridges=max_scored_bridges))
    monkeypatch.setattr(bridge_grounding, "get_pipeline_config", lambda: cfg)


def _stub_leg_tier(monkeypatch, tier="curated-causal", calls=None):
    async def fake(x, y):
        if calls is not None:
            calls.append((x, y))
        return tier
    monkeypatch.setattr(bridge_grounding, "leg_tier", fake)


# --- default-off no-op -----------------------------------------------------------------

async def test_disabled_is_noop(monkeypatch):
    # When disabled, the node must do nothing and call no Kestrel.
    _enable(monkeypatch, enabled=False)
    async def boom(*a, **k):
        raise AssertionError("leg_tier must not be called when disabled")
    monkeypatch.setattr(bridge_grounding, "leg_tier", boom)
    out = await bridge_grounding.run({"bridges": [_bridge()]})
    assert out == {}


def test_node_enabled_by_default():
    # L4 flip: the node now ships enabled=True (validation eval passed 2026-06-18).
    from kestrel_backend.graph.pipeline_config import BridgeGroundingConfig
    assert BridgeGroundingConfig().enabled is True


# --- happy path ------------------------------------------------------------------------

async def test_enabled_labels_curated_causal(monkeypatch):
    _enable(monkeypatch)
    calls = []
    _stub_leg_tier(monkeypatch, "curated-causal", calls)
    out = await bridge_grounding.run({"bridges": [_bridge()]})
    grounded = out["grounded_bridges"]
    assert len(grounded) == 1
    g = grounded[0].grounding
    assert g.label == "both legs curated-causal"
    assert [leg.evidence_tier for leg in g.legs] == ["curated-causal", "curated-causal"]
    assert [(leg.from_curie, leg.to_curie) for leg in g.legs] == [
        ("CHEBI:1", "HGNC:2"), ("HGNC:2", "MONDO:3")]
    assert calls == [("CHEBI:1", "HGNC:2"), ("HGNC:2", "MONDO:3")]  # 2 legs, start-only
    assert out["bridge_grounding_errors"] == []


async def test_weakest_leg_label(monkeypatch):
    _enable(monkeypatch)
    tiers = iter(["curated-causal", "text-mined"])
    async def fake(x, y):
        return next(tiers)
    monkeypatch.setattr(bridge_grounding, "leg_tier", fake)
    out = await bridge_grounding.run({"bridges": [_bridge()]})
    assert out["grounded_bridges"][0].grounding.label == "weakest leg text-mined"


# --- filtering -------------------------------------------------------------------------

async def test_skips_non_3node_and_subgraph_bridges(monkeypatch):
    _enable(monkeypatch)
    calls = []
    _stub_leg_tier(monkeypatch, "curated-causal", calls)
    bridges = [
        _bridge(entities=("A", "B"), predicate_directions=(True,)),          # 2-node
        _bridge(entities=("A", "B", "C", "D"), predicate_directions=(True, False, True)),  # 4-node
        _bridge(entities=("A", "B", "C"), predicate_directions=()),          # subgraph: pd == []
    ]
    out = await bridge_grounding.run({"bridges": bridges})
    assert out["grounded_bridges"] == []
    assert calls == []  # nothing scoreable → no Kestrel


async def test_max_scored_bridges_cap(monkeypatch):
    _enable(monkeypatch, max_scored_bridges=1)
    _stub_leg_tier(monkeypatch, "curated-causal")
    out = await bridge_grounding.run({"bridges": [_bridge(), _bridge(entities=("X:1", "Y:2", "Z:3"))]})
    assert len(out["grounded_bridges"]) == 1  # capped


# --- error isolation -------------------------------------------------------------------

async def test_per_bridge_error_isolation(monkeypatch):
    _enable(monkeypatch)
    good = _bridge(entities=("CHEBI:1", "HGNC:2", "MONDO:3"))
    bad = _bridge(entities=("X:1", "Y:2", "Z:3"))

    async def fake(x, y):
        if x == "X:1":
            raise RuntimeError("kestrel boom")
        return "curated-causal"
    monkeypatch.setattr(bridge_grounding, "leg_tier", fake)
    out = await bridge_grounding.run({"bridges": [good, bad]})
    assert len(out["grounded_bridges"]) == 2  # both still returned
    assert len(out["bridge_grounding_errors"]) == 1  # the bad one logged
    by_entities = {tuple(b.entities): b for b in out["grounded_bridges"]}
    assert by_entities[("CHEBI:1", "HGNC:2", "MONDO:3")].grounding.label == "both legs curated-causal"
    assert by_entities[("X:1", "Y:2", "Z:3")].grounding.label == "no KG edge"  # degraded, not crashed


# --- graph wiring (both study-type paths) ----------------------------------------------

def test_graph_routes_through_bridge_grounding_both_paths():
    edges = {(e.source, e.target) for e in build_discovery_graph().get_graph().edges}
    assert ("bridge_grounding", "synthesis") in edges          # single join point
    assert ("temporal", "bridge_grounding") in edges           # longitudinal path
    assert ("integration", "bridge_grounding") in edges        # non-longitudinal conditional path
    # Neither study type bypasses the node straight into synthesis:
    assert ("temporal", "synthesis") not in edges
    assert ("integration", "synthesis") not in edges


# --- synthesis rendering ---------------------------------------------------------------

def test_format_bridges_renders_provenance_label():
    b = _bridge()
    labels = {tuple(b.entities): "both legs curated-causal"}
    out = format_bridges([b], labels)
    assert "**Evidence provenance**: both legs curated-causal" in out


def test_format_bridges_without_labels_omits_provenance_line():
    out = format_bridges([_bridge()])
    assert "Evidence provenance" not in out


def test_format_bridges_tier2_includes_predicates():
    out = format_bridges([_bridge(tier=2)])
    assert "Predicates:" in out  # Tier 2 lists per-hop predicates


def test_format_bridges_tier3_omits_predicates_keeps_label():
    # Regression guard: the _render refactor must preserve the original behavior — Tier 3
    # (speculative) bridges deliberately OMIT the Predicates line (Greptile PR #77).
    b = _bridge(tier=3)
    out = format_bridges([b], {tuple(b.entities): "weakest leg text-mined"})
    assert "### Speculative Bridges (Tier 3)" in out
    assert "Predicates:" not in out  # omitted for Tier 3
    assert "**Evidence provenance**: weakest leg text-mined" in out  # label still rendered


def test_grounding_labels_from_state_builds_map():
    b = _bridge()
    gb = b.model_copy(update={"grounding": BridgeGrounding(legs=[], label="no KG edge")})
    m = grounding_labels_from_state({"grounded_bridges": [gb]})
    assert m[tuple(b.entities)] == "no KG edge"
