"""Axis C, Unit 3 — emit-only wiring of bridge specificity into the integration node.

The scoring pass populates the ``specificity_by_bridge`` state SIDE-MAP (keyed by
``tuple(bridge.entities)``) non-destructively: the ``bridges`` list, the frozen ``Bridge`` model,
tiers, and the ``grounding`` key are all unchanged. Per-bridge failure isolates to a missing map
entry and never fails the node. ``enabled=False`` -> empty map, zero Kestrel calls.

Run with: uv run python -m pytest tests/test_integration_bridge_specificity.py -v
"""

import pytest

from kestrel_backend.graph.nodes import bridge_specificity as bs
from kestrel_backend.graph.nodes import integration
from kestrel_backend.graph.pipeline_config import BridgeSpecificityConfig, PipelineConfig
from kestrel_backend.graph.state import Bridge, BridgeSpecificity, EntityResolution, Finding


def _multi_hop_bridge(entities, tier=2):
    return Bridge(
        path_description="metabolite → gene → disease (2 hops)",
        entities=list(entities),
        entity_names=list(entities),
        predicates=["biolink:affects", "biolink:causes"],
        predicate_directions=[True, False],
        tier=tier,
        novelty="known",
        significance="why it matters",
    )


def _subgraph_bridge(present, intermediates):
    # Mirrors _parse_subgraph_bridges: entities = present + intermediates (endpoints FIRST).
    entities = list(present) + list(intermediates)
    return Bridge(
        path_description=f"Connecting subgraph among {', '.join(present)}",
        entities=entities,
        entity_names=entities,
        predicates=["biolink:related_to"],
        predicate_directions=[],
        tier=2,
        novelty="known",
        significance="connecting structure",
    )


def _degree_map(monkeypatch, degrees: dict[str, int], calls=None):
    """Stub call_kestrel_tool so each CURIE resolves to a preview results_count == its degree."""
    async def fake(tool_name, params):
        if calls is not None:
            calls.append((tool_name, params))
        curie = params["start_node_ids"]
        d = degrees.get(curie, 0)
        return {"isError": False, "content": [{"text": f'{{"results_count": {d}}}'}]}
    monkeypatch.setattr(bs, "call_kestrel_tool", fake)


# =============================================================================
# score_bridges pass (unit level)
# =============================================================================

async def test_score_bridges_keys_by_entities_tuple(monkeypatch):
    _degree_map(monkeypatch, {"MID:hub": bs.GENERIC_CUTOFF * 3, "MID:spec": 5})
    b_generic = _multi_hop_bridge(["A:1", "MID:hub", "Z:9"])
    b_specific = _multi_hop_bridge(["A:2", "MID:spec", "Z:8"])
    pairs = [(b_generic, ["MID:hub"]), (b_specific, ["MID:spec"])]

    smap, errors = await bs.score_bridges(pairs, max_scored_bridges=20, concurrency=4)

    assert errors == []
    assert smap[("A:1", "MID:hub", "Z:9")].label == "generic"
    assert smap[("A:2", "MID:spec", "Z:8")].label == "specific"


async def test_score_bridges_error_isolation(monkeypatch):
    calls = []
    good = _multi_hop_bridge(["A:1", "MID:ok", "Z:9"])
    bad = _multi_hop_bridge(["A:2", "MID:bad", "Z:8"])

    async def fake(tool_name, params):
        calls.append(params["start_node_ids"])
        if params["start_node_ids"] == "MID:bad":
            raise RuntimeError("kestrel exploded")
        return {"isError": False, "content": [{"text": '{"results_count": 5}'}]}
    monkeypatch.setattr(bs, "call_kestrel_tool", fake)

    # The provider swallows fetch exceptions to None, so the bad bridge still scores (unknown),
    # never raising. Assert the good bridge is present and the node-level pass never raises.
    smap, errors = await bs.score_bridges(
        [(good, ["MID:ok"]), (bad, ["MID:bad"])], max_scored_bridges=20, concurrency=4)
    assert ("A:1", "MID:ok", "Z:9") in smap
    assert smap[("A:2", "MID:bad", "Z:8")].label == "unknown"  # degree fetch degraded to None


async def test_score_bridges_respects_cap(monkeypatch):
    _degree_map(monkeypatch, {"MID:1": 5, "MID:2": 5, "MID:3": 5})
    pairs = [
        (_multi_hop_bridge(["A:1", "MID:1", "Z:1"]), ["MID:1"]),
        (_multi_hop_bridge(["A:2", "MID:2", "Z:2"]), ["MID:2"]),
        (_multi_hop_bridge(["A:3", "MID:3", "Z:3"]), ["MID:3"]),
    ]
    smap, _ = await bs.score_bridges(pairs, max_scored_bridges=2, concurrency=4)
    assert len(smap) == 2  # third bridge beyond the cap has no entry


# =============================================================================
# integration.run wiring (emit-only invariant)
# =============================================================================

def _stub_integration(monkeypatch, *, enabled, multi_hop_bridges=(), subgraph_bridges=(), degrees=None):
    cfg = PipelineConfig(bridge_specificity=BridgeSpecificityConfig(enabled=enabled))
    monkeypatch.setattr(integration, "get_pipeline_config", lambda: cfg)

    async def fake_detect_bridges(resolved_entities, **_):
        return list(multi_hop_bridges), []
    async def fake_detect_subgraphs(resolved_entities, **_):
        return list(subgraph_bridges), []
    monkeypatch.setattr(integration, "detect_bridges_via_api", fake_detect_bridges)
    monkeypatch.setattr(integration, "detect_subgraphs_via_api", fake_detect_subgraphs)

    async def fake_query(*a, **k):
        return '{"gaps": []}', None
    monkeypatch.setattr(integration, "query_with_usage", fake_query)
    monkeypatch.setattr(integration, "HAS_SDK", True)

    if degrees is not None:
        _degree_map(monkeypatch, degrees)


def _state(resolved):
    return {
        "resolved_entities": resolved,
        # IntegrationInput requires at least one findings branch non-empty.
        "direct_findings": [Finding(entity="CHEBI:1", claim="seed", tier=2, source="direct_kg")],
        "disease_associations": [],
        "pathway_memberships": [],
        "inferred_associations": [],
        "biological_themes": [],
    }


def _resolved(curie, category):
    return EntityResolution(
        raw_name=curie, curie=curie, resolved_name=curie,
        category=category, confidence=1.0, method="exact",
    )


async def test_run_emit_only_bridges_unchanged(monkeypatch):
    bridges = [_multi_hop_bridge(["CHEBI:1", "HGNC:2", "MONDO:3"])]
    _stub_integration(
        monkeypatch, enabled=True, multi_hop_bridges=bridges,
        degrees={"HGNC:2": 5},
    )
    resolved = [_resolved("CHEBI:1", "biolink:ChemicalEntity"), _resolved("MONDO:3", "biolink:Disease")]
    out = await integration.run(_state(resolved))

    # bridges list identical (count + fields + no new Bridge field)
    assert len(out["bridges"]) == 1
    b = out["bridges"][0]
    assert b == bridges[0]  # frozen model, byte-identical
    assert not hasattr(b, "specificity")
    # the ONLY new output is the side-map
    smap = out["specificity_by_bridge"]
    assert set(smap) == {("CHEBI:1", "HGNC:2", "MONDO:3")}
    assert isinstance(smap[("CHEBI:1", "HGNC:2", "MONDO:3")], BridgeSpecificity)
    assert smap[("CHEBI:1", "HGNC:2", "MONDO:3")].label == "specific"


async def test_run_disabled_is_empty_and_no_kestrel(monkeypatch):
    bridges = [_multi_hop_bridge(["CHEBI:1", "HGNC:2", "MONDO:3"])]

    async def boom(*a, **k):
        raise AssertionError("no degree fetch when bridge_specificity disabled")
    monkeypatch.setattr(bs, "call_kestrel_tool", boom)
    _stub_integration(monkeypatch, enabled=False, multi_hop_bridges=bridges)
    resolved = [_resolved("CHEBI:1", "biolink:ChemicalEntity"), _resolved("MONDO:3", "biolink:Disease")]

    out = await integration.run(_state(resolved))
    assert out["specificity_by_bridge"] == {}
    assert len(out["bridges"]) == 1


async def test_run_subgraph_scaffold_is_non_input_curies_not_positional(monkeypatch):
    # Subgraph bridge: endpoints FIRST (present), then intermediates. A positional entities[1:-1]
    # slice would mis-score it. The pass must use the non-input-CURIE entities as the scaffold.
    present = ["CHEBI:1", "MONDO:3"]           # the resolved inputs (endpoints)
    intermediates = ["GO:hub", "GO:spec"]      # the connecting scaffold
    sub = _subgraph_bridge(present, intermediates)
    _stub_integration(
        monkeypatch, enabled=True, subgraph_bridges=[sub],
        degrees={"GO:hub": bs.GENERIC_CUTOFF * 4, "GO:spec": 5},
    )
    resolved = [_resolved("CHEBI:1", "biolink:ChemicalEntity"), _resolved("MONDO:3", "biolink:Disease")]

    out = await integration.run(_state(resolved))
    spec = out["specificity_by_bridge"][tuple(sub.entities)]
    # scaffold recovered = intermediates only; the hub condemns it as generic
    assert set(spec.intermediate_curies) == {"GO:hub", "GO:spec"}
    assert "CHEBI:1" not in spec.intermediate_curies  # endpoint excluded (not positional)
    assert spec.label == "generic"
    assert "GO:hub" in spec.generic_intermediates
