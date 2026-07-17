"""Axis B, Unit 2 — intramodular-centrality classifier + inverted routing in triage.

|kME|-based, per-module, relative top-k% hub detection (with a kIM veto for the low-n kME caveat),
joined to entities by normalized name, driving inverted routing (hub → cold_start). Flag-gated:
disabled / no ModuleSpine / no kME → the unchanged edge-count path (fallback identity, T1).

ModuleSpine is axis A's schema (not in this base); we read it duck-typed, so these fixtures use
lightweight stand-ins shaped like axis A's ModuleSpine/MemberWeight (attribute access) AND dicts.
"""

import json
import types

import pytest

from kestrel_backend.graph.nodes import triage
from kestrel_backend.graph.nodes.triage import compute_hub_members, _normalize_name
from kestrel_backend.graph.pipeline_config import PipelineConfig, TriageConfig
from kestrel_backend.graph.state import EntityResolution


# --- fixtures --------------------------------------------------------------------------

def _entity(curie, name, method="biomapper"):
    return EntityResolution(
        raw_name=name, curie=curie, resolved_name=name,
        category="biolink:Gene", confidence=0.9, method=method,
    )


def _member(name, kme, kim=None):
    return types.SimpleNamespace(name=name, kme=kme, kim=kim)


def _spine(modules, as_dict=False):
    """modules: {group: {name: (kme, kim)}} → {group: ModuleSpine-like}."""
    out = {}
    for group, mem in modules.items():
        members = {n: (_member(n, kme, kim) if not as_dict else {"name": n, "kme": kme, "kim": kim})
                   for n, (kme, kim) in mem.items()}
        out[group] = (types.SimpleNamespace(group=group, members=members) if not as_dict
                      else {"group": group, "members": members})
    return out


def _cfg(monkeypatch, **kw):
    cfg = PipelineConfig(triage=TriageConfig(**kw))
    monkeypatch.setattr(triage, "get_pipeline_config", lambda: cfg)
    monkeypatch.setattr(triage, "_RETRY_BACKOFF_S", 0.0)
    return cfg


def _kestrel_counts(counts):
    """counts: {curie: edge_count}. Returns an async call_kestrel_tool stub."""
    async def call(tool, args):
        c = counts.get(args["start_node_ids"], 0)
        return {"isError": False, "content": [{"text": json.dumps({"results_count": c})}]}
    return call


async def _run(monkeypatch, entities, counts, spine=None, **cfgkw):
    _cfg(monkeypatch, **cfgkw)
    monkeypatch.setattr(triage, "call_kestrel_tool", _kestrel_counts(counts))
    state = {"resolved_entities": entities}
    if spine is not None:
        state["module_spine"] = spine
    return await triage.run(state)


# --- T5 (pure): relative top-k% per module, no absolute |kME| leak ---------------------

def test_compute_hub_members_relative_cutoff_per_module():
    cfg = TriageConfig(intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=10.0)
    # module M1: 10 members (top-10% → ceil(1.0)=1 hub); M2: 20 members (→ ceil(2.0)=2 hubs)
    m1 = {f"a{i}": (0.1 * i, None) for i in range(1, 11)}   # a10 has highest |kME|
    m2 = {f"b{i}": (0.01 * i, None) for i in range(1, 21)}  # b20, b19 highest
    hubs = compute_hub_members(_spine({"M1": m1, "M2": m2}), cfg)
    crowned = {n for n, e in hubs.items() if e["crowned"]}
    assert crowned == {"a10", "b20", "b19"}


def test_compute_hub_members_all_low_kme_still_yields_top_k():
    # A module of uniformly LOW |kME| still yields its top-k% (relative, not absolute threshold).
    cfg = TriageConfig(intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=10.0)
    mod = {f"x{i}": (0.001 * i, None) for i in range(1, 11)}
    hubs = compute_hub_members(_spine({"M": mod}), cfg)
    assert {n for n, e in hubs.items() if e["crowned"]} == {"x10"}


def test_compute_hub_members_uses_absolute_kme_magnitude():
    # A large NEGATIVE kME is as central as a large positive one (|kME|).
    cfg = TriageConfig(intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=20.0)
    mod = {"neg": (-0.9, None), "pos": (0.2, None), "mid": (0.3, None), "low": (0.05, None)}
    hubs = compute_hub_members(_spine({"M": mod}), cfg)
    assert hubs[_normalize_name("neg")]["crowned"] is True  # |−0.9| is the top


def test_compute_hub_members_dict_shaped_spine():
    # Duck-typed: a dict-shaped ModuleSpine (post JSON round-trip) works identically.
    cfg = TriageConfig(intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=50.0)
    mod = {"h": (0.9, None), "l": (0.1, None)}
    hubs = compute_hub_members(_spine({"M": mod}, as_dict=True), cfg)
    assert hubs["h"]["crowned"] is True and hubs["l"]["crowned"] is False


# --- T1: fallback identity -------------------------------------------------------------

async def test_no_spine_is_edge_count_baseline(monkeypatch):
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta")]
    counts = {"G:1": 500, "G:2": 5}
    out = await _run(monkeypatch, ents, counts, spine=None, intramodular_centrality_enabled=True)
    assert out["well_characterized_curies"] == ["G:1"]
    assert out["sparse_curies"] == ["G:2"]
    assert all(s.is_intramodular_hub is False for s in out["novelty_scores"])


async def test_flag_off_ignores_spine(monkeypatch):
    ents = [_entity("G:1", "alpha")]
    spine = _spine({"M": {"alpha": (0.99, 50.0)}})
    out = await _run(monkeypatch, ents, {"G:1": 500}, spine=spine,
                     intramodular_centrality_enabled=False)
    assert out["well_characterized_curies"] == ["G:1"]  # unchanged; no inversion
    assert out["cold_start_curies"] == []
    assert out["novelty_scores"][0].is_intramodular_hub is False


# --- T2: hub promotion + inversion -----------------------------------------------------

async def test_hub_promoted_and_rerouted_to_cold_start(monkeypatch):
    # alpha: high |kME| (module hub) AND high edge_count (would be well_characterized → direct_kg).
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta"), _entity("G:3", "gamma")]
    counts = {"G:1": 800, "G:2": 300, "G:3": 250}
    spine = _spine({"M": {"alpha": (0.95, None), "beta": (0.2, None), "gamma": (0.15, None)}})
    out = await _run(monkeypatch, ents, counts, spine=spine,
                     intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=33.0)
    # alpha crowned (top 33% of 3 = ceil(0.99) = 1) → out of well_characterized, into cold_start
    assert "G:1" not in out["well_characterized_curies"]
    assert "G:1" in out["cold_start_curies"]
    hub = next(s for s in out["novelty_scores"] if s.curie == "G:1")
    assert hub.is_intramodular_hub is True
    assert hub.kme == 0.95
    assert hub.classification == "well_characterized"  # classification unchanged; hub rides boolean
    # beta/gamma not hubs → keep their edge-count routing
    assert "G:2" in out["well_characterized_curies"]  # 300 >= 200


# --- T3: kIM veto ----------------------------------------------------------------------

async def test_kim_veto_blocks_false_hub(monkeypatch):
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta")]
    counts = {"G:1": 800, "G:2": 800}
    # alpha: top |kME| but kIM below floor → vetoed; with floor met it would be crowned.
    spine = _spine({"M": {"alpha": (0.95, 1.0), "beta": (0.1, 99.0)}})
    out = await _run(monkeypatch, ents, counts, spine=spine,
                     intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=50.0,
                     intramodular_kim_floor=10.0)
    assert next(s for s in out["novelty_scores"] if s.curie == "G:1").is_intramodular_hub is False
    assert "G:1" in out["well_characterized_curies"]  # not rerouted


async def test_kim_floor_met_crowns(monkeypatch):
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta")]
    counts = {"G:1": 800, "G:2": 800}
    spine = _spine({"M": {"alpha": (0.95, 50.0), "beta": (0.1, 1.0)}})
    out = await _run(monkeypatch, ents, counts, spine=spine,
                     intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=50.0,
                     intramodular_kim_floor=10.0)
    assert next(s for s in out["novelty_scores"] if s.curie == "G:1").is_intramodular_hub is True
    assert "G:1" in out["cold_start_curies"]


# --- T4: kIM absent → |kME| alone ------------------------------------------------------

async def test_kim_absent_crowns_on_kme_alone(monkeypatch):
    ents = [_entity("G:1", "alpha")]
    spine = _spine({"M": {"alpha": (0.95, None)}})  # kim None
    out = await _run(monkeypatch, ents, {"G:1": 800}, spine=spine,
                     intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=100.0,
                     intramodular_kim_floor=10.0)  # floor set but kim None → veto skipped
    assert out["novelty_scores"][0].is_intramodular_hub is True


# --- T6: measurement-failure interaction preserved -------------------------------------

async def test_measurement_failure_not_rescued_but_hub_reroutes(monkeypatch):
    # beta's edge count can't be measured (isError forever) → moderate + marker (reliability fix).
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta")]

    async def call(tool, args):
        if args["start_node_ids"] == "G:2":
            return {"isError": True, "content": []}
        return {"isError": False, "content": [{"text": json.dumps({"results_count": 800})}]}

    _cfg(monkeypatch, intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=50.0)
    monkeypatch.setattr(triage, "call_kestrel_tool", call)
    # beta IS a valid top-k% kME hub (top of its own module) → should reroute to cold_start.
    spine = _spine({"M1": {"alpha": (0.9, None)}, "M2": {"beta": (0.8, None)}})
    out = await triage.run({"resolved_entities": ents, "module_spine": spine})

    beta = next(s for s in out["novelty_scores"] if s.curie == "G:2")
    assert beta.classification == "moderate"  # measurement failure NOT rescued to a real count
    assert beta.is_intramodular_hub is True   # but it IS a kME hub → cold_start
    assert "G:2" in out["cold_start_curies"]
    assert "G:2" not in out["moderate_curies"]
    assert any("edge-count failed" in e for e in out["errors"])  # visible marker preserved
