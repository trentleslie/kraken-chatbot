"""Axis B, Unit 3 — measurement hooks in the triage_outcome line (R9-R12).

The hooks measure DIVERGENCE (edge-degree hub set vs |kME| hub set) and BLAST RADIUS (routing
shift), with an `expected_hub_n` so a silent name-join failure (actual << expected) is caught rather
than read as "a small hub set". Emitted only when the centrality pass runs; the flag-off line is
unchanged (fallback identity).
"""

import json
import logging
import types

from kestrel_backend.graph.nodes import triage
from kestrel_backend.graph.pipeline_config import PipelineConfig, TriageConfig
from kestrel_backend.graph.state import EntityResolution


def _entity(curie, name):
    return EntityResolution(raw_name=name, curie=curie, resolved_name=name,
                            category="biolink:Gene", confidence=0.9, method="biomapper")


def _spine(modules):
    out = {}
    for g, mem in modules.items():
        members = {n: types.SimpleNamespace(name=n, kme=kme, kim=kim) for n, (kme, kim) in mem.items()}
        out[g] = types.SimpleNamespace(group=g, members=members)
    return out


def _cfg(monkeypatch, **kw):
    cfg = PipelineConfig(triage=TriageConfig(**kw))
    monkeypatch.setattr(triage, "get_pipeline_config", lambda: cfg)
    monkeypatch.setattr(triage, "_RETRY_BACKOFF_S", 0.0)


def _kestrel(counts):
    async def call(tool, args):
        return {"isError": False, "content": [{"text": json.dumps({"results_count": counts[args["start_node_ids"]]})}]}
    return call


def _outcome_from_logs(caplog):
    for r in caplog.records:
        msg = r.getMessage()
        if msg.startswith("triage_outcome "):
            return json.loads(msg[len("triage_outcome "):])
    raise AssertionError("no triage_outcome line emitted")


async def test_hooks_emitted_when_active_disjoint_sets(monkeypatch, caplog):
    # alpha: edge-degree hub (500 >= 200) but LOW kME → not a kME hub.
    # beta:  moderate edge count (50, NOT an edge-degree hub) but HIGH kME → kME hub.
    # → hub sets disjoint (Jaccard 0); beta's reroute is a direct→cold shift (moderate would go direct_kg).
    ents = [_entity("G:1", "alpha"), _entity("G:2", "beta")]
    _cfg(monkeypatch, intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=50.0)
    monkeypatch.setattr(triage, "call_kestrel_tool", _kestrel({"G:1": 500, "G:2": 50}))
    spine = _spine({"M": {"alpha": (0.1, None), "beta": (0.9, None)}})

    with caplog.at_level(logging.INFO, logger="kestrel_backend.graph.nodes.triage"):
        out = await triage.run({"resolved_entities": ents, "module_spine": spine})

    o = _outcome_from_logs(caplog)
    assert o["edge_degree_hub_n"] == 1        # {G:1}
    assert o["kme_hub_n"] == 1                # {G:2}
    assert o["hub_set_jaccard"] == 0.0        # disjoint
    assert o["hub_set_only_edge_degree_n"] == 1
    assert o["hub_set_only_kme_n"] == 1
    assert o["routing_shift_direct_to_cold"] == 1  # beta (moderate) → cold_start
    assert o["routing_shift_cold_to_direct"] == 0  # pure inversion
    assert o["expected_hub_n"] == 1                # ceil(50% * 2)
    # and the routing actually happened
    assert "G:2" in out["cold_start_curies"] and "G:2" not in out["moderate_curies"]


async def test_hooks_absent_when_flag_off(monkeypatch, caplog):
    ents = [_entity("G:1", "alpha")]
    _cfg(monkeypatch, intramodular_centrality_enabled=False)
    monkeypatch.setattr(triage, "call_kestrel_tool", _kestrel({"G:1": 500}))
    spine = _spine({"M": {"alpha": (0.9, None)}})
    with caplog.at_level(logging.INFO, logger="kestrel_backend.graph.nodes.triage"):
        await triage.run({"resolved_entities": ents, "module_spine": spine})
    o = _outcome_from_logs(caplog)
    assert "hub_set_jaccard" not in o          # flag-off line unchanged (fallback identity)
    assert "routing_shift_direct_to_cold" not in o
    assert o["well_characterized"] == 1        # base fields still present


async def test_expected_vs_actual_mismatch_visible_on_join_failure(monkeypatch, caplog):
    # ModuleSpine has a hub, but NO entity name matches it (silent mis-join): kme_hub_n=0 while
    # expected_hub_n=1 — the mismatch is the check that catches the join failure.
    ents = [_entity("G:1", "alpha")]
    _cfg(monkeypatch, intramodular_centrality_enabled=True, intramodular_hub_top_k_pct=100.0)
    monkeypatch.setattr(triage, "call_kestrel_tool", _kestrel({"G:1": 500}))
    spine = _spine({"M": {"UNMATCHED_NAME": (0.9, None)}})
    with caplog.at_level(logging.INFO, logger="kestrel_backend.graph.nodes.triage"):
        await triage.run({"resolved_entities": ents, "module_spine": spine})
    o = _outcome_from_logs(caplog)
    assert o["expected_hub_n"] == 1
    assert o["kme_hub_n"] == 0  # join produced nothing → visible mismatch
