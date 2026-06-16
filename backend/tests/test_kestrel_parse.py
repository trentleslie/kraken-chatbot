"""Tests for kestrel_client.parse_kestrel_response.

The helper replaces the ``data.get("paths", data)`` silent-fallback parse that shipped to prod
in three nodes for ~3.5 months. It MUST fail loudly to an empty result on a missing/mis-shaped
``results`` and NEVER fall back to the raw dict (the bug that over-validated bridges in synthesis
and silently dropped them in integration).

Run with: uv run python -m pytest tests/test_kestrel_parse.py -v
"""

import json

from src.kestrel_backend.kestrel_client import parse_kestrel_response


def _envelope(inner: dict | list) -> dict:
    """Wrap an inner Kestrel response in the MCP tool-result envelope."""
    return {"content": [{"type": "text", "text": json.dumps(inner)}], "isError": False}


# The real multi_hop_query response shape (see .claude/skills/kestrel-api): results -> per-result
# paths (lists of CURIE strings) + end_node_id; names come from the top-level nodes dict.
REAL = _envelope({
    "results": [
        {"end_node_id": "MONDO:0005148", "paths": [["CHEBI:4167", "MONDO:0005148"]],
         "degree": 11051, "score": 0.81},
    ],
    "nodes": {"CHEBI:4167": {"name": "glucose"}, "MONDO:0005148": {"name": "type 2 diabetes"}},
    "edges": {},
})


def test_real_shape_parses_paths_and_names():
    r = parse_kestrel_response(REAL)
    assert r["n_paths"] == 1
    assert r["paths"][0]["curies"] == ["CHEBI:4167", "MONDO:0005148"]
    assert r["paths"][0]["names"] == ["glucose", "type 2 diabetes"]
    assert r["end_node_ids"] == ["MONDO:0005148"]


def test_empty_results_no_fallback_to_dict():
    r = parse_kestrel_response(_envelope({"results": [], "nodes": {}, "edges": {}}))
    assert r["n_paths"] == 0
    assert r["paths"] == []
    assert r["end_node_ids"] == []


def test_missing_results_key_fails_loud_not_dict():
    # No "results" key — must NOT return the raw dict (the silent-fallback bug).
    r = parse_kestrel_response(_envelope({"nodes": {"X": {"name": "x"}}}))
    assert r["n_paths"] == 0 and r["paths"] == []


def test_old_bug_shape_yields_nothing():
    # The shape Kestrel never returns ({"paths": [{"nodes": [...]}]}) must produce 0 paths,
    # not be mistaken for a real path set (regression guard for the historic bug).
    r = parse_kestrel_response(_envelope({"paths": [{"nodes": ["A", "B"], "predicates": ["x"]}]}))
    assert r["n_paths"] == 0


def test_malformed_inputs_never_raise():
    for bad in [
        _envelope({"results": "nope"}),          # results not a list
        _envelope(["x"]),                          # inner not a dict
        {"content": [{"text": "{not json"}]},      # bad JSON
        {"content": []},                           # empty content
        {},                                        # no content key
        {"content": [{}]},                         # content item missing text
    ]:
        r = parse_kestrel_response(bad)
        assert r["n_paths"] == 0 and r["paths"] == [] and r["end_node_ids"] == []


def test_short_path_excluded_but_end_node_kept():
    # A result with a terminal end_node_id but a too-short path: the path is dropped (len<2),
    # yet the end_node_id is still surfaced for reachable-node callers (pathway_enrichment).
    env = _envelope({
        "results": [{"end_node_id": "CHEBI:9", "paths": [["CHEBI:9"]]}],
        "nodes": {"CHEBI:9": {"name": "lonely"}},
    })
    r = parse_kestrel_response(env)
    assert r["n_paths"] == 0
    assert r["end_node_ids"] == ["CHEBI:9"]


def test_end_node_ids_deduped_order_preserving():
    env = _envelope({
        "results": [
            {"end_node_id": "B", "paths": [["A", "B"]]},
            {"end_node_id": "C", "paths": [["A", "C"]]},
            {"end_node_id": "B", "paths": [["A", "X", "B"]]},  # duplicate end-node
        ],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}, "C": {"name": "c"}, "X": {"name": "x"}},
    })
    r = parse_kestrel_response(env)
    assert r["end_node_ids"] == ["B", "C"]
    assert r["n_paths"] == 3


def test_missing_node_name_falls_back_to_curie():
    env = _envelope({
        "results": [{"end_node_id": "GO:1", "paths": [["CHEBI:1", "GO:1"]]}],
        "nodes": {"CHEBI:1": {"name": "named"}},  # GO:1 absent from nodes
    })
    r = parse_kestrel_response(env)
    assert r["paths"][0]["names"] == ["named", "GO:1"]


# ---------------------------------------------------------------------------
# Per-hop predicate derivation (U0 — resolves O3). The multi_hop response carries
# `edges` (edge-id -> compact tuple) + `edge_schema` (column order). Each path gains a
# `predicates` list (one entry per hop) with the predicate and its orientation vs the path.
# ---------------------------------------------------------------------------

# Canonical column order per the kestrel-api skill.
SCHEMA = ["subject", "predicate", "object", "qualifiers", "primary_knowledge_source",
          "supporting_sources", "aggregator_knowledge_source", "knowledge_level",
          "agent_type", "id"]


def _edge(subject, predicate, obj, eid=1):
    """Build a compact edge tuple in SCHEMA column order."""
    return [subject, predicate, obj, [], "infores:x", [], "infores:y", "knowledge_assertion",
            "manual_agent", eid]


def test_predicates_forward_two_hops():
    env = _envelope({
        "results": [{"end_node_id": "C", "paths": [["A", "B", "C"]], "edge_ids": [10, 11]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}, "C": {"name": "c"}},
        "edge_schema": SCHEMA,
        "edges": {"10": _edge("A", "biolink:affects", "B", 10),
                  "11": _edge("B", "biolink:causes", "C", 11)},
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [
        {"predicate": "biolink:affects", "forward": True},
        {"predicate": "biolink:causes", "forward": True},
    ]


def test_predicate_reverse_orientation_recovered():
    # Edge stored B->A for the A->B hop: predicate recorded, forward=False (the direction signal).
    env = _envelope({
        "results": [{"end_node_id": "B", "paths": [["A", "B"]], "edge_ids": [5]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}},
        "edge_schema": SCHEMA,
        "edges": {"5": _edge("B", "biolink:treats", "A", 5)},
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [{"predicate": "biolink:treats", "forward": False}]


def test_predicate_indices_read_from_edge_schema_not_hardcoded():
    # Reorder the schema so predicate is NOT at index 1; derivation must still be correct.
    schema = ["id", "object", "subject", "predicate"]
    env = _envelope({
        "results": [{"end_node_id": "B", "paths": [["A", "B"]], "edge_ids": [1]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}},
        "edge_schema": schema,
        "edges": {"1": [1, "B", "A", "biolink:related_to"]},  # id, object=B, subject=A, predicate
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [{"predicate": "biolink:related_to", "forward": True}]


def test_predicate_missing_edge_is_none_no_positional_shift():
    # Only the second hop has an edge; the first hop must be None (not shifted into position 0).
    env = _envelope({
        "results": [{"end_node_id": "C", "paths": [["A", "B", "C"]], "edge_ids": [2]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}, "C": {"name": "c"}},
        "edge_schema": SCHEMA,
        "edges": {"2": _edge("B", "biolink:causes", "C", 2)},
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [
        {"predicate": None, "forward": None},
        {"predicate": "biolink:causes", "forward": True},
    ]


def test_predicate_multiple_edges_deterministic_primary():
    # Two forward edges for the same pair -> pick the lexicographically smallest predicate.
    env = _envelope({
        "results": [{"end_node_id": "B", "paths": [["A", "B"]], "edge_ids": [1, 2]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}},
        "edge_schema": SCHEMA,
        "edges": {"1": _edge("A", "biolink:related_to", "B", 1),
                  "2": _edge("A", "biolink:affects", "B", 2)},
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [{"predicate": "biolink:affects", "forward": True}]  # 'affects' < 'related_to'


def test_predicates_empty_edges_all_none_length_preserved():
    # No edges/edge_schema (the existing REAL-style shape) -> predicates is all-None, len = hops.
    env = _envelope({
        "results": [{"end_node_id": "C", "paths": [["A", "B", "C"]]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}, "C": {"name": "c"}},
        "edges": {},
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [{"predicate": None, "forward": None}, {"predicate": None, "forward": None}]


def test_predicate_edge_ids_scope_falls_back_to_all_when_unresolved():
    # edge_ids reference ids absent from the edges dict -> fall back to scanning all edges.
    env = _envelope({
        "results": [{"end_node_id": "B", "paths": [["A", "B"]], "edge_ids": [999]}],
        "nodes": {"A": {"name": "a"}, "B": {"name": "b"}},
        "edge_schema": SCHEMA,
        "edges": {"7": _edge("A", "biolink:affects", "B", 7)},  # key 7, not 999
    })
    preds = parse_kestrel_response(env)["paths"][0]["predicates"]
    assert preds == [{"predicate": "biolink:affects", "forward": True}]
