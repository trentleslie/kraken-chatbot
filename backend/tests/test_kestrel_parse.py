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
