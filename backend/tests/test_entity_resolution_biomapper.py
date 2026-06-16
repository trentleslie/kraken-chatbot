"""Tests for the Biomapper pre-resolver: reconciliation helper (Unit 3) + node wiring (Unit 4)."""

import json

import pytest
from biomapper import BioMapperAuthError

from kestrel_backend.graph.nodes import entity_resolution
from kestrel_backend.graph.nodes.entity_resolution import reconcile_to_kestrel, run
from kestrel_backend.graph.pipeline_config import (
    BiomapperConfig,
    EntityResolutionConfig,
    PipelineConfig,
)


# --------------------------- get_nodes envelope helpers ---------------------------

def _gn_envelope(curie: str, node: dict | None, as_list: bool = False) -> dict:
    """Build a Kestrel get_nodes MCP envelope.

    Live shape (confirmed 2026-06-16): present -> {curie: {node}} (node dict directly);
    `as_list=True` exercises the alternate {curie: [node]} wrapping; absent -> {curie: []}.
    """
    if node is None:
        body = {curie: []}
    else:
        body = {curie: [node] if as_list else node}
    return {"content": [{"type": "text", "text": json.dumps(body)}], "isError": False}


def _human_gene_node(curie="NCBIGene:7132"):
    return {"id": curie, "categories": ["biolink:Gene", "biolink:Protein"],
            "equivalent_ids": [curie, "HGNC:11916", "UniProtKB:P19438"]}


def _ortholog_node(curie="NCBIGene:397020"):
    return {"id": curie, "categories": ["biolink:Gene", "biolink:Protein"],
            "equivalent_ids": [curie, "RGD:14161077", "UniProtKB:P50555"]}  # no HGNC


def _biomapper_result(curie="NCBIGene:7132", tier="high", xrefs=None):
    return {"curie": curie, "tier": tier, "confidence": 2.5, "resolved_name": "TNFRSF1A",
            "category": "biolink:Gene", "xrefs": xrefs or {"HGNC": ["11916"], "NCBIGene": ["7132"]}}


# =============================== Unit 3: reconcile_to_kestrel ===============================

class TestReconcile:
    async def test_primary_confirms_with_human_marker(self, monkeypatch):
        async def fake_get_nodes(tool, args):
            return _gn_envelope(args["curies"], _human_gene_node(args["curies"]))
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake_get_nodes)
        out = await reconcile_to_kestrel(_biomapper_result(), "gene")
        assert out == ("NCBIGene:7132", "biolink:Gene")

    async def test_gene_without_hgnc_rejected(self, monkeypatch):
        # Ortholog confirms in Kestrel but lacks HGNC → defense gate rejects; no other candidate → None.
        async def fake(tool, args):
            c = args["curies"]
            # Only the ortholog primary exists; it has no HGNC equivalent.
            return _gn_envelope(c, _ortholog_node(c) if c == "NCBIGene:397020" else None)
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        # Build inline so xrefs is genuinely empty (the helper's `xrefs or {...}` would refill it).
        result = {"curie": "NCBIGene:397020", "tier": "high", "resolved_name": "X", "xrefs": {}}
        assert await reconcile_to_kestrel(result, "gene") is None

    async def test_metabolite_skips_hgnc_gate(self, monkeypatch):
        # A metabolite node with no HGNC is accepted (gate is gene/protein only).
        node = {"id": "CHEBI:16946", "categories": ["biolink:ChemicalEntity"], "equivalent_ids": ["HMDB:H1"]}
        async def fake(tool, args):
            return _gn_envelope(args["curies"], node)
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        out = await reconcile_to_kestrel(
            {"curie": "CHEBI:16946", "tier": "high", "xrefs": {}}, "metabolite")
        assert out == ("CHEBI:16946", "biolink:ChemicalEntity")

    async def test_none_confirm_returns_none(self, monkeypatch):
        async def fake(tool, args):
            return _gn_envelope(args["curies"], None)  # empty list = absent
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        assert await reconcile_to_kestrel(_biomapper_result(xrefs={}), "gene") is None

    async def test_primary_absent_falls_to_xref(self, monkeypatch):
        # primary_curie absent; the HGNC xref candidate confirms (human).
        async def fake(tool, args):
            c = args["curies"]
            if c == "NCBIGene:7132":
                return _gn_envelope(c, None)
            if c == "HGNC:11916":
                return _gn_envelope(c, _human_gene_node("NCBIGene:7132"))
            return _gn_envelope(c, None)
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        out = await reconcile_to_kestrel(_biomapper_result(), "gene")
        assert out == ("NCBIGene:7132", "biolink:Gene")  # node.id, not the queried HGNC curie

    async def test_accepts_list_wrapped_node_envelope(self, monkeypatch):
        # Some Kestrel versions wrap the node as {curie: [node]}; both shapes must parse.
        async def fake(tool, args):
            return _gn_envelope(args["curies"], _human_gene_node(args["curies"]), as_list=True)
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        out = await reconcile_to_kestrel(_biomapper_result(), "gene")
        assert out == ("NCBIGene:7132", "biolink:Gene")

    async def test_transport_error_tries_next_candidate(self, monkeypatch):
        calls = {"n": 0}
        async def fake(tool, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("kestrel down")
            return _gn_envelope(args["curies"], _human_gene_node(args["curies"]))
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        out = await reconcile_to_kestrel(_biomapper_result(), "gene")
        assert out is not None and calls["n"] >= 2


# =============================== Unit 4: run() pre-pass wiring ===============================

def _enable_biomapper(monkeypatch):
    cfg = PipelineConfig(entity_resolution=EntityResolutionConfig(biomapper=BiomapperConfig(enabled=True)))
    monkeypatch.setattr(entity_resolution, "get_pipeline_config", lambda: cfg)


class TestRunPrepass:
    async def test_flag_off_does_not_call_biomapper(self, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("biomapper must not be called when flag is off")
        monkeypatch.setattr(entity_resolution, "biomapper_resolve", boom)
        # Tier 1 resolves the entity so it doesn't fall to SDK Tier 2.
        async def fake_kestrel(tool, args):
            return {"content": [{"text": json.dumps(
                {args.get("search_text", ""): [{"id": "NCBIGene:7132", "name": "TNFRSF1A",
                 "score": 2.0, "categories": ["biolink:Gene"]}]})}], "isError": False}
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake_kestrel)
        state = {"raw_entities": ["TNFRSF1A"], "entity_type_hints": {"TNFRSF1A": "gene"}, "entity_aliases": {}}
        out = await run(state)
        assert len(out["resolved_entities"]) == 1
        assert out["resolved_entities"][0].method != "biomapper"

    async def test_flag_on_hit_uses_biomapper_and_skips_tier1(self, monkeypatch):
        _enable_biomapper(monkeypatch)
        async def fake_bm(name, hint, base_url=None):
            return _biomapper_result()
        monkeypatch.setattr(entity_resolution, "biomapper_resolve", fake_bm)
        seen_tools = []
        async def fake_kestrel(tool, args):
            seen_tools.append(tool)
            assert tool == "get_nodes", f"hybrid_search should not run for a biomapper-confirmed entity (got {tool})"
            return _gn_envelope(args["curies"], _human_gene_node(args["curies"]))
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake_kestrel)
        state = {"raw_entities": ["TNFRSF1A"], "entity_type_hints": {"TNFRSF1A": "gene"}, "entity_aliases": {}}
        out = await run(state)
        er = out["resolved_entities"][0]
        assert er.method == "biomapper"
        assert er.curie == "NCBIGene:7132"
        assert "hybrid_search" not in seen_tools

    async def test_flag_on_no_hint_skips_biomapper(self, monkeypatch):
        _enable_biomapper(monkeypatch)
        async def boom(*a, **k):
            raise AssertionError("no hint → biomapper must be skipped")
        monkeypatch.setattr(entity_resolution, "biomapper_resolve", boom)
        async def fake_kestrel(tool, args):
            return {"content": [{"text": json.dumps(
                {args.get("search_text", ""): [{"id": "NCBIGene:7132", "name": "x",
                 "score": 2.0, "categories": ["biolink:Gene"]}]})}], "isError": False}
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake_kestrel)
        state = {"raw_entities": ["TNFRSF1A"], "entity_type_hints": {}, "entity_aliases": {}}
        out = await run(state)
        assert len(out["resolved_entities"]) == 1

    async def test_flag_on_auth_error_propagates(self, monkeypatch):
        _enable_biomapper(monkeypatch)
        async def fake_bm(name, hint, base_url=None):
            raise BioMapperAuthError("bad key")
        monkeypatch.setattr(entity_resolution, "biomapper_resolve", fake_bm)
        state = {"raw_entities": ["TNFRSF1A"], "entity_type_hints": {"TNFRSF1A": "gene"}, "entity_aliases": {}}
        with pytest.raises(BioMapperAuthError):
            await run(state)
