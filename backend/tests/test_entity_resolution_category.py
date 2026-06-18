"""Tier 1 wrong-namespace fix (Unit 3): category-constrained resolve_via_api + call-site threading.

Design (verified live 2026-06-17 — Kestrel `category` matching is list-membership and genuinely filters):
- category=None      -> byte-identical to today (limit=1, no category).
- in-category hit above tier1_min_score -> accept via existing bands.
- in-category hit at/below tier1_min_score, or empty -> return None (routes to Tier 2 LLM).
- isError on the constrained call -> fall back to the unconstrained call at tier1_fallback_confidence.
"""

import json

from kestrel_backend.graph.nodes import entity_resolution
from kestrel_backend.graph.nodes.entity_resolution import resolve_via_api, run
from kestrel_backend.graph.pipeline_config import get_pipeline_config


def _hs(search_text, rows, is_error=False):
    """Build the real hybrid_search MCP envelope: {content:[{text: {search_text: [rows]}}]}."""
    body = {search_text: rows}
    return {"content": [{"type": "text", "text": json.dumps(body)}], "isError": is_error}


MONDO_CML = {
    "id": "MONDO:0011996", "name": "chronic myelogenous leukemia, BCR-ABL1 positive",
    "score": 2.495, "categories": ["biolink:DiseaseOrPhenotypicFeature", "biolink:Disease"],
}
KEGG_CML = {
    "id": "KEGG:05220", "name": "Chronic myeloid leukemia",
    "score": 4.86, "categories": ["biolink:Pathway"],
}


class TestResolveViaApiCategory:
    async def test_constrained_hit_above_threshold_resolves_in_category(self, monkeypatch):
        calls = []
        async def fake(tool, args):
            calls.append(args)
            assert args.get("category") == "biolink:Disease"  # category threaded through
            return _hs(args["search_text"], [MONDO_CML])
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("chronic myeloid leukemia", category="biolink:Disease")
        assert res is not None
        assert res.curie == "MONDO:0011996"
        assert len(calls) == 1  # single call, no fallback

    async def test_low_score_in_category_routes_to_tier2_not_wrong_node(self, monkeypatch):
        # The regression-critical case: a correct MONDO node scoring BELOW tier1_min_score must
        # return None (-> Tier 2 LLM), NOT fall back to and accept the high-score KEGG pathway.
        calls = []
        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [dict(MONDO_CML, score=0.55)])
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("chronic myeloid leukemia", category="biolink:Disease")
        assert res is None                 # routed to Tier 2
        assert len(calls) == 1             # NO unconstrained fallback attempted
        assert calls[0].get("category") == "biolink:Disease"

    async def test_no_hint_is_byte_identical_to_today(self, monkeypatch):
        calls = []
        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [MONDO_CML])
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("chronic myeloid leukemia", category=None)
        assert res is not None and res.curie == "MONDO:0011996"
        assert len(calls) == 1
        assert "category" not in calls[0]  # unconstrained — no category sent

    async def test_iserror_with_category_falls_back_to_unconstrained(self, monkeypatch):
        fb = get_pipeline_config().entity_resolution.tier1_fallback_confidence
        async def fake(tool, args):
            if args.get("category"):
                return _hs(args["search_text"], [], is_error=True)   # constrained errors
            return _hs(args["search_text"], [KEGG_CML])              # unconstrained succeeds
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("chronic myeloid leukemia", category="biolink:Disease")
        assert res is not None
        assert res.curie == "KEGG:05220"
        assert res.confidence == fb
        assert "fallback" in res.method

    async def test_constrained_empty_routes_to_tier2(self, monkeypatch):
        calls = []
        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [])   # no in-category rows
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("mystery disease", category="biolink:Disease")
        assert res is None
        assert len(calls) == 1  # empty -> None directly, no unconstrained fallback

    async def test_iserror_no_category_returns_none(self, monkeypatch):
        # Today's behavior unchanged: isError on the unconstrained path -> None, no recursion.
        calls = []
        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [], is_error=True)
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("whatever", category=None)
        assert res is None
        assert len(calls) == 1


class TestCallSiteThreading:
    async def test_primary_call_site_threads_category_from_hint(self, monkeypatch):
        seen = []
        async def fake(tool, args):
            seen.append(args)
            return _hs(args["search_text"], [MONDO_CML])
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        state = {
            "raw_entities": ["chronic myeloid leukemia"],
            "entity_type_hints": {"chronic myeloid leukemia": "disease"},
            "entity_aliases": {},
        }
        out = await run(state)
        assert out["resolved_entities"][0].curie == "MONDO:0011996"
        assert any(a.get("category") == "biolink:Disease" for a in seen)

    async def test_alias_call_site_threads_parent_category(self, monkeypatch):
        # Primary "mystery disease" returns empty (-> None -> Tier 1.5); its alias resolves.
        # The alias call must carry the PARENT entity's category (keyed on the parent name).
        seen = []
        async def fake(tool, args):
            seen.append(args)
            if args["search_text"] == "CML":
                return _hs("CML", [MONDO_CML])
            return _hs(args["search_text"], [])  # primary empty
        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        state = {
            "raw_entities": ["mystery disease"],
            "entity_type_hints": {"mystery disease": "disease"},
            "entity_aliases": {"mystery disease": ["CML"]},
        }
        out = await run(state)
        er = out["resolved_entities"][0]
        assert er.curie == "MONDO:0011996"
        assert er.method.startswith("alias:")
        alias_call = next(a for a in seen if a["search_text"] == "CML")
        assert alias_call.get("category") == "biolink:Disease"
