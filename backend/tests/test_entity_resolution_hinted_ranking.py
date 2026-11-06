"""#114: hinted Tier-1 ranking — human taxon for gene/protein, small-molecule re-rank, hinted category.

Fixture rows are trimmed copies of live Kestrel responses (kestrel.krakenkg.com, 2026-11-06).
"""

import json

from kestrel_backend.graph.nodes import entity_resolution
from kestrel_backend.graph.nodes.entity_resolution import (
    _rerank_small_molecule,
    resolve_via_api,
)

SM = "biolink:SmallMolecule"


def _hs(search_text, rows, is_error=False):
    return {"content": [{"type": "text", "text": json.dumps({search_text: rows})}], "isError": is_error}


GLU_SALT = {"id": "CHEBI:13107", "name": "L-2-Aminopentanedioic Acid Sodium Salt", "score": 3.11,
            "categories": [SM], "neighbors_count": 1, "synonyms": ["Glutamate"]}
GLU_2MINUS = {"id": "CHEBI:29987", "name": "glutamate(2-)", "score": 3.05,
              "categories": [SM], "neighbors_count": 63, "synonyms": []}
D_GLU = {"id": "RM:0033230", "name": "D-Glutamic acid", "score": 3.03,
         "categories": [SM], "neighbors_count": 159, "synonyms": []}
GLU_ACID = {"id": "RM:0135893", "name": "Glutamic acid", "score": 0.58,
            "categories": ["biolink:Drug", SM], "neighbors_count": 2658, "synonyms": ["L-Glutamic acid"]}

MAN_CONJ = {"id": "HMDB:HMDB0062473", "name": "Glycoprotein-phospho-D-mannose", "score": 5.1,
            "categories": [SM], "neighbors_count": 656, "synonyms": ["Mannose", "mannose"]}
MAN = {"id": "RM:0135630", "name": "Mannose", "score": 3.48,
       "categories": [SM], "neighbors_count": 73, "synonyms": []}
GDP_MAN = {"id": "RM:0157861", "name": "GDP-mannose", "score": 1.32,
           "categories": [SM], "neighbors_count": 119, "synonyms": []}

INS_LOINC = {"id": "LOINC:LP14676-8", "name": "Insulin", "score": 3.5, "categories": ["biolink:Protein"],
             "taxon": None, "neighbors_count": 1926}
INS_HUMAN = {"id": "HGNC:6081", "name": "INS", "score": 3.3, "categories": ["biolink:Gene", "biolink:Protein"],
             "taxon": "NCBITaxon:9606", "neighbors_count": 7301}


class TestRerankSmallMolecule:
    def test_glutamate_skips_salt_and_ion_for_connected_acid(self):
        pick = _rerank_small_molecule("glutamate", [GLU_SALT, GLU_2MINUS, D_GLU, GLU_ACID])
        assert pick["id"] == "RM:0135893"

    def test_mannose_skips_glycoprotein_conjugate_and_non_matching_names(self):
        pick = _rerank_small_molecule("mannose", [MAN_CONJ, MAN, GDP_MAN])
        assert pick["id"] == "RM:0135630"  # GDP-mannose is better connected but is not "mannose"

    def test_no_lexical_match_returns_none_to_keep_top_hit(self):
        assert _rerank_small_molecule("mannose", [GDP_MAN, GLU_ACID]) is None

    def test_query_naming_a_derivative_is_not_screened(self):
        salt_rich = dict(GLU_SALT, name="Monosodium glutamate", neighbors_count=40, synonyms=["sodium glutamate"])
        assert _rerank_small_molecule("sodium glutamate", [salt_rich])["id"] == "CHEBI:13107"

    def test_one_edge_node_never_wins(self):
        lone = dict(MAN, neighbors_count=1)
        assert _rerank_small_molecule("mannose", [lone]) is None


def test_biomapper_candidates_try_chosen_then_primary_then_xrefs():
    from kestrel_backend.graph.nodes.entity_resolution import (
        _biomapper_candidate_curies,
    )

    got = _biomapper_candidate_curies(
        {"curie": "RM:0135893", "primary_curie": "CHEBI:13107", "xrefs": {"HMDB": ["HMDB0000148"]}},
        "metabolite",
    )
    assert got == ["RM:0135893", "CHEBI:13107", "HMDB:HMDB0000148"]


class TestResolveViaApiHinted:
    async def test_metabolite_reranks_and_reports_hinted_category(self, monkeypatch):
        calls = []

        async def fake(tool, args):
            calls.append((tool, args))
            if tool == "hybrid_search":
                return _hs(args["search_text"], [GLU_SALT, GLU_2MINUS, D_GLU])
            if args["search_text"] == "glutamic acid":
                return _hs("glutamic acid", [GLU_ACID])
            return _hs(args["search_text"], [GLU_SALT])

        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("glutamate", category=SM)
        assert res.curie == "RM:0135893"
        assert res.category == SM          # node is tagged Drug first; the hint wins
        assert res.confidence == 0.95      # confidence still from the top hybrid score
        assert {a["search_text"] for t, a in calls if t == "text_search"} == {"glutamate", "glutamic acid"}

    async def test_metabolite_low_top_score_still_routes_to_tier2(self, monkeypatch):
        calls = []

        async def fake(tool, args):
            calls.append(tool)
            return _hs(args["search_text"], [dict(MAN, score=0.4)])

        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        assert await resolve_via_api("mannose", category=SM) is None
        assert calls == ["hybrid_search"]  # no re-rank searches on a non-confident query

    async def test_protein_hint_filters_to_human(self, monkeypatch):
        calls = []

        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [INS_HUMAN] if args.get("taxon") else [INS_LOINC, INS_HUMAN])

        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("insulin", category="biolink:Protein")
        assert res.curie == "HGNC:6081"
        assert res.category == "biolink:Protein"
        assert calls[0]["taxon"] == "NCBITaxon:9606" and calls[0]["limit"] == 1

    async def test_empty_human_search_retries_without_taxon(self, monkeypatch):
        calls = []

        async def fake(tool, args):
            calls.append(args)
            return _hs(args["search_text"], [] if args.get("taxon") else [INS_LOINC])

        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("insulin", category="biolink:Protein")
        assert res.curie == "LOINC:LP14676-8"
        assert len(calls) == 2 and "taxon" not in calls[1]

    async def test_no_hint_is_unchanged(self, monkeypatch):
        calls = []

        async def fake(tool, args):
            calls.append((tool, args))
            return _hs(args["search_text"], [dict(GLU_SALT, categories=["biolink:Drug", SM])])

        monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
        res = await resolve_via_api("glutamate", category=None)
        assert res.curie == "CHEBI:13107" and res.category == "biolink:Drug"
        assert calls == [("hybrid_search", {"search_text": "glutamate", "limit": 1})]
