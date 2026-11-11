"""#124/#125: a null field on one KG row or model row must never discard the rest.

Both bugs were the same shape: ``d.get("x", default)`` returns ``None`` when the key is present with
a null value, so the default never applied and one row's ValidationError/TypeError took out the
whole entity (#124, glucose) or every shared neighbor (#125, 917 -> 0).
"""

import json

from kestrel_backend.graph.nodes import direct_kg
from kestrel_backend.graph.nodes.pathway_enrichment import parse_enrichment_result
from kestrel_backend.graph.nodes.synthesis import format_shared_neighbors


class TestDirectKgNullName:
    def _deduped(self, nodes, edges):
        deduped: dict = {}
        rows = [{"end_node_id": nid, "edge_ids": [i]} for i, nid in enumerate(nodes)]
        direct_kg._merge_into_deduped(deduped, rows, "established", nodes, edges)
        return deduped

    def test_null_name_neighbor_falls_back_to_curie_and_keeps_siblings(self):
        nodes = {"MONDO:0005148": {"name": "type 2 diabetes mellitus"}, "MONDO:0000001": {"name": None}}
        edges = {
            "0": ["HMDB:HMDB0000122", "biolink:causes", "MONDO:0005148", None, "infores:ctd", []],
            "1": ["HMDB:HMDB0000122", "biolink:related_to", "MONDO:0000001", None, "infores:hmdb", []],
        }
        diseases, findings = direct_kg.parse_deduped_diseases(
            "HMDB:HMDB0000122", "glucose", self._deduped(nodes, edges)
        )
        names = {d.disease_curie: d.disease_name for d in diseases}
        assert names == {"MONDO:0005148": "type 2 diabetes mellitus", "MONDO:0000001": "MONDO:0000001"}
        assert len(findings) == 2

    def test_null_predicate_falls_back_to_related_to(self):
        nodes = {"MONDO:0005148": {"name": "type 2 diabetes mellitus"}}
        edges = {"0": ["HMDB:HMDB0000122", None, "MONDO:0005148", None, None, []]}
        diseases, _ = direct_kg.parse_deduped_diseases(
            "HMDB:HMDB0000122", "glucose", self._deduped(nodes, edges)
        )
        assert diseases[0].predicate == "biolink:related_to"
        assert diseases[0].source == "unknown"

    def test_null_name_pathway_is_kept(self):
        nodes = {"REACT:R-HSA-1": {"name": None}}
        edges = {"0": ["HMDB:HMDB0000122", "biolink:participates_in", "REACT:R-HSA-1", None, "infores:reactome", []]}
        pathways, _ = direct_kg.parse_deduped_pathways(
            "HMDB:HMDB0000122", "glucose", self._deduped(nodes, edges)
        )
        assert pathways[0].pathway_name == "REACT:R-HSA-1"


def _enrichment(*neighbors):
    return json.dumps({"shared_neighbors": list(neighbors), "themes": []})


T2D = {
    "curie": "MONDO:0005148", "name": "type 2 diabetes mellitus", "category": "biolink:Disease",
    "degree": None, "is_hub": False, "connected_inputs": ["RM:0136379", "HGNC:13633"],
    "predicates": ["biolink:contributes_to"],
}


class TestSharedNeighborNullFields:
    def test_null_degree_is_kept_as_unknown_not_dropped(self):
        shared, _, errors = parse_enrichment_result(_enrichment(T2D))
        assert errors == []
        assert [(s.curie, s.degree, s.is_hub) for s in shared] == [("MONDO:0005148", None, False)]

    def test_null_name_and_category_get_defaults(self):
        row = dict(T2D, name=None, category=None, degree=42, connected_inputs=None, predicates=None)
        shared, _, errors = parse_enrichment_result(_enrichment(row))
        assert errors == []
        assert (shared[0].name, shared[0].category, shared[0].connected_inputs) == (
            "MONDO:0005148", "biolink:NamedThing", [],
        )

    def test_known_degree_still_flags_hub(self):
        shared, _, _ = parse_enrichment_result(_enrichment(dict(T2D, curie="GO:0005515", degree=50000)))
        assert shared[0].is_hub is True

    def test_one_bad_row_does_not_drop_the_others(self):
        good = dict(T2D, curie="HGNC:6081", degree=7301)
        shared, _, _ = parse_enrichment_result(_enrichment(T2D, good))
        assert {s.curie for s in shared} == {"MONDO:0005148", "HGNC:6081"}


def test_synthesis_omits_unknown_degree_instead_of_printing_none():
    shared, _, _ = parse_enrichment_result(
        _enrichment(T2D, dict(T2D, curie="GO:0005515", name="protein binding", is_hub=True))
    )
    text = format_shared_neighbors(shared, [])
    assert "None" not in text
    assert "Degree: 0" not in text and "— 0 edges" not in text
    assert "type 2 diabetes mellitus" in text
