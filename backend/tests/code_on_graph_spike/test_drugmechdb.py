"""Unit 0.2 — DrugMechDB record extraction (pure, no network)."""
from tests.code_on_graph_spike.drugmechdb import extract

IMATINIB = {
    "graph": {"_id": "DB00619_MESH_D015464_1", "disease": "CML (ph+)",
              "disease_mesh": "MESH:D015464", "drug": "imatinib",
              "drug_mesh": "MESH:D000068877", "drugbank": "DB:DB00619"},
    "nodes": [
        {"id": "MESH:D000068877", "label": "Drug", "name": "imatinib"},
        {"id": "UniProt:P00519", "label": "Protein", "name": "BCR/ABL"},
        {"id": "MESH:D015464", "label": "Disease", "name": "CML (ph+)"},
    ],
    "links": [
        {"key": "decreases activity of", "source": "MESH:D000068877", "target": "UniProt:P00519"},
        {"key": "causes", "source": "UniProt:P00519", "target": "MESH:D015464"},
    ],
}


def test_extract_single_interior():
    rec = extract(IMATINIB)
    assert rec is not None
    assert rec.drug_id == "MESH:D000068877" and rec.drug_name == "imatinib"
    assert rec.disease_id == "MESH:D015464"
    assert [i.id for i in rec.interior] == ["UniProt:P00519"]
    assert rec.interior[0].name == "BCR/ABL" and rec.interior[0].label == "Protein"


def test_extract_multi_interior_is_ordered():
    rec_dict = {
        "graph": {"_id": "x", "drug": "d", "disease": "dis"},
        "nodes": [{"id": "DRUG", "name": "d", "label": "Drug"},
                  {"id": "P1", "name": "p1", "label": "Protein"},
                  {"id": "PW", "name": "pw", "label": "Pathway"},
                  {"id": "DIS", "name": "dis", "label": "Disease"}],
        "links": [{"source": "DRUG", "target": "P1"}, {"source": "P1", "target": "PW"},
                  {"source": "PW", "target": "DIS"}],
    }
    rec = extract(rec_dict)
    assert rec is not None
    assert [i.id for i in rec.interior] == ["P1", "PW"]  # ordered, excludes drug & disease


def test_extract_returns_none_on_malformed():
    assert extract({}) is None
    assert extract({"nodes": [], "links": []}) is None
    # broken chain: interior id absent from nodes
    broken = {"graph": {}, "nodes": [{"id": "DRUG", "name": "d", "label": "Drug"},
                                     {"id": "DIS", "name": "x", "label": "Disease"}],
              "links": [{"source": "DRUG", "target": "MISSING"}, {"source": "MISSING", "target": "DIS"}]}
    assert extract(broken) is None
