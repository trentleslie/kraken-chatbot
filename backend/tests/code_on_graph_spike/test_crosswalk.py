"""Unit 0.2 — DrugMechDB record → Kestrel gold item (mocked resolver)."""
from tests.code_on_graph_spike.crosswalk import to_gold_item
from tests.code_on_graph_spike.drugmechdb import DmdbRecord, DmdbInterior


class FakeRest:
    def __init__(self, resolve_map):
        self.resolve_map = resolve_map
        self.kestrel_calls = 0

    async def resolve(self, name, category=None):
        self.kestrel_calls += 1
        c = self.resolve_map.get(name)
        return (c, 1000 if c else None)


def _rec():
    return DmdbRecord(id="r1", drug_name="imatinib", drug_id="MESH:1",
                      disease_name="CML", disease_id="MESH:2",
                      interior=[DmdbInterior(id="UniProt:P00519", name="BCR/ABL", label="Protein")])


async def test_to_gold_item_resolves_all_endpoints():
    rest = FakeRest({"imatinib": "CHEBI:45783", "CML": "MONDO:0011996", "BCR/ABL": "NCBIGene:25"})
    item = await to_gold_item(rest, _rec())
    assert item is not None
    assert item["start_curie"] == "CHEBI:45783"
    assert item["gold_target_curie"] == "MONDO:0011996"
    assert item["gold_bridge_curies"] == ["NCBIGene:25"]
    assert item["stratum"] == "random" and item["hop_length"] == 2


async def test_to_gold_item_none_when_drug_unresolved():
    rest = FakeRest({"CML": "MONDO:0011996", "BCR/ABL": "NCBIGene:25"})  # no drug
    assert await to_gold_item(rest, _rec()) is None


async def test_to_gold_item_none_when_an_interior_unresolved():
    rest = FakeRest({"imatinib": "CHEBI:45783", "CML": "MONDO:0011996"})  # bridge missing
    assert await to_gold_item(rest, _rec()) is None
