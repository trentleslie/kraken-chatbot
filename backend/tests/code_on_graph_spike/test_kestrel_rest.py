"""Unit 0.1 — REST client + cap-free parser + grounding (mocked, no live KG)."""
import httpx
import respx

from tests.code_on_graph_spike.kestrel_rest import (
    KestrelREST, parse_paths, path_contains_all, any_path_recovers, is_grounded,
)

BASE = "https://kestrel.nathanpricelab.com/api"


# --- pure parser / matching (no network) ---

def test_parse_paths_extracts_nested_results_paths():
    resp = {"results": [
        {"paths": [["CHEBI:1", "NCBIGene:2", "MONDO:3"]]},
        {"paths": [["CHEBI:1", "MONDO:3"]]},
    ]}
    assert parse_paths(resp) == [["CHEBI:1", "NCBIGene:2", "MONDO:3"], ["CHEBI:1", "MONDO:3"]]


def test_parse_paths_handles_empty_and_missing():
    assert parse_paths({}) == []
    assert parse_paths({"results": []}) == []
    assert parse_paths({"results": [{"paths": []}]}) == []


def test_path_contains_all_is_canonical_case_insensitive():
    # frozen bridge unit: ALL gold interior nodes, canonical match (finding #3 casing)
    assert path_contains_all(["chebi:6801", "ncbigene:5562", "mondo:3"], ["CHEBI:6801", "NCBIGene:5562"])
    assert not path_contains_all(["CHEBI:6801", "MONDO:3"], ["NCBIGene:5562"])


def test_any_path_recovers():
    paths = [["CHEBI:1", "MONDO:3"], ["CHEBI:1", "NCBIGene:5562", "MONDO:3"]]
    assert any_path_recovers(paths, ["NCBIGene:5562"])
    assert not any_path_recovers(paths, ["NCBIGene:9999"])


# --- REST client (respx-mocked) ---

async def test_resolve_prefers_highest_degree_hit():
    with respx.mock(base_url=BASE) as mock:
        mock.post("/hybrid-search").mock(return_value=httpx.Response(200, json={
            "metformin": [
                {"id": "NCBIGene:99", "neighbors_count": 12},     # ortholog-like, low degree
                {"id": "CHEBI:6801", "neighbors_count": 5675},    # the real one
            ]
        }))
        async with KestrelREST() as rest:
            curie, deg = await rest.resolve("metformin")
            assert curie == "CHEBI:6801" and deg == 5675
            assert rest.kestrel_calls == 1


async def test_multi_hop_round_trips_through_parser():
    with respx.mock(base_url=BASE) as mock:
        mock.post("/multi-hop").mock(return_value=httpx.Response(200, json={
            "results": [{"paths": [["CHEBI:6801", "MONDO:0005148"]]}]
        }))
        async with KestrelREST() as rest:
            data = await rest.multi_hop(["CHEBI:6801"], ["MONDO:0005148"], max_path_length=2, limit=100)
            assert parse_paths(data) == [["CHEBI:6801", "MONDO:0005148"]]


async def test_is_grounded_canonical_match():
    class _Fake:
        async def equivalent_ids(self, c):
            return set()
    assert await is_grounded(_Fake(), "CHEBI:6801", {"chebi:6801"})  # casing-insensitive
    assert not await is_grounded(_Fake(), "CHEBI:9999", {"CHEBI:6801"})


async def test_is_grounded_via_equivalent_ids():
    class _Fake:
        async def equivalent_ids(self, c):
            return {"CHEBI:6801"}  # emitted MESH id is the equivalent of a returned CHEBI node
    assert await is_grounded(_Fake(), "MESH:D008687", {"CHEBI:6801"})
