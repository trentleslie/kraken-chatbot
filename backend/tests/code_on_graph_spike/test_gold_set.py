"""Unit 0.2 — random slice + reachability filter + merge (mocked resolver)."""
import httpx
import pytest

from tests.code_on_graph_spike import gold_set as gold_set_mod
from tests.code_on_graph_spike.drugmechdb import DmdbRecord, DmdbInterior
from tests.code_on_graph_spike.gold_set import (
    GoldSetBuildError, build_random_slice, build_unified, is_reachable,
)


class FakeRest:
    """resolve_map: name->curie; reachable_pairs: set of (drug_curie, disease_curie)."""
    def __init__(self, resolve_map, reachable_pairs):
        self.resolve_map = resolve_map
        self.reachable_pairs = reachable_pairs
        self.kestrel_calls = 0

    async def resolve(self, name, category=None):
        self.kestrel_calls += 1
        c = self.resolve_map.get(name)
        return (c, 1000 if c else None)

    async def multi_hop(self, start, end, **kw):
        self.kestrel_calls += 1
        if (start[0], end[0]) in self.reachable_pairs:
            return {"results": [{"paths": [[start[0], "MID", end[0]]]}]}
        return {"results": []}


def _rec(i, interior_name="bridge", interior_count=1):
    interior = [DmdbInterior(id=f"P{i}", name=f"{interior_name}{i}", label="Protein")
                for _ in range(interior_count)]
    return DmdbRecord(id=f"r{i}", drug_name=f"drug{i}", drug_id=f"MESH:d{i}",
                      disease_name=f"dis{i}", disease_id=f"MESH:s{i}", interior=interior)


async def test_is_reachable_uses_existence_not_baseline():
    rest = FakeRest({}, {("CHEBI:1", "MONDO:1")})
    assert await is_reachable(rest, "CHEBI:1", "MONDO:1", 5) is True
    assert await is_reachable(rest, "CHEBI:1", "MONDO:9", 5) is False


async def test_build_random_slice_filters_and_stops_at_n():
    records = [_rec(i) for i in range(10)]
    # resolvable + reachable only for even i
    resolve_map, reachable = {}, set()
    for i in range(0, 10, 2):
        resolve_map[f"drug{i}"] = f"CHEBI:{i}"
        resolve_map[f"dis{i}"] = f"MONDO:{i}"
        resolve_map[f"bridge{i}"] = f"NCBIGene:{i}"
        reachable.add((f"CHEBI:{i}", f"MONDO:{i}"))
    rest = FakeRest(resolve_map, reachable)
    out = await build_random_slice(rest, records, n=3, seed=42, max_hops=5)
    assert len(out) == 3
    assert all(o["start_curie"].startswith("CHEBI:") for o in out)
    assert all(o["stratum"] == "random" for o in out)


async def test_build_random_slice_is_seed_reproducible():
    records = [_rec(i) for i in range(10)]
    rm = {f"drug{i}": f"CHEBI:{i}" for i in range(10)}
    rm.update({f"dis{i}": f"MONDO:{i}" for i in range(10)})
    rm.update({f"bridge{i}": f"NCBIGene:{i}" for i in range(10)})
    reach = {(f"CHEBI:{i}", f"MONDO:{i}") for i in range(10)}
    a = await build_random_slice(FakeRest(rm, reach), records, n=5, seed=7, max_hops=5)
    b = await build_random_slice(FakeRest(rm, reach), records, n=5, seed=7, max_hops=5)
    assert [x["dmdb_id"] for x in a] == [x["dmdb_id"] for x in b]


async def test_build_random_slice_excludes_paths_over_cap():
    deep = [_rec(0, interior_count=6)]  # hop_length 7 > max_hops 5
    rest = FakeRest({"drug0": "CHEBI:0", "dis0": "MONDO:0", "bridge0": "NCBIGene:0"},
                    {("CHEBI:0", "MONDO:0")})
    out = await build_random_slice(rest, deep, n=5, seed=1, max_hops=5)
    assert out == []


def test_build_unified_merges_and_dedups():
    anchors = [{"trial_id": "t2d-01", "stratum": "t2d"}]
    rand = [{"trial_id": "dmdb-x", "stratum": "random"}]
    u = build_unified(anchors, rand)
    assert u["_meta"]["n"] == 2 and u["_meta"]["anchors"] == 1 and u["_meta"]["random"] == 1
    assert len(u["items"]) == 2


def test_build_unified_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        build_unified([{"trial_id": "dup", "stratum": "t2d"}], [{"trial_id": "dup", "stratum": "random"}])


# --- silent-drop regression: transport failures must not masquerade as non-reachability ---

class FlakyRest(FakeRest):
    """Like FakeRest, but multi_hop raises a transient ReadTimeout for any start CURIE in
    `fail_starts` until that start has failed `fail_times` times, then it recovers."""
    def __init__(self, resolve_map, reachable_pairs, fail_starts, fail_times=10 ** 9):
        super().__init__(resolve_map, reachable_pairs)
        self.fail_starts = set(fail_starts)
        self.fail_times = fail_times
        self._fails: dict[str, int] = {}

    async def multi_hop(self, start, end, **kw):
        s = start[0]
        if s in self.fail_starts and self._fails.get(s, 0) < self.fail_times:
            self._fails[s] = self._fails.get(s, 0) + 1
            self.kestrel_calls += 1
            raise httpx.ReadTimeout("simulated transient")
        return await super().multi_hop(start, end, **kw)


def _all_reachable(n):
    rm = {f"drug{i}": f"CHEBI:{i}" for i in range(n)}
    rm.update({f"dis{i}": f"MONDO:{i}" for i in range(n)})
    rm.update({f"bridge{i}": f"NCBIGene:{i}" for i in range(n)})
    reach = {(f"CHEBI:{i}", f"MONDO:{i}") for i in range(n)}
    return rm, reach


async def test_build_random_slice_raises_on_persistent_transport_failure(monkeypatch):
    # Reachability times out forever -> the build must fail loudly, NOT silently return
    # a short list (the bug: transport errors were dropped as if "not reachable").
    monkeypatch.setattr(gold_set_mod, "_RETRY_BACKOFF_BASE", 0)
    records = [_rec(i) for i in range(4)]
    rm, reach = _all_reachable(4)
    rest = FlakyRest(rm, reach, fail_starts={f"CHEBI:{i}" for i in range(4)})  # never recover
    with pytest.raises(GoldSetBuildError):
        await build_random_slice(rest, records, n=2, seed=1, max_hops=5)


async def test_build_random_slice_retries_transient_then_succeeds(monkeypatch):
    # A transient blip that recovers on retry must not drop the record.
    monkeypatch.setattr(gold_set_mod, "_RETRY_BACKOFF_BASE", 0)
    records = [_rec(i) for i in range(4)]
    rm, reach = _all_reachable(4)
    rest = FlakyRest(rm, reach, fail_starts={f"CHEBI:{i}" for i in range(4)}, fail_times=1)
    out = await build_random_slice(rest, records, n=4, seed=1, max_hops=5)
    assert len(out) == 4
    assert all(o["start_curie"].startswith("CHEBI:") for o in out)


async def test_build_random_slice_genuine_exhaustion_returns_short_not_raise():
    # No transport errors: a corpus that simply lacks enough reachable items returns a
    # short list (unchanged behavior), distinct from the transport-failure path.
    records = [_rec(i) for i in range(2)]
    rm, reach = _all_reachable(2)
    out = await build_random_slice(FakeRest(rm, reach), records, n=5, seed=1, max_hops=5)
    assert len(out) == 2
