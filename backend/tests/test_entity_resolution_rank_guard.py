"""Unit 2 — rank-collapse guard in entity_resolution (Axis D, Guard 1).

Two seams:
  * ``_apply_rank_guard`` (Tier-1): abstain-only — no candidate set is in scope at Tier-1, so a
    species→genus collapse goes straight to method="failed" + rank_collapsed=True (reader budget).
  * ``_apply_rank_guard_tier2``: resolve-finer when the prefetched candidate set holds a node at the
    requested rank (same genus), else abstain.

Byte-identical guarantee: when the label yields no rank (non-taxa) the guard is a pass-through.
"""

import json

import pytest

from kestrel_backend.graph.nodes import entity_resolution
from kestrel_backend.graph.nodes.entity_resolution import (
    _apply_rank_guard,
    _apply_rank_guard_tier2,
    resolve_via_api,
)
from kestrel_backend.graph.state import EntityResolution


def _res(raw, curie, name, method="exact", conf=0.95):
    return EntityResolution(
        raw_name=raw, curie=curie, resolved_name=name, category="biolink:OrganismTaxon",
        confidence=conf, method=method,
    )


# ----------------------------- Tier-1 guard (abstain-only) -----------------------------

class TestTier1Guard:
    def test_species_resolving_to_genus_abstains(self):
        collapsed = _res("Ruminococcus gnavus", "NCBITaxon:1263", "Ruminococcus")
        out = _apply_rank_guard("Ruminococcus gnavus", collapsed)
        assert out.curie is None
        assert out.method == "failed"
        assert out.rank_collapsed is True
        assert out.requested_rank == "species"
        assert out.resolved_rank == "genus"

    def test_genus_resolving_to_genus_is_untouched(self):
        clean = _res("Ruminococcus", "NCBITaxon:1263", "Ruminococcus")
        out = _apply_rank_guard("Ruminococcus", clean)
        assert out is clean  # pass-through, byte-identical

    def test_non_taxa_is_untouched(self):
        clean = _res("glucose", "CHEBI:17234", "glucose")
        out = _apply_rank_guard("glucose", clean)
        assert out is clean

    def test_already_failed_is_untouched(self):
        failed = EntityResolution(raw_name="Ruminococcus gnavus", curie=None, method="failed")
        out = _apply_rank_guard("Ruminococcus gnavus", failed)
        assert out is failed


# ----------------------------- Tier-2 guard (resolve-finer / abstain) -----------------------------

class TestTier2Guard:
    def test_resolve_finer_prefers_species_candidate(self):
        chosen = _res("Ruminococcus gnavus", "NCBITaxon:1263", "Ruminococcus")
        candidates = [
            {"curie": "NCBITaxon:1263", "name": "Ruminococcus", "category": "biolink:OrganismTaxon", "score": 2.0},
            {"curie": "NCBITaxon:33038", "name": "Ruminococcus gnavus", "category": "biolink:OrganismTaxon", "score": 1.5},
        ]
        out = _apply_rank_guard_tier2("Ruminococcus gnavus", chosen, candidates)
        assert out.curie == "NCBITaxon:33038"
        assert out.resolved_name == "Ruminococcus gnavus"
        assert out.rank_collapsed is False

    def test_abstain_when_no_finer_candidate(self):
        chosen = _res("Ruminococcus gnavus", "NCBITaxon:1263", "Ruminococcus")
        candidates = [
            {"curie": "NCBITaxon:1263", "name": "Ruminococcus", "category": "biolink:OrganismTaxon", "score": 2.0},
            {"curie": "NCBITaxon:999", "name": "Blautia", "category": "biolink:OrganismTaxon", "score": 1.0},
        ]
        out = _apply_rank_guard_tier2("Ruminococcus gnavus", chosen, candidates)
        assert out.curie is None
        assert out.method == "failed"
        assert out.rank_collapsed is True

    def test_no_collapse_is_untouched(self):
        chosen = _res("Ruminococcus gnavus", "NCBITaxon:33038", "Ruminococcus gnavus")
        candidates = [{"curie": "NCBITaxon:33038", "name": "Ruminococcus gnavus", "category": None, "score": 2.0}]
        out = _apply_rank_guard_tier2("Ruminococcus gnavus", chosen, candidates)
        assert out is chosen

    def test_non_taxa_is_untouched(self):
        chosen = _res("glucose", "CHEBI:17234", "glucose")
        out = _apply_rank_guard_tier2("glucose", chosen, [])
        assert out is chosen


# ----------------------------- resolve_via_api integration (Tier-1 wired) -----------------------------

def _hs(search_text, rows):
    return {"content": [{"type": "text", "text": json.dumps({search_text: rows})}], "isError": False}


@pytest.mark.asyncio
async def test_resolve_via_api_abstains_on_rank_collapse(monkeypatch):
    genus_node = {
        "id": "NCBITaxon:1263", "name": "Ruminococcus", "score": 2.5,
        "categories": ["biolink:OrganismTaxon"],
    }

    async def fake(tool, args):
        return _hs(args["search_text"], [genus_node])

    monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
    out = await resolve_via_api("Ruminococcus gnavus")
    assert out is not None
    assert out.curie is None
    assert out.rank_collapsed is True
    assert out.method == "failed"


@pytest.mark.asyncio
async def test_category_fallback_preserves_failed_sentinel_on_abstain(monkeypatch):
    # Constrained call errors → unconstrained retry resolves species→genus → rank guard abstains.
    # The abstention (curie=None, method="failed") must NOT be relabeled "category-fallback", else it
    # would be silently dropped from every triage bucket.
    genus_node = {
        "id": "NCBITaxon:1263", "name": "Ruminococcus", "score": 2.5,
        "categories": ["biolink:OrganismTaxon"],
    }

    async def fake(tool, args):
        if args.get("category") is not None:
            return {"content": [], "isError": True}
        return _hs(args["search_text"], [genus_node])

    monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
    out = await resolve_via_api("Ruminococcus gnavus", category="biolink:OrganismTaxon")
    assert out is not None
    assert out.curie is None
    assert out.method == "failed"
    assert out.rank_collapsed is True


@pytest.mark.asyncio
async def test_resolve_via_api_clean_genus_is_unaffected(monkeypatch):
    genus_node = {
        "id": "NCBITaxon:1263", "name": "Ruminococcus", "score": 2.5,
        "categories": ["biolink:OrganismTaxon"],
    }

    async def fake(tool, args):
        return _hs(args["search_text"], [genus_node])

    monkeypatch.setattr(entity_resolution, "call_kestrel_tool", fake)
    out = await resolve_via_api("Ruminococcus")
    assert out is not None
    assert out.curie == "NCBITaxon:1263"
    assert out.rank_collapsed is False
