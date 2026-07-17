"""Unit 3 — resolution-contract: additive optional rank fields on EntityResolution.

The frozen model gains ``requested_rank`` / ``resolved_rank`` / ``rank_collapsed`` as additive
optionals. Old constructions (which omit them) must be unaffected — defaults apply and equality
still holds — and the new marker must round-trip via ``model_copy(update=...)``.
"""

from kestrel_backend.graph.state import EntityResolution


def test_defaults_are_inert():
    r = EntityResolution(raw_name="glucose", curie="CHEBI:17234", method="exact", confidence=0.95)
    assert r.requested_rank is None
    assert r.resolved_rank is None
    assert r.rank_collapsed is False


def test_old_construction_equality_unaffected():
    # Two objects built the legacy way (no rank fields) remain equal — no breaking change to readers
    # that compare EntityResolution instances.
    a = EntityResolution(raw_name="x", curie="C:1", method="exact", confidence=0.9)
    b = EntityResolution(raw_name="x", curie="C:1", method="exact", confidence=0.9)
    assert a == b


def test_marker_round_trips_via_model_copy():
    base = EntityResolution(raw_name="Ruminococcus gnavus", curie=None, method="failed")
    flagged = base.model_copy(
        update={"requested_rank": "species", "resolved_rank": "genus", "rank_collapsed": True}
    )
    assert flagged.rank_collapsed is True
    assert flagged.requested_rank == "species"
    assert flagged.resolved_rank == "genus"
    # frozen: original is untouched
    assert base.rank_collapsed is False


def test_direct_construction_with_rank_fields():
    r = EntityResolution(
        raw_name="Ruminococcus gnavus",
        curie=None,
        method="failed",
        requested_rank="species",
        resolved_rank="genus",
        rank_collapsed=True,
    )
    assert r.rank_collapsed is True
