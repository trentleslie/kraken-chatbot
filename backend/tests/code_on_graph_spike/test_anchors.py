"""Unit 0.1 — anchor loader/validator over the committed gold set."""
from tests.code_on_graph_spike.anchors import (
    load_anchors, smoke_test_anchors, validate_anchors,
)


def test_loads_20_balanced_unique_anchors():
    anchors = load_anchors()
    assert len(anchors) == 20
    assert sum(a.stratum == "t2d" for a in anchors) == 10
    assert sum(a.stratum == "alzheimers" for a in anchors) == 10
    assert len({a.trial_id for a in anchors}) == 20
    # every anchor carries a measured difficulty and at least one gold bridge
    assert all(a.difficulty in ("easy", "hard") for a in anchors)
    assert all(a.gold_bridge_curies for a in anchors)


def test_difficulty_spread_matches_measured():
    anchors = load_anchors()
    assert sum(a.difficulty == "easy" for a in anchors) == 14
    assert sum(a.difficulty == "hard" for a in anchors) == 6


def test_smoke_test_anchors_are_easy_and_baseline_recovers():
    smoke = smoke_test_anchors(load_anchors())
    assert len(smoke) >= 2
    assert all(a.difficulty == "easy" and a.baseline_2hop_recovers for a in smoke)


async def test_validate_flags_low_degree_ortholog():
    anchors = load_anchors()[:1]

    class _FakeRest:
        kestrel_calls = 0

        async def degree(self, curie):
            return 12  # suspiciously low -> ortholog mismatch (finding #3)

    report = await validate_anchors(_FakeRest(), anchors)
    assert report["ok"] is False
    assert any("ortholog" in f["reason"] for f in report["flagged"])


async def test_validate_passes_for_high_degree_human_genes():
    anchors = load_anchors()[:2]

    class _FakeRest:
        kestrel_calls = 0

        async def degree(self, curie):
            return 5000  # plausible human gene

    report = await validate_anchors(_FakeRest(), anchors)
    assert report["ok"] is True
    assert report["flagged"] == []
