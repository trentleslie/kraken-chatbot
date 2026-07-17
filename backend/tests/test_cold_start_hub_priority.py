"""Axis B, Unit 4 — cold_start must analyze a rerouted intramodular hub, not drop it (Q6 / P0-1).

Triage reroutes intramodular hubs into cold_start_curies, but they carry a HIGH edge count and
cold_start prioritizes *ascending* by edge count capped at MAX_COLD_START=3 — so without a carve-out
a hub sorts last and is silently dropped, giving hubs LESS analysis (the opposite of the axis's
goal). The fix: a rerouted hub (is_intramodular_hub on its NoveltyScore) gets top priority.
"""

from kestrel_backend.graph.nodes.cold_start import (
    HUB_PRIORITY,
    _priority_score,
    score_entity_complexity,
)
from kestrel_backend.graph.state import NoveltyScore


def test_hub_priority_beats_zero_edge_entities():
    hubs = {"HUB:1"}
    # a hub (high edge count) must outrank even a genuine 0-edge cold_start entity
    assert _priority_score("HUB:1", 9999, hubs) == HUB_PRIORITY
    assert _priority_score("HUB:1", 9999, hubs) < _priority_score("Z:1", 0, hubs)
    # non-hubs keep the ascending-edge-count ordering
    assert _priority_score("Z:1", 0, hubs) == score_entity_complexity(0)
    assert _priority_score("S:1", 15, hubs) == score_entity_complexity(15)


def test_hub_survives_the_cap_under_the_real_sort():
    # Reproduce cold_start's sort+cap: 1 hub (high edge count) + 3 genuine 0-edge entities, MAX=3.
    hub_curies = {"HUB:1"}
    pairs = [
        ("HUB:1", _priority_score("HUB:1", 9999, hub_curies)),
        ("Z:1", _priority_score("Z:1", 0, hub_curies)),
        ("Z:2", _priority_score("Z:2", 0, hub_curies)),
        ("Z:3", _priority_score("Z:3", 0, hub_curies)),
    ]
    pairs.sort(key=lambda x: x[1])
    selected = [c for c, _ in pairs[:3]]
    assert "HUB:1" in selected, "rerouted hub must survive the MAX_COLD_START cap"


def test_hub_curies_derived_from_novelty_scores():
    # The hub set the fix keys on is exactly the is_intramodular_hub flag from triage's NoveltyScores.
    scores = [
        NoveltyScore(curie="HUB:1", raw_name="a", edge_count=800,
                     classification="well_characterized", is_intramodular_hub=True),
        NoveltyScore(curie="X:1", raw_name="b", edge_count=0, classification="cold_start"),
    ]
    hub_curies = {s.curie for s in scores if s.is_intramodular_hub}
    assert hub_curies == {"HUB:1"}
