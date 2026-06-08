"""Load and validate the committed 20-item disease-anchor gold set.

The anchors were hand-curated and difficulty-measured live (see
fixtures/gold_set_anchors.json). This loader re-validates that each CURIE still
resolves to the intended (human) node — guarding against the ortholog-mismatch
class that the naive resolver produced (plan finding #3).
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .kestrel_rest import KestrelREST
from kestrel_backend.graph.nodes.entity_resolution import _canonical_curie

ANCHORS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "code_on_graph_spike" / "gold_set_anchors.json"

# A resolved bridge below this degree is almost certainly a non-human ortholog
# (finding #3): the canonical human gene nodes are in the thousands.
MIN_PLAUSIBLE_BRIDGE_DEGREE = 100


class Anchor(BaseModel):
    trial_id: str
    drug: str
    start_curie: str
    bridge: str
    gold_bridge_curies: list[str]
    gold_target_curie: str
    stratum: str           # "t2d" | "alzheimers"
    hop_length: int
    difficulty: str        # "easy" | "hard"
    bridge_deg: int | None = None
    baseline_2hop_recovers: bool | None = None
    note: str | None = None


def load_anchors() -> list[Anchor]:
    data = json.loads(ANCHORS_FIXTURE.read_text())
    anchors = [Anchor(**a) for a in data["anchors"]]
    ids = [a.trial_id for a in anchors]
    if len(ids) != len(set(ids)):
        raise ValueError("anchor trial_id values are not unique")
    return anchors


def smoke_test_anchors(anchors: list[Anchor]) -> list[Anchor]:
    """The known-easy anchors used as a loop smoke test (plan Unit 0.1): difficulty
    easy AND the static 2-hop baseline already recovers the gold bridge."""
    return [a for a in anchors if a.difficulty == "easy" and a.baseline_2hop_recovers]


async def validate_anchors(rest: KestrelREST, anchors: list[Anchor]) -> dict:
    """Re-resolve each anchor's CURIEs against live Kestrel; flag any gold bridge
    that resolves to a suspiciously low-degree node (ortholog mismatch, finding #3).
    Returns a report; does not raise (callers decide)."""
    flags: list[dict] = []
    for a in anchors:
        for bridge_curie in a.gold_bridge_curies:
            deg = await rest.degree(bridge_curie)
            if deg is not None and deg < MIN_PLAUSIBLE_BRIDGE_DEGREE:
                flags.append({"trial_id": a.trial_id, "bridge": a.bridge,
                              "curie": bridge_curie, "degree": deg,
                              "reason": "low-degree (possible non-human ortholog)"})
        # endpoints must canonicalize cleanly
        for label, curie in (("start", a.start_curie), ("target", a.gold_target_curie)):
            if _canonical_curie(curie) is None:
                flags.append({"trial_id": a.trial_id, "curie": curie,
                              "reason": f"uncanonicalizable {label} CURIE"})
    return {"n": len(anchors), "flagged": flags, "ok": not flags,
            "kestrel_calls": rest.kestrel_calls}
