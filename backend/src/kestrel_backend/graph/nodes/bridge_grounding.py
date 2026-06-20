"""L3 — bridge_grounding node: attach a deterministic evidence-provenance label to bridges.

For each ordered 3-node multi-hop bridge (A→B→C), label each leg's best KG-edge evidence tier
(via L1 ``provenance.leg_tier``) and compose a per-bridge chain summary (``provenance.bridge_label``).
Writes the labeled bridges to the separate ``grounded_bridges`` state key (NOT ``bridges``, whose
operator.add reducer would duplicate). Best-effort and never throws: a failure on one bridge is
isolated to ``bridge_grounding_errors`` and the node still returns.

Ships ``enabled=False`` (no-op, no Kestrel calls) until L4's validation eval gates the flip.
"""

import asyncio
import logging
from typing import Any

from ..pipeline_config import get_pipeline_config
from ..state import BridgeGrounding, DiscoveryState, LegSummary
from ..state_contracts import (
    BridgeGroundingInput,
    BridgeGroundingOutput,
    validate_state,
)
from ...bridge_grounding.provenance import bridge_label, cached_leg_fetcher, leg_tier

logger = logging.getLogger(__name__)


def _is_scoreable(bridge: Any) -> bool:
    """Ordered 3-node multi-hop bridges only.

    ``predicate_directions`` is non-empty ONLY for multi-hop bridges (subgraph "connecting"
    bridges leave it ``[]``); combined with the 3-entity check this selects A→B→C chains and
    excludes subgraph / 2-node / 4+-node bridges.
    """
    return len(bridge.entities) == 3 and bool(bridge.predicate_directions)


@validate_state(BridgeGroundingInput, BridgeGroundingOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """Label ordered 3-node bridges with their evidence provenance; default-off no-op."""
    # Read the flag fresh (not an import-time singleton) so get_pipeline_config.cache_clear() in
    # tests takes effect. Flag-off → no Kestrel calls, empty grounded_bridges.
    cfg = get_pipeline_config().bridge_grounding
    if not cfg.enabled:
        return {}

    bridges = state.get("bridges", []) or []
    scoreable = [b for b in bridges if _is_scoreable(b)][: cfg.max_scored_bridges]
    logger.info(
        "bridge_grounding: labeling %d/%d bridges (max_scored=%d)",
        len(scoreable), len(bridges), cfg.max_scored_bridges,
    )

    # One fetcher per run: bridges are labeled concurrently and a CURIE shared across bridges
    # (hub endpoints recur in a co-expression module) is fetched once, not once per bridge.
    fetch = cached_leg_fetcher(cfg.concurrency)

    async def _label(b: Any) -> tuple[Any, str | None]:
        try:
            a, mid, c = b.entities
            # leg_tier is best-effort (returns "none" on a Kestrel failure, never raises).
            t1 = await leg_tier(a, mid, fetch=fetch)
            t2 = await leg_tier(mid, c, fetch=fetch)
            grounding = BridgeGrounding(
                legs=[
                    LegSummary(from_curie=a, to_curie=mid, evidence_tier=t1),
                    LegSummary(from_curie=mid, to_curie=c, evidence_tier=t2),
                ],
                label=bridge_label(t1, t2),
            )
            return b.model_copy(update={"grounding": grounding}), None
        except Exception as e:  # per-bridge isolation: degrade this bridge, keep the node alive
            logger.warning("bridge_grounding: failed for %s: %s", b.path_description, e)
            return (
                b.model_copy(update={"grounding": BridgeGrounding(legs=[], label="no KG edge")}),
                f"bridge_grounding: {b.path_description}: {e}",
            )

    results = await asyncio.gather(*[_label(b) for b in scoreable])
    grounded = [g for g, _ in results]
    errors = [err for _, err in results if err is not None]
    return {"grounded_bridges": grounded, "bridge_grounding_errors": errors}
