"""L4 — validation eval: the PRODUCTION evidence-provenance labeler over real discovered bridges.

Reuses kg_bridge_leg_probe's bridge discovery (canonical endpoint pairs A→C, multi_hop_query
max_path_length=2) to obtain real [A, B, C] bridges, then runs the *production* labeler
(src.kestrel_backend.bridge_grounding.provenance.leg_tier + bridge_label) over each bridge's legs —
exactly what the bridge_grounding node emits in the pipeline.

Records the per-bridge chain-label distribution, the per-leg evidence-tier distribution, and the
curated-causal leg fraction, and compares the latter to the kg_bridge_leg_probe baseline (~23%).
This is a SANITY CHECK, not a calibration gate: the go signal for flipping the node enabled=True is a
sensible (non-collapsed) distribution whose curated-causal leg fraction sits near the baseline — a
material drop toward all-`none` would signal resolution residual voiding legs (hold the flip).

The endpoints are hand-verified canonical CURIEs (MONDO/CHEBI/…) — i.e. exactly what Tier 1 (PR #75)
+ Tier 2 (biomapper2) resolution now produce — so this isolates the LABELER's behavior given correct
resolution. Resolution quality itself is validated separately in those PRs.

Persists a timestamped JSON artifact by default (expensive/live-run SOP).
Usage: PYTHONPATH=. uv run python assessment_data/bridge_grounding_eval.py
"""

import asyncio
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

from assessment_data.kg_bridge_leg_probe import ENDPOINTS, bridges_for  # noqa: E402

from src.kestrel_backend.bridge_grounding.provenance import bridge_label, leg_tier  # noqa: E402

MAX_BRIDGES_PER_PAIR = 3  # bound the per-leg one_hop fetches (2 per bridge); leg_tier is heavy


async def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "timestamp": stamp,
        "note": "production labeler over real discovered bridges; canonical endpoints",
        "max_bridges_per_pair": MAX_BRIDGES_PER_PAIR,
        "endpoints": [],
        "chain_labels": collections.Counter(),
        "leg_tiers": collections.Counter(),
    }
    total_legs = cc_legs = 0
    print(f"{'kind':5} {'endpoint':24} {'#br':4} chain labels")
    print("-" * 100)
    for label, a, c, kind in ENDPOINTS:
        try:
            paths, _edges, err = await bridges_for(a, c)
        except Exception as ex:
            print(f"{kind:5} {label:24} ERR {str(ex)[:50]}")
            continue
        if err or not paths:
            print(f"{kind:5} {label:24} no-paths")
            out["endpoints"].append({"label": label, "kind": kind, "bridges": 0})
            continue
        ep_labels = collections.Counter()
        for path in paths[:MAX_BRIDGES_PER_PAIR]:
            a0, b0, c0 = path
            t1 = await leg_tier(a0, b0)
            t2 = await leg_tier(b0, c0)
            chain = bridge_label(t1, t2)
            ep_labels[chain] += 1
            out["chain_labels"][chain] += 1
            out["leg_tiers"][t1] += 1
            out["leg_tiers"][t2] += 1
            total_legs += 2
            cc_legs += int(t1 == "curated-causal") + int(t2 == "curated-causal")
        out["endpoints"].append({
            "label": label, "kind": kind, "bridges": sum(ep_labels.values()),
            "chain_labels": dict(ep_labels)})
        print(f"{kind:5} {label:24} {sum(ep_labels.values()):<4} {dict(ep_labels)}", flush=True)

    print("-" * 100)
    pct = (100 * cc_legs / total_legs) if total_legs else 0
    out["curated_causal_legs"] = f"{cc_legs}/{total_legs}"
    out["curated_causal_leg_pct"] = round(pct, 1)
    out["chain_labels"] = dict(out["chain_labels"])
    out["leg_tiers"] = dict(out["leg_tiers"])
    print(f"curated-causal LEG fraction: {cc_legs}/{total_legs} ({pct:.0f}%)  [baseline ~23%]")
    print(f"per-leg tier distribution:   {out['leg_tiers']}")
    print(f"per-bridge chain labels:     {out['chain_labels']}")

    rundir = Path(__file__).parent / "bridge_grounding_eval_runs"
    rundir.mkdir(exist_ok=True)
    art = rundir / f"{stamp}.json"
    art.write_text(json.dumps(out, indent=2))
    print(f"artifact: {art}")


if __name__ == "__main__":
    asyncio.run(main())
