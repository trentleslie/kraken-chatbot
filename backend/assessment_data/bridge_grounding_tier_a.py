"""U8 Tier A calibration spike runner — PAID (real NCBI EFetch + real LLM labeling).

Builds each PANEL chain as a synthetic Bridge, retrieves per-leg co-occurrence pools, labels each
leg via the Claude Agent SDK, scores the chain (U4), and applies the PRE-REGISTERED gate (panel.py).
Persists every artifact by default to a timestamped run dir (expensive-run SOP) — pools, per-abstract
labels, counts, scores, the verdict, and a manifest pinning the model + limit + thresholds.

Reproducibility caveat: the Claude Agent SDK exposes no temperature, so determinism cannot be pinned
via temp=0. The frozen per-abstract label snapshot is the recomputable artifact; the model is pinned.

Usage (from backend/):
    NCBI_API_KEY=... uv run python assessment_data/bridge_grounding_tier_a.py [--model M] [--limit 50] [--out DIR]

Requires Claude SDK auth (claude login) and, recommended, NCBI_API_KEY (runs ~3x slower without).
This is a build-precondition TRIPWIRE (falsification only), not calibration — see panel.py.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.kestrel_backend.bridge_grounding.labeling import label_leg_via_sdk, tally_labels
from src.kestrel_backend.bridge_grounding.panel import (
    PANEL,
    PRE_REGISTERED,
    ChainResult,
    ChainSpec,
    build_bridge,
    evaluate_tier_a,
)
from src.kestrel_backend.bridge_grounding.prompts import build_leg_prompt
from src.kestrel_backend.bridge_grounding.retrieval import cooccurrence_pmids, dedupe_co_mention
from src.kestrel_backend.bridge_grounding.scoring import DECISION_INSUFFICIENT, score_chain
from src.kestrel_backend.graph.sdk_utils import DEFAULT_MODEL_NAME
from src.kestrel_backend.pubmed_client import fetch_abstracts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tier_a")

LEGS = ((0, 1), (1, 2))  # A–B, B–C


async def _score_chain(spec: ChainSpec, model: str, limit: int) -> tuple[ChainResult, dict]:
    bridge = build_bridge(spec)
    names = bridge.entity_names

    # Retrieve both legs, then keep-first co-mention dedup across them.
    raw_pools = [await cooccurrence_pmids(names[i], names[j], limit=limit) for (i, j) in LEGS]
    a_kept, b_kept, dropped = dedupe_co_mention(raw_pools[0], raw_pools[1])
    leg_pmids = [a_kept, b_kept]

    leg_counts: list[tuple[int, int, int]] = []
    leg_off_topic: list[float] = []
    leg_snaps: list[dict] = []
    for idx, (i, j) in enumerate(LEGS):
        pmids = leg_pmids[idx]
        bodies = await fetch_abstracts(pmids)
        abstracts = [{"pmid": p, "abstract": bodies[p]} for p in pmids if p in bodies]
        pred = bridge.predicates[idx] if idx < len(bridge.predicates) else None
        fwd = bridge.predicate_directions[idx] if idx < len(bridge.predicate_directions) else None
        direction_known = bool(pred)

        if abstracts:
            prompt = build_leg_prompt(
                names[i], names[j], pred, fwd, abstracts, direction_known=direction_known)
            labels, _usage = await label_leg_via_sdk(prompt, model=model)
        else:
            labels = []
        counts = tally_labels(labels)

        leg_counts.append((counts["support"], counts["refute"], counts["neither"]))
        labeled = counts["support"] + counts["refute"] + counts["neither"]
        denom = labeled + counts["off_topic"]
        leg_off_topic.append((counts["off_topic"] / denom) if denom else 0.0)
        leg_snaps.append({
            "leg": [names[i], names[j]], "predicate": pred, "forward": fwd,
            "pool_size": len(pmids), "with_bodies": len(abstracts),
            "counts": counts, "labels": labels,
        })

    cs = score_chain(leg_counts, binding_leg_floor=2)
    min_labeled = min((sum(c) for c in leg_counts), default=0)
    result = ChainResult(
        name=spec.name, polarity=spec.polarity, decision=cs.decision,
        support_fraction=None if cs.decision == DECISION_INSUFFICIENT else cs.support_fraction,
        min_labeled_per_leg=min_labeled,
        max_off_topic_fraction=max(leg_off_topic, default=0.0),
    )
    snap = {
        "name": spec.name, "polarity": spec.polarity, "entity_names": names,
        "dropped_co_mention": dropped, "legs": leg_snaps,
        "score": {"support_fraction": cs.support_fraction, "strong_leg_fraction": cs.strong_leg_fraction,
                  "strong_leg_n": cs.strong_leg_n, "decision": cs.decision,
                  "weak_leg_index": cs.weak_leg_index},
        "result": result.__dict__,
    }
    return result, snap


async def main() -> None:
    ap = argparse.ArgumentParser(description="U8 Tier A calibration spike (paid).")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Labeling model (pinned in manifest).")
    ap.add_argument("--limit", type=int, default=50, help="Per-leg PMID cap (<=50).")
    ap.add_argument("--out", default=None, help="Output dir (default: timestamped run dir).")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.out) if args.out else (
        Path(__file__).parent / "bridge_grounding_tier_a_runs" / stamp)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Tier A run -> %s (model=%s, limit=%d)", outdir, args.model, args.limit)

    results: list[ChainResult] = []
    for spec in PANEL:
        logger.info("scoring chain: %s (%s)", spec.name, spec.polarity)
        result, snap = await _score_chain(spec, args.model, args.limit)
        (outdir / f"{spec.name}.json").write_text(json.dumps(snap, indent=2))
        logger.info("  -> decision=%s support_fraction=%s off_topic_max=%.2f min_labeled=%d",
                    result.decision, result.support_fraction,
                    result.max_off_topic_fraction, result.min_labeled_per_leg)
        results.append(result)

    verdict = evaluate_tier_a(results)
    manifest = {
        "timestamp": stamp, "model": args.model, "limit": args.limit,
        "thresholds": PRE_REGISTERED.__dict__, "verdict": verdict,
        "results": [r.__dict__ for r in results],
    }
    (outdir / "verdict.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n" + "=" * 70)
    print(f"TIER A VERDICT: passed={verdict.get('passed')} "
          f"(reason={verdict.get('reason', 'evaluated')})")
    if "margin" in verdict:
        print(f"  margin={verdict['margin']:.3f} (>= {PRE_REGISTERED.margin_min})")
        print(f"  checks={verdict['checks']}")
    print(f"  artifacts: {outdir}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
