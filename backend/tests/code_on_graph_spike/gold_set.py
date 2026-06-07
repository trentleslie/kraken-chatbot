"""Build the unified 100-item gold set: 20 committed anchors + 80 random DrugMechDB.

The random slice uses a frozen seed (P5). Reachability is **hop-agnostic** — any path
≤ max_path_length connecting drug↔disease — and deliberately NOT the 2-hop baseline
query (using the baseline would keep only items the static arm already solves and
delete iteration's headroom; plan finding / R6).
"""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from .anchors import load_anchors
from .config import CONFIG
from .crosswalk import to_gold_item
from .drugmechdb import DmdbRecord, all_records
from .kestrel_rest import KestrelREST, parse_paths

GOLD_SET_PATH = Path(__file__).parent.parent / "fixtures" / "code_on_graph_spike" / "gold_set.json"


async def is_reachable(rest: KestrelREST, drug: str, disease: str, max_hops: int) -> bool:
    """Hop-agnostic existence check: does ANY path ≤ max_hops connect the endpoints?
    NOT the 2-hop baseline (avoids rigging the test toward the static arm)."""
    data = await rest.multi_hop([drug], [disease], max_path_length=max_hops,
                                min_path_length=1, limit=1, mode="slim")
    return bool(parse_paths(data))


async def _evaluate(rest: KestrelREST, rec: DmdbRecord, max_hops: int) -> dict | None:
    if len(rec.interior) + 1 > max_hops:
        return None  # gold path longer than the executor cap -> unrecoverable by either arm
    item = await to_gold_item(rest, rec, stratum="random")
    if not item:
        return None
    if not await is_reachable(rest, item["start_curie"], item["gold_target_curie"], max_hops):
        return None
    return item


async def build_random_slice(rest: KestrelREST, records: list[DmdbRecord], n: int,
                             seed: int, max_hops: int, concurrency: int = 16) -> list[dict]:
    """Seeded shuffle, then resolve + reachability-filter until n items survive.

    Records are evaluated in parallel chunks for throughput, but survivors are appended
    in the original shuffled order, so the result is reproducible for a given seed
    (Kestrel resolution/reachability are deterministic)."""
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    out: list[dict] = []
    idx = 0
    while len(out) < n and idx < len(shuffled):
        chunk = shuffled[idx:idx + concurrency]
        idx += len(chunk)
        results = await asyncio.gather(*[_evaluate(rest, r, max_hops) for r in chunk],
                                       return_exceptions=True)
        for item in results:  # preserve shuffled order within the chunk
            if isinstance(item, dict):  # skip None (filtered) and Exceptions (transient)
                out.append(item)
                if len(out) >= n:
                    break
    return out[:n]


def _anchor_to_item(a) -> dict:
    return {
        "trial_id": a.trial_id, "drug": a.drug, "start_curie": a.start_curie,
        "bridge": a.bridge, "gold_bridge_curies": a.gold_bridge_curies,
        "gold_target_curie": a.gold_target_curie, "stratum": a.stratum,
        "hop_length": a.hop_length, "difficulty": a.difficulty, "source": "anchor",
    }


def build_unified(anchors_items: list[dict], random_items: list[dict]) -> dict:
    items = anchors_items + random_items
    ids = [i["trial_id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trial_id across anchors + random slice")
    return {
        "_meta": {
            "n": len(items), "anchors": len(anchors_items), "random": len(random_items),
            "seed": CONFIG.drugmechdb_sample_seed, "max_path_length": CONFIG.max_path_length,
            "strata": {s: sum(i["stratum"] == s for i in items) for s in ("t2d", "alzheimers", "random")},
        },
        "items": items,
    }


async def build_and_write(rest: KestrelREST, n_random: int | None = None) -> dict:
    n_random = n_random if n_random is not None else (CONFIG.n_target - 20)
    anchors = [_anchor_to_item(a) for a in load_anchors()]
    random_items = await build_random_slice(
        rest, all_records(), n=n_random, seed=CONFIG.drugmechdb_sample_seed, max_hops=CONFIG.max_path_length)
    unified = build_unified(anchors, random_items)
    GOLD_SET_PATH.write_text(json.dumps(unified, indent=2))
    return unified
