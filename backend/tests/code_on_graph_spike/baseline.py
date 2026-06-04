"""Static baseline arm: one fixed full-depth /multi-hop query per item.

The static plan is a single multi-hop between the gold endpoints at the same reach
the iterate loop has (config.baseline_max_path_length == max_path_length) and the
same frozen `limit` (finding #2). A hit = a returned path containing ALL gold interior
nodes (the frozen bridge unit). Off-gold intermediates are collected for later EITL
pooling. Deterministic, so no cassette is needed for the baseline (the variance lives
only in the LLM iterate arm).
"""
from __future__ import annotations

from .config import CONFIG
from .kestrel_rest import KestrelREST, parse_paths, any_path_recovers


def _intermediates(paths: list[list[str]], start: str, target: str) -> list[str]:
    """Off-gold candidate bridges this method surfaced (for the EITL pool)."""
    return sorted({c for p in paths for c in p if c not in (start, target)})


async def run_baseline(rest: KestrelREST, item: dict) -> dict:
    start, target = item["start_curie"], item["gold_target_curie"]
    gold = item["gold_bridge_curies"]
    rec = {"trial_id": item["trial_id"], "method": "static", "stratum": item.get("stratum")}
    try:
        data = await rest.multi_hop([start], [target],
                                    max_path_length=CONFIG.baseline_max_path_length,
                                    limit=CONFIG.multi_hop_limit, mode="full")
    except Exception as exc:  # transport failure is not a method miss (R10)
        rec.update(hit=False, terminal_state="transport-failed", error=str(exc),
                   n_paths=0, intermediates=[], kestrel_calls=1)
        return rec
    paths = parse_paths(data)
    rec.update(
        hit=any_path_recovers(paths, gold),
        n_paths=len(paths),
        intermediates=_intermediates(paths, start, target),
        terminal_state="found" if paths else "empty",
        kestrel_calls=1,
    )
    return rec


async def run_baseline_all(rest: KestrelREST, items: list[dict]) -> list[dict]:
    return [await run_baseline(rest, it) for it in items]
