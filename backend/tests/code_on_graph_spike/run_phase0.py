"""Phase-0 kill-test orchestrator: pilot -> both arms -> score -> recall gate.

Run (live Kestrel REST + Claude SDK for the iterate arm):
    cd backend && uv run python -m tests.code_on_graph_spike.run_phase0 --limit 6 --k 1
    cd backend && uv run python -m tests.code_on_graph_spike.run_phase0            # full 50 x K

Exit code: 0 = PROCEED-TO-PHASE-1, 1 = NO-GO, 2 = INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV)

from .baseline import run_baseline_all
from .gate_recall import evaluate_gate
from .gold_set import GOLD_SET_PATH
from .iterate_loop import default_llm_fn, run_iterate_item_k
from .kestrel_rest import KestrelREST
from .pilot import run_pilot
from .recall_scorer import score

_EXIT = {"PROCEED-TO-PHASE-1": 0, "NO-GO": 1, "INCONCLUSIVE": 2}


def load_gold_items() -> list[dict]:
    if not GOLD_SET_PATH.exists():
        raise FileNotFoundError(f"{GOLD_SET_PATH} not found — build it first (gold_set.build_and_write).")
    return json.loads(GOLD_SET_PATH.read_text())["items"]


async def run_phase0(rest: KestrelREST, llm_fn, items: list[dict], k: int | None = None,
                     n_pilot: int = 15) -> dict:
    pilot = await run_pilot(rest, items, n_pilot)
    baseline_records = await run_baseline_all(rest, items)
    iterate_records = []
    for item in items:
        iterate_records.append(await run_iterate_item_k(rest, item, llm_fn, k))
    scored = score(baseline_records, iterate_records)
    gate = evaluate_gate(scored, pilot, iterate_records)
    return {"pilot": pilot, "gate": gate,
            "baseline_records": baseline_records, "iterate_records": iterate_records}


def _print_report(result: dict) -> None:
    g, p = result["gate"], result["pilot"]
    print("\n=== Phase-0 recall gate ===")
    print(f"  pilot R0 (overall): {p['r0']['overall']:.2f}  by stratum: "
          + ", ".join(f"{s}={v:.2f}" for s, v in p["r0"]["by_stratum"].items()))
    print(f"  powered N: {p['powered_n']}   gate form: {g['gate_form']}   N scored: {g['n']}")
    print(f"  baseline recall: {g['baseline_recall']:.2f}   iterate recall: {g['iterate_recall']:.2f}   "
          f"abs lift: {g['abs_lift']:+.2f}")
    print(f"  McNemar: b={g['discordant']['b_baseline_only']} c={g['discordant']['c_iterate_only']} "
          f"p={g['mcnemar_p']:.4f}  significant={g['significant']}")
    print(f"  hallucinated CURIEs: {g['hallucinated_curies']}   cost(advisory): {g['cost_advisory']}")
    print(f"  per-stratum lift: " + ", ".join(
        f"{s}: {t['baseline_recall']:.2f}->{t['iterate_recall']:.2f}" for s, t in g["per_stratum"].items()))
    print(f"\n  VERDICT: {g['verdict']}\n")


async def _amain(args) -> int:
    items = load_gold_items()
    if args.limit:
        items = items[: args.limit]
    async with KestrelREST() as rest:
        result = await run_phase0(rest, default_llm_fn, items, k=args.k, n_pilot=min(15, len(items)))
    _print_report(result)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    return _EXIT.get(result["gate"]["verdict"], 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of gold items (smoke runs)")
    ap.add_argument("--k", type=int, default=None, help="iterate reruns (default config.k_reruns)")
    ap.add_argument("--out", type=str, default=None, help="write full result JSON here")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
