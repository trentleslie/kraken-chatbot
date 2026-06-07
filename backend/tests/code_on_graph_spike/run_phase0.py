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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Every run is persisted here by default (a powered run costs hours of live
# Kestrel + SDK; losing the artifact to a forgotten --out flag is not acceptable).
RUNS_DIR = Path(__file__).resolve().parent / "runs"

_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV)

from .baseline import run_baseline
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
                     n_pilot: int = 15, progress=None, checkpoint=None) -> dict:
    """`checkpoint(baseline_records, iterate_records, idx)` is invoked after each item so a
    crash mid-run never discards completed work (the run is hours of live Kestrel + SDK)."""
    import time
    pilot = await run_pilot(rest, items, n_pilot)
    if progress:
        progress(f"pilot: R0(overall)={pilot['r0']['overall']:.2f} "
                 f"by_stratum={ {s: round(v, 2) for s, v in pilot['r0']['by_stratum'].items()} } "
                 f"powered_n={pilot['powered_n']} gate_form={pilot['gate_form']}")
    baseline_records: list[dict] = []
    iterate_records: list[dict] = []
    nb = ni = 0
    t0 = time.time()
    for idx, item in enumerate(items, 1):
        br = await run_baseline(rest, item)
        ir = await run_iterate_item_k(rest, item, llm_fn, k)
        baseline_records.append(br)
        iterate_records.append(ir)
        nb += int(br["hit"])
        ni += int(ir["hit"])
        if checkpoint:
            checkpoint(baseline_records, iterate_records, idx)
        if progress:
            mins = (time.time() - t0) / 60
            progress(f"[{idx}/{len(items)} t+{mins:.0f}m] {str(item.get('stratum'))[:4]:<4} "
                     f"base={'H' if br['hit'] else '.'} iter={'H' if ir['hit'] else '.'} "
                     f"| running base={nb}/{idx} iter={ni}/{idx} | gv={ir['grounding_violations']} "
                     f"var={ir['variance']}")
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
    print(f"  finding-level hallucinations: {g['finding_level_hallucinations']}   "
          f"query-arg leakage (caveat): {g['query_arg_leakage']}   cost(advisory): {g['cost_advisory']}")
    print(f"  per-stratum lift: " + ", ".join(
        f"{s}: {t['baseline_recall']:.2f}->{t['iterate_recall']:.2f}" for s, t in g["per_stratum"].items()))
    print(f"\n  VERDICT: {g['verdict']}\n")


async def _amain(args) -> int:
    items = load_gold_items()
    if args.limit:
        items = items[: args.limit]
    # Resolve the output path up front so checkpoints and the final result share it.
    # Persist by default — a forgotten flag must never discard an hours-long run.
    if args.out:
        out_path = Path(args.out)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = RUNS_DIR / f"phase0_n{len(items)}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".partial.json")

    def checkpoint(bl, it, idx):  # crash-safety: keep completed items if the run dies mid-flight
        ckpt_path.write_text(json.dumps(
            {"_partial_through": idx, "n_total": len(items),
             "baseline_records": bl, "iterate_records": it}, indent=2, default=str))

    print(f"Phase-0 gate: {len(items)} items, k={args.k or 'config'}. Live progress below.", flush=True)
    print(f"  (checkpointing to {ckpt_path.name} after each item)\n", flush=True)
    async with KestrelREST() as rest:
        result = await run_phase0(rest, default_llm_fn, items, k=args.k,
                                  n_pilot=min(15, len(items)),
                                  progress=lambda m: print("  " + m, flush=True),
                                  checkpoint=checkpoint)
    _print_report(result)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    if ckpt_path.exists():
        ckpt_path.unlink()  # completed cleanly — drop the partial
    print(f"  results saved → {out_path}", flush=True)
    return _EXIT.get(result["gate"]["verdict"], 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of gold items (smoke runs)")
    ap.add_argument("--k", type=int, default=None, help="iterate reruns (default config.k_reruns)")
    ap.add_argument("--out", type=str, default=None,
                    help="override the result-JSON path (default: runs/phase0_n<N>_<timestamp>.json)")
    return asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
