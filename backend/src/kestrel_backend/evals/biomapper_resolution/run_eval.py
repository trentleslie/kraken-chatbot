"""False-cold-start gold-set eval for the biomapper pre-resolver (merge gate).

Runs the **production** resolution path (``resolve_entity`` + ``reconcile_to_kestrel``) over a
gold set of human genes and scores species/namespace correctness. Using the shipped path — not a
separate batch surface — means the gate measures exactly what production does (no surface-parity gap).

Per the expensive-run SOP, a timestamped artifact is **always written** (``--out`` is an override,
never the only way to save), with pinned reproduce-inputs (biomapper version, base_url, gold-set
SHA, timestamp). The API key is never written to the artifact.

Run:  cd backend && uv run python -m kestrel_backend.evals.biomapper_resolution.run_eval --env dev
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Awaitable, Callable

from ...biomapper_client import resolve_entity
from ...config import get_settings, resolve_biomapper_base_url
from ...graph.nodes.entity_resolution import reconcile_to_kestrel

_GOLD_PATH = Path(__file__).parent / "gold_set.json"
_RUNS_DIR = Path(__file__).parent / "runs"

# Below this biomapper hit-rate on a gene gold set, assume a throttled/degraded backend
# (HTTP 200 + empty matches) rather than real no-matches — fail loud, don't record as passes.
_DEGRADED_HIT_RATE = 0.3

ResolveFn = Callable[..., Awaitable[dict | None]]
ReconcileFn = Callable[..., Awaitable[tuple[str, str | None] | None]]


def load_gold(path: Path = _GOLD_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["entities"]


async def evaluate(
    gold_entities: list[dict[str, Any]],
    base_url: str | None,
    resolve_fn: ResolveFn = resolve_entity,
    reconcile_fn: ReconcileFn = reconcile_to_kestrel,
) -> list[dict[str, Any]]:
    """Run the production resolve+reconcile path over the gold set; return per-entity rows."""
    rows: list[dict[str, Any]] = []
    for ent in gold_entities:
        name, hint, expected = ent["name"], ent["entity_type_hint"], ent["expected_curie"]
        resolved: str | None = None
        method = "unresolved"
        r = await resolve_fn(name, hint, base_url=base_url)
        if r:
            rec = await reconcile_fn(r, hint)
            if rec:
                resolved, method = rec[0], "biomapper"
        rows.append({
            "name": name,
            "entity_type_hint": hint,
            "expected_curie": expected,
            "resolved_curie": resolved,
            "method": method,
            "correct": resolved == expected,
            "known_residual": ent.get("known_residual", False),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    resolved = sum(1 for r in rows if r["resolved_curie"] is not None)
    # Wrong-species/namespace: a resolved CURIE that doesn't match the human oracle.
    mismatched = sum(1 for r in rows if r["resolved_curie"] is not None and not r["correct"])
    residual = sum(1 for r in rows if r["known_residual"])
    hit_rate = (resolved / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "biomapper_hits": resolved,
        "biomapper_hit_rate": round(hit_rate, 4),
        "mismatched": mismatched,
        "known_residuals": residual,
        # Accuracy excluding the documented Kestrel-recall residuals (e.g. GH1).
        "accuracy_excl_residual": (
            round(correct / (total - residual), 4) if (total - residual) else 0.0
        ),
        "degraded_backend_suspected": hit_rate < _DEGRADED_HIT_RATE,
    }


def _gold_sha(path: Path = _GOLD_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_artifact(rows: list[dict[str, Any]], base_url: str | None, env: str, ts: str) -> dict[str, Any]:
    try:
        bm_version = version("biomapper")
    except PackageNotFoundError:
        bm_version = "unknown"
    return {
        "eval": "biomapper_resolution_gold_set",
        "timestamp": ts,
        "reproduce_inputs": {  # pinned so the result is reproducible
            "biomapper_version": bm_version,
            "biomapper_env": env,
            "biomapper_base_url": base_url,  # NOT the api key
            "gold_set_sha": _gold_sha(),
        },
        "summary": summarize(rows),
        "rows": rows,
    }


def default_artifact_path(ts: str) -> Path:
    return _RUNS_DIR / f"biomapper_resolution_{ts}.json"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Biomapper resolution gold-set eval (merge gate)")
    parser.add_argument("--env", default="dev", choices=["production", "dev"],
                        help="Which biomapper2 API to evaluate (default: dev — the HGNC fix)")
    parser.add_argument("--gold", type=str, default=None, help="Path to gold_set.json")
    parser.add_argument("--out", type=str, default=None,
                        help="Override artifact path (an artifact is ALWAYS written either way)")
    args = parser.parse_args()

    settings = get_settings()
    base_url = resolve_biomapper_base_url(args.env, settings)
    gold = load_gold(Path(args.gold)) if args.gold else load_gold()

    rows = await evaluate(gold, base_url)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = build_artifact(rows, base_url, args.env, ts)

    # Persist by default (SOP); --out only overrides the path.
    out_path = Path(args.out) if args.out else default_artifact_path(ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, default=str))

    s = artifact["summary"]
    print(json.dumps(s, indent=2))
    if s["degraded_backend_suspected"]:
        print("WARNING: low hit-rate — biomapper backend may be throttled/degraded; do NOT trust this run.")
    print(f"Artifact written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
