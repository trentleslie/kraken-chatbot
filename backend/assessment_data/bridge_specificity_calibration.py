"""Axis C, Unit 4 — bridge-specificity CALIBRATION probe (live Kestrel).

Records the REAL KG degree of the 1-MNA scaffold nodes ({blood, placenta, cancer}) plus a spread
of specific/moderate control intermediates, derives the intermediate-degree distribution, and
recommends the label cut points (`SPECIFIC_CUT`, `MODERATE_CUT`) and the per-node generic cutoff
(`GENERIC_CUTOFF`) by QUANTILE — not round score values (with w=0.4, naive round-number score cut
points push every biologically real degree into the penalty side and leave `specific` empty).

Two outputs, BOTH saved by default (persist-expensive-run-artifacts SOP):
  1. a timestamped run artifact under assessment_data/bridge_specificity_calibration_runs/
     (recommended constants + full degree distribution + per-scaffold scores + pinning inputs);
  2. the retrodiction fixture tests/fixtures/bridge_specificity_1mna_degrees.json (real measured
     degrees), consumed by test_bridge_specificity_retrodiction.py.

The 1-MNA hypothesis ("TNFRSF10A -> 1-MNA via blood, placenta, cancer") INVERTED: its intermediates
are near-universal high-degree nodes. This probe is what makes "would 1-MNA have been flagged?" a
real answer (measured degrees) rather than an arithmetic tautology.

Usage: PYTHONPATH=. uv run python assessment_data/bridge_specificity_calibration.py
       [--out DIR] [--no-fixture]

Requires a reachable Kestrel (backend/.env loaded). The CURIEs below are best-effort canonical
identifiers — the live run should confirm/repair any that fail to resolve (a None degree is logged,
never fatal), and MONDO/UBERON drift should be corrected here before trusting the calibration.
"""

import argparse
import asyncio
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.kestrel_backend.graph.nodes.bridge_specificity import (  # noqa: E402
    DAMPING_W,
    GENERIC_CUTOFF,
    MODERATE_CUT,
    SPECIFIC_CUT,
    bridge_specificity,
    make_degree_provider,
)

# The 1-MNA scaffold — the near-universal hubs the inverted hypothesis leaned on.
ONE_MNA_SCAFFOLD = {
    "UBERON:0000178": "blood",
    "UBERON:0001987": "placenta",
    "MONDO:0004992": "cancer",
}

# A low-degree control expected to reach `specific`.
CONTROL_SPECIFIC = {
    "CHEBI:16797": "1-methylnicotinamide",
}

# A spread of intermediates to characterize the real degree distribution (specific -> generic).
# Extend with real scaffold CURIEs discovered from a representative module run before trusting
# the recommended cut points.
DISTRIBUTION_PROBE = {
    "CHEBI:15377": "water",
    "CHEBI:16236": "ethanol",
    "CHEBI:17234": "glucose",
    "CHEBI:29101": "sodium(1+)",
    "GO:0006954": "inflammatory response",
    "GO:0005515": "protein binding",
    "UBERON:0002107": "liver",
    "UBERON:0000955": "brain",
    "MONDO:0005148": "type 2 diabetes mellitus",
    "HP:0000118": "phenotypic abnormality",
}


def _query_sha(curies: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(curies)).encode()).hexdigest()[:16]


def _quantile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).parent / "bridge_specificity_calibration_runs"
    ap.add_argument("--out", type=Path, default=default_out,
                    help="Run-artifact directory (a timestamped JSON is written inside). "
                         "Overrides the default; saving is NOT opt-in.")
    ap.add_argument("--no-fixture", action="store_true",
                    help="Do not overwrite the retrodiction fixture (still saves the run artifact).")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_curies = {**ONE_MNA_SCAFFOLD, **CONTROL_SPECIFIC, **DISTRIBUTION_PROBE}
    query_sha = _query_sha(list(all_curies))

    # One bounded, per-run cached provider for the whole probe (mirrors the node's usage).
    provider = make_degree_provider(concurrency=8)

    degrees: dict[str, int | None] = {}
    for curie in all_curies:
        d = await provider(curie)
        degrees[curie] = d
        print(f"  {curie:20} {all_curies[curie]:32} degree={d}")

    # --- recommended constants by quantile of the observed (known) intermediate degrees --------
    known = sorted(d for d in degrees.values() if d is not None)
    recommended = {}
    if known:
        # per-node generic cutoff: ~90th percentile (the hubs); specific/moderate score cuts
        # derived from degree quantiles so a real fraction of nodes clears `specific`.
        gen_cut = _quantile(known, 0.90)
        p40 = max(_quantile(known, 0.40), 1.0)
        p70 = max(_quantile(known, 0.70), 1.0)
        recommended = {
            "GENERIC_CUTOFF": round(gen_cut),
            "SPECIFIC_CUT": round(p40 ** (-DAMPING_W), 4),   # score at the 40th-pct degree
            "MODERATE_CUT": round(p70 ** (-DAMPING_W), 4),   # score at the 70th-pct degree
            "basis": {"p40_degree": round(p40), "p70_degree": round(p70), "p90_degree": round(gen_cut)},
        }

    # --- score the reference scaffolds under the CURRENT constants (preview) --------------------
    def _score(names: dict[str, str]) -> dict:
        curies = list(names)
        ds = [degrees[c] for c in curies]
        spec = bridge_specificity(curies, ds)
        return {"curies": curies, "degrees": ds, "label": spec.label, "score": spec.score,
                "generic_intermediates": spec.generic_intermediates}

    one_mna_score = _score(ONE_MNA_SCAFFOLD)
    control_score = _score(CONTROL_SPECIFIC)

    artifact = {
        "timestamp": stamp,
        "query_sha": query_sha,
        "damping_w": DAMPING_W,
        "current_constants": {
            "SPECIFIC_CUT": SPECIFIC_CUT, "MODERATE_CUT": MODERATE_CUT,
            "GENERIC_CUTOFF": GENERIC_CUTOFF,
        },
        "recommended_constants": recommended,
        "degrees": degrees,
        "degree_distribution": {
            "n_known": len(known),
            "min": known[0] if known else None,
            "median": statistics.median(known) if known else None,
            "max": known[-1] if known else None,
        },
        "one_mna_scaffold_score": one_mna_score,
        "control_specific_score": control_score,
    }

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / f"{stamp}.json"
    art_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\ncurrent-constant 1-MNA label: {one_mna_score['label']} "
          f"(score={one_mna_score['score']})  control: {control_score['label']}")
    print(f"recommended constants: {recommended}")
    print(f"run artifact: {art_path}")

    if not args.no_fixture:
        fixture = {
            "provenance": {
                "source": "live Kestrel calibration run",
                "query_sha": query_sha,
                "recorded_at": stamp,
                "damping_w": DAMPING_W,
                "note": "1-MNA hypothesis: TNFRSF10A -> 1-MNA via blood, placenta, cancer. "
                        "Intermediates are near-universal high-degree nodes; the bridge INVERTED.",
            },
            "one_mna_scaffold": {
                c: {"name": ONE_MNA_SCAFFOLD[c], "degree": degrees[c]} for c in ONE_MNA_SCAFFOLD
            },
            "control_specific": {
                c: {"name": CONTROL_SPECIFIC[c], "degree": degrees[c]} for c in CONTROL_SPECIFIC
            },
            "recorded_intermediate_population": [d for d in degrees.values() if d is not None],
        }
        fx_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / \
            "bridge_specificity_1mna_degrees.json"
        fx_path.write_text(json.dumps(fixture, indent=2))
        print(f"retrodiction fixture updated: {fx_path}")


if __name__ == "__main__":
    asyncio.run(main())
