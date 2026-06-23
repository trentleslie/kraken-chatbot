"""Brown-module C1 PILOT: instrumented full-pipeline E2E run, biomapper-ON (dev).

Runs a ~24-analyte slice of the Brown WGCNA module through the REAL 12-node discovery
pipeline (intake -> resolution -> triage -> ... -> hypothesis_extraction -> bridge_grounding
-> literature_grounding -> synthesis), with biomapper resolution enabled against the dev
endpoint (the fixed resolver). Purpose: confirm synthesis survives the volume, measure
cost/latency (incl. the rate-limited literature_grounding), and produce honest per-node
coverage accounting + the actual synthesis report.

Guards (per the requirements-doc review):
  - biomapper is force-ENABLED (flag is default-off and NOT env-sourced) by mutating the
    get_pipeline_config() singleton; asserted before the run.
  - biomapper_env="dev" so it hits the FIXED resolver (prod/main is the legacy ablation baseline).
  - POST-RUN assert biomapper actually fired (resolved_entities method=="biomapper" count > 0),
    else a silent ValueError fallback would let a biomapper-OFF run masquerade as ON.
  - 20-min wall-clock ceiling: on breach, abort and SAVE the partial accumulated state.
  - Saves a timestamped artifact by default (expensive-run SOP), pinning inputs.

Paid: SDK calls (entity_resolution Tier-2, hypothesis_extraction, synthesis) + external
literature APIs (PubMed/OpenAlex/Exa/S2; S2 serialized ~10s/req). Scratch tooling.
Run: cd backend && uv run python -m assessment_data.brown_c1_pilot_e2e [--n-proteins 12 --n-metabolites 12]
"""

import argparse
import asyncio
import csv
import json
import operator
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, get_args, get_origin, get_type_hints

from dotenv import load_dotenv

load_dotenv()

# --- biomapper creds: dev endpoint + key (key from the biomapper2 repo .env; never hardcode) ---
if not os.environ.get("BIOMAPPER_API_KEY"):
    _bm_env = Path("../../biomapper2/.env")
    if _bm_env.exists():
        for _line in _bm_env.read_text().splitlines():
            if _line.startswith("BIOMAPPER2_API_KEY="):
                os.environ["BIOMAPPER_API_KEY"] = _line.split("=", 1)[1].strip()
                break
os.environ.setdefault("BIOMAPPER_DEV_BASE_URL", "https://dev-biomapper.expertintheloop.io/api/v1")

from kestrel_backend.graph.nodes.synthesis import assemble_synthesis_context  # noqa: E402
from kestrel_backend.graph.pipeline_config import get_pipeline_config  # noqa: E402
from kestrel_backend.graph.runner import stream_discovery  # noqa: E402
from kestrel_backend.graph.state import DiscoveryState  # noqa: E402

TSV = Path("../docs/data/Frailty_Multiomic_WGCNA-modules.tsv")
OUT_DIR = Path("assessment_data/brown_diagnostic_runs")

# Chars/token ratio measured on the 48-analyte CURIE-dense context (Diagnostic Evidence, 2026-06-22).
CHARS_PER_TOKEN = 3.5
TOKEN_BUDGET = 100_000  # assembled-context target, well under the ~200K-token model window


def _additive_list_keys() -> set[str]:
    """Keys declared ``Annotated[list[...], operator.add]`` in DiscoveryState.

    These accumulate across streamed node deltas (stream_mode="updates"), so a reducer-blind
    ``merged.update(out)`` clobbers all but the last node's slice — the ~110x coverage under-count
    (R6). Derived from the annotations so a future reducer field can't silently reintroduce the bug.
    """
    keys: set[str] = set()
    for name, ann in get_type_hints(DiscoveryState, include_extras=True).items():
        if get_origin(ann) is Annotated:
            base, *meta = get_args(ann)
            if operator.add in meta and get_origin(base) is list:
                keys.add(name)
    return keys


_ADDITIVE_KEYS = _additive_list_keys()


def merge_delta(merged: dict, out: dict) -> None:
    """Reducer-aware merge mirroring LangGraph semantics: concatenate operator.add list keys across
    deltas; overwrite plain last-write-wins keys. Replaces the under-counting ``merged.update(out)``."""
    for k, v in out.items():
        if k in _ADDITIVE_KEYS and isinstance(v, list):
            merged[k] = merged.get(k, []) + v
        else:
            merged[k] = v


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pick_pilot(n_prot, n_met):
    """All named Brown analytes (proteins + metabolites/chemistry).

    Phase 1 (intake analyte-name robustness): intake now keeps internal commas (the query is
    newline-delimited, one analyte per line) and strips alias parentheticals to a resolvable
    primary name, so the prior defensive "(" / "," exclusion is no longer needed. Rows with no
    name (14 Chemistry rows with empty ChemName) are inherently un-submittable and stay excluded.
    """
    prot, met = [], []
    with open(TSV, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["ModuleID"] != "Brown":
                continue
            if r["Dataset"] == "Protein" and r["GeneSymbol"] not in ("", "NA"):
                prot.append(r["GeneSymbol"])
            elif r["Dataset"] in ("Metabolite", "Chemistry") and r["ChemName"] not in ("", "NA"):
                met.append(r["ChemName"])
    return prot[:n_prot], met[:n_met]


def build_query(proteins, metabolites):
    # Newline-delimited sections (one analyte per line) so commas inside chemical names are safe;
    # intake's labeled-section parser treats each line as one analyte (Phase 1).
    return (
        "Analyze the biological relationships among these co-expressed analytes "
        "from the Brown WGCNA module.\n\n"
        "Proteins:\n" + "\n".join(proteins) + "\n\n"
        "Metabolites:\n" + "\n".join(metabolites) + "\n"
    )


def coverage(state):
    """Per-node analyzed-vs-dropped accounting from the (merged) final state."""
    re_ = state.get("resolved_entities", []) or []
    methods = Counter(getattr(e, "method", None) or (e.get("method") if isinstance(e, dict) else None)
                      for e in re_)
    bm = sum(v for k, v in methods.items() if k and str(k).startswith("biomapper"))
    # Phase 1 (R5): resolution-level collision measurement — how many resolved entities collapse
    # to a shared CURIE (e.g. isomers the KG has no distinct node for). This is the evidence that
    # decides whether the Phase-2 isomer-preservation/provenance machinery is worth building.
    curies = [getattr(e, "curie", None) or (e.get("curie") if isinstance(e, dict) else None) for e in re_]
    curies = [c for c in curies if c]
    distinct_curies = len(set(curies))
    hyps = state.get("hypotheses", []) or []
    lit_total = sum(len(getattr(h, "literature_support", None) or
                        (h.get("literature_support") if isinstance(h, dict) else []) or []) for h in hyps)
    report = state.get("synthesis_report", "") or ""
    return {
        "resolved_by_method": {str(k): v for k, v in methods.items()},
        "biomapper_confirmed": bm,
        "resolved_entities_count": len(re_),
        "distinct_curies": distinct_curies,
        "curie_collisions": len(curies) - distinct_curies,
        "triage_buckets": {
            "well_characterized": len(state.get("well_characterized_curies", []) or []),
            "moderate": len(state.get("moderate_curies", []) or []),
            "sparse": len(state.get("sparse_curies", []) or []),
            "cold_start": len(state.get("cold_start_curies", []) or []),
        },
        "direct_findings": len(state.get("direct_findings", []) or []),
        "cold_start_findings": len(state.get("cold_start_findings", []) or []),
        "cold_start_skipped_count": state.get("cold_start_skipped_count"),
        "biological_themes": len(state.get("biological_themes", []) or []),
        "bridges": len(state.get("bridges", []) or []),
        "grounded_bridges": len(state.get("grounded_bridges", []) or []),
        "hypotheses": len(hyps),
        "literature_support_total": lit_total,
        "synthesis_report_chars": len(report),
        "errors": len(state.get("errors", []) or []),
        "literature_errors": len(state.get("literature_errors", []) or []),
        "bridge_grounding_errors": len(state.get("bridge_grounding_errors", []) or []),
    }


def synthesis_duration(node_order):
    """Wall-seconds spent in the synthesis node (its completion minus the prior node's)."""
    for i, n in enumerate(node_order):
        if n["node"] == "synthesis":
            prev = node_order[i - 1]["t"] if i > 0 else 0.0
            return round(n["t"] - prev, 1)
    return None


def acceptance_checks(merged, cov, synth_dur):
    """U7 machine-checkable acceptance — reproducible despite LLM non-determinism.

    Reconstructs the assembled synthesis context from the final merged state to measure the real
    size that overflowed (it is internal to the node, not stored in the artifact). The
    signal-preservation diff vs the 24-analyte pilot is a SEPARATE human gate, not checked here.
    """
    report = merged.get("synthesis_report", "") or ""
    errors = merged.get("errors", []) or []
    # Match the fallback marker that synthesis.run() actually emits: both the empty-output and the
    # exception cases end with "... fell back to deterministic report" (the exception case also says
    # "LLM call failed"). The earlier "synthesis_degraded" string never matched any real marker, so
    # this gate was permanently green. The marker text is owned by synthesis.run(); keep in sync.
    degraded = [e for e in errors if "fell back to deterministic" in str(e) or "LLM call failed" in str(e)]
    try:
        ctx_chars = len(assemble_synthesis_context(merged))
    except Exception as e:  # noqa: BLE001
        ctx_chars = -1
        print(f"  (acceptance: could not reassemble context: {type(e).__name__}: {e})", flush=True)
    est_tokens = round(ctx_chars / CHARS_PER_TOKEN) if ctx_chars >= 0 else -1
    # A module narrative should have few/no raw per-entity "### CURIE" dump sections.
    raw_curie_sections = len(re.findall(r"(?m)^###\s+[A-Za-z][\w.]*:\S", report))
    total_findings = cov["direct_findings"] + cov["cold_start_findings"]
    return {
        "no_synthesis_degraded": (len(degraded) == 0, f"{len(degraded)} degraded entries"),
        "synthesis_ran_real_llm": (
            synth_dur is not None and synth_dur > 30,
            f"synthesis_duration={synth_dur}s (want >30s, not the ~3.5s fallback)",
        ),
        "context_under_token_budget": (
            0 <= est_tokens < TOKEN_BUDGET,
            f"~{est_tokens} est tokens / {ctx_chars} chars (budget {TOKEN_BUDGET} tokens)",
        ),
        "report_not_raw_dump": (
            raw_curie_sections < 10,
            f"{raw_curie_sections} raw '### CURIE' sections (want <10)",
        ),
        "findings_counted_plausibly": (
            total_findings > 37,
            f"{total_findings} findings (R6: ≫37 expected at module scale post-merge-fix)",
        ),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proteins", type=int, default=12)
    ap.add_argument("--n-metabolites", type=int, default=12)
    ap.add_argument("--ceiling-min", type=int, default=20,
                    help="wall-clock ceiling in minutes; on breach abort and save partial")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ceiling_seconds = args.ceiling_min * 60

    if not os.environ.get("BIOMAPPER_API_KEY"):
        raise SystemExit("BIOMAPPER_API_KEY not set / not found in ../../biomapper2/.env — cannot run biomapper-ON.")

    # Force-enable biomapper (default-off, not env-sourced) on the config singleton; assert it took.
    cfg = get_pipeline_config()
    cfg.entity_resolution.biomapper.enabled = True
    assert cfg.entity_resolution.biomapper.enabled is True, "failed to enable biomapper on the config singleton"

    proteins, metabolites = pick_pilot(args.n_proteins, args.n_metabolites)
    query = build_query(proteins, metabolites)
    submitted = proteins + metabolites

    # Intake-parse check: how many of the submitted names survive extract_entities (denominator honesty).
    from kestrel_backend.graph.nodes.intake import extract_entities  # noqa: PLC0415
    parsed = extract_entities(query)
    print(f"PILOT: {len(proteins)} proteins + {len(metabolites)} metabolites = {len(submitted)} submitted; "
          f"intake parsed {len(parsed)}; biomapper=dev enabled; ceiling={ceiling_seconds//60}min", flush=True)

    merged: dict = {}
    node_order = []
    t0 = time.monotonic()

    async def consume():
        async for ev in stream_discovery(query, biomapper_env="dev"):
            node, out = ev["node"], ev["node_output"]
            if isinstance(out, dict):
                merge_delta(merged, out)  # reducer-aware: concatenate additive keys (R6 fix)
            node_order.append({"node": node, "t": round(time.monotonic() - t0, 1)})
            print(f"  [{time.monotonic() - t0:6.1f}s] {node}", flush=True)

    status = "complete"
    try:
        await asyncio.wait_for(consume(), timeout=ceiling_seconds)
    except asyncio.TimeoutError:
        status = "timeout_aborted"
        print(f"  !! exceeded {ceiling_seconds}s ceiling — aborting, saving partial", flush=True)
    except Exception as e:  # noqa: BLE001
        status = f"error:{type(e).__name__}"
        print(f"  !! run error: {type(e).__name__}: {e} — saving partial", flush=True)

    wall = round(time.monotonic() - t0, 1)
    cov = coverage(merged)
    synth_dur = synthesis_duration(node_order)
    checks = acceptance_checks(merged, cov, synth_dur)

    artifact = {
        "_meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "phase": "C1-pilot", "module": "Brown", "status": status, "wall_seconds": wall,
            "ceiling_seconds": ceiling_seconds, "biomapper_env": "dev",
            "biomapper_enabled": True, "git_sha": git_sha(),
            "submitted": submitted, "n_submitted": len(submitted), "n_parsed": len(parsed),
            "nodes_reached": [n["node"] for n in node_order],
            "synthesis_duration_s": synth_dur,
        },
        "coverage": cov,
        "acceptance": {k: {"passed": p, "detail": d} for k, (p, d) in checks.items()},
        "node_timeline": node_order,
        "synthesis_report": merged.get("synthesis_report", ""),
    }
    out = Path(args.output) if args.output else OUT_DIR / f"brown_c1_pilot_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str))

    # Post-run guard: did biomapper actually fire? (only meaningful if resolution completed)
    if cov["resolved_by_method"]:
        bm_ok = cov["biomapper_confirmed"] > 0
        print(f"\n  biomapper fired: {bm_ok} (confirmed={cov['biomapper_confirmed']}, "
              f"methods={cov['resolved_by_method']})", flush=True)
        if not bm_ok:
            print("  !! WARNING: biomapper-ON requested but 0 entities resolved via biomapper — "
                  "run is effectively biomapper-OFF (check type-hint assignment / dev endpoint).", flush=True)

    print("\n=== U7 ACCEPTANCE (machine checks) ===")
    all_pass = all(p for p, _ in checks.values())
    for name, (p, d) in checks.items():
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {d}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'} "
          f"(+ signal-preservation diff vs the 24-analyte pilot is a separate human gate)")

    print(f"\n=== C1 PILOT [{status}] {wall}s ===")
    print(json.dumps({"_meta": artifact["_meta"], "coverage": cov,
                      "acceptance": artifact["acceptance"]}, indent=2, default=str))
    print(f"\n--- SYNTHESIS REPORT ({cov['synthesis_report_chars']} chars) ---\n")
    print(merged.get("synthesis_report", "(none — run did not reach synthesis)")[:6000])
    print(f"\nArtifact: {out}")


if __name__ == "__main__":
    asyncio.run(main())
