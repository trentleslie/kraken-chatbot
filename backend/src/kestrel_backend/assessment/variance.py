"""Variance computation and tolerance band generation for baseline assessment.

Computes per-metric statistics across multiple runs of the same query,
producing tolerance bands that structural checks use to detect regressions.

Tolerance bands are mean +/- 2 standard deviations per metric. Queries with
coefficient of variation (CV) > 0.5 on finding counts are flagged as 'warning'
rather than producing hard pass/fail boundaries.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Metrics extracted from pipeline output for variance analysis
STRUCTURAL_METRICS = [
    "resolved_entity_count",
    "direct_finding_count",
    "cold_start_finding_count",
    "hypothesis_count",
    "shared_neighbor_count",
    "bridge_count",
    "error_count",
]

# Pre-LLM deterministic fields — these should have zero variance across runs
# with the same KG responses (useful for refactor verification)
DETERMINISTIC_FIELDS = [
    "resolved_curies",  # Set of CURIEs from entity resolution
    "node_execution_set",  # Which nodes ran
]

CV_WARNING_THRESHOLD = 0.5


def extract_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """Extract assessment metrics from a serialized DiscoveryState.

    Returns a dict of metric_name -> value for variance analysis.
    """
    metrics = {}

    # Count-based metrics
    metrics["resolved_entity_count"] = len(state.get("resolved_entities", []))
    metrics["direct_finding_count"] = len(state.get("direct_findings", []))
    metrics["cold_start_finding_count"] = len(state.get("cold_start_findings", []))
    metrics["hypothesis_count"] = len(state.get("hypotheses", []))
    metrics["shared_neighbor_count"] = len(state.get("shared_neighbors", []))
    metrics["bridge_count"] = len(state.get("bridges", []))
    metrics["error_count"] = len(state.get("errors", []))

    # Deterministic fields (for refactor verification)
    resolved = state.get("resolved_entities", [])
    if resolved:
        curies = sorted(
            e.get("curie", "") if isinstance(e, dict) else ""
            for e in resolved
            if (e.get("curie") if isinstance(e, dict) else None)
        )
        metrics["resolved_curies"] = curies
    else:
        metrics["resolved_curies"] = []

    # Node execution detection (which nodes produced output)
    node_indicators = {
        "intake": "raw_entities",
        "entity_resolution": "resolved_entities",
        "triage": "novelty_scores",
        "direct_kg": "direct_findings",
        "cold_start": "cold_start_findings",
        "pathway_enrichment": "shared_neighbors",
        "integration": "bridges",
        "temporal": "temporal_classifications",
        "synthesis": "synthesis_report",
        "literature_grounding": "hypotheses",
    }
    executed = sorted(
        node for node, field in node_indicators.items()
        if state.get(field) is not None and state.get(field) != []
    )
    metrics["node_execution_set"] = executed

    return metrics


def compute_tolerance_bands(
    runs: list[dict[str, Any]],
    query_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute tolerance bands from multiple runs of the same query.

    Args:
        runs: List of metric dicts (one per run) from extract_metrics()
        query_metadata: Optional metadata about the query (path_type, etc.)

    Returns:
        Tolerance band dict with per-metric statistics and warning flags.
    """
    if not runs:
        return {"error": "No runs provided"}

    bands = {}
    warnings = []

    for metric in STRUCTURAL_METRICS:
        values = [r.get(metric, 0) for r in runs]
        n = len(values)

        if n < 2:
            bands[metric] = {
                "mean": values[0] if values else 0,
                "stddev": 0,
                "min": values[0] if values else 0,
                "max": values[0] if values else 0,
                "lower_bound": values[0] if values else 0,
                "upper_bound": values[0] if values else 0,
                "n": n,
                "cv": 0,
                "status": "insufficient_data",
            }
            continue

        mean = statistics.mean(values)
        stddev = statistics.stdev(values)
        cv = stddev / mean if mean > 0 else 0

        status = "ok"
        if cv > CV_WARNING_THRESHOLD:
            status = "warning"
            warnings.append(f"{metric}: CV={cv:.3f} exceeds threshold {CV_WARNING_THRESHOLD}")

        bands[metric] = {
            "mean": round(mean, 4),
            "stddev": round(stddev, 4),
            "min": min(values),
            "max": max(values),
            "lower_bound": round(max(0, mean - 2 * stddev), 4),
            "upper_bound": round(mean + 2 * stddev, 4),
            "n": n,
            "cv": round(cv, 4),
            "status": status,
        }

    # Deterministic fields — check for consistency
    for field in DETERMINISTIC_FIELDS:
        field_values = [json.dumps(r.get(field, []), sort_keys=True) for r in runs]
        unique = set(field_values)
        bands[field] = {
            "consistent": len(unique) == 1,
            "unique_values": len(unique),
            "canonical_value": json.loads(field_values[0]) if field_values else None,
        }
        if len(unique) > 1:
            warnings.append(f"{field}: expected consistent but found {len(unique)} unique values")

    result = {
        "metric_bands": bands,
        "warnings": warnings,
        "run_count": len(runs),
    }

    if query_metadata:
        result["query_metadata"] = query_metadata

    return result


def select_canonical_run(
    runs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Select the canonical run (median finding count) for Tier 2 scoring.

    Args:
        runs: List of metric dicts from extract_metrics()
        outputs: List of full pipeline output dicts (parallel to runs)

    Returns:
        (run_index, output_dict) of the canonical run
    """
    # Use total finding count as the selection metric
    finding_counts = [
        r.get("direct_finding_count", 0) + r.get("cold_start_finding_count", 0)
        for r in runs
    ]

    # Find the run closest to the median
    median_val = statistics.median(finding_counts)
    closest_idx = min(
        range(len(finding_counts)),
        key=lambda i: abs(finding_counts[i] - median_val),
    )

    return closest_idx, outputs[closest_idx]


def compute_all_tolerance_bands(
    output_dir: Path,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute tolerance bands for all queries from stored outputs.

    Reads pipeline outputs from output_dir, extracts metrics, computes
    tolerance bands, and returns the full tolerance band dataset.

    Args:
        output_dir: Directory containing pipeline output JSON files
        queries: List of query dicts with "query" key

    Returns:
        Dict mapping query_hash -> tolerance_band_data
    """
    from .capture import query_hash as qhash

    all_bands = {}

    for q in queries:
        query_text = q["query"]
        qh = qhash(query_text)

        # Find all output files for this query
        run_outputs = []
        run_metrics = []
        run_num = 1

        while True:
            output_path = output_dir / "outputs" / f"{qh}_run{run_num}.json"
            if not output_path.exists():
                break
            output_data = json.loads(output_path.read_text())
            if "error" not in output_data or output_data.get("error") is None:
                metrics = extract_metrics(output_data)
                run_metrics.append(metrics)
                run_outputs.append(output_data)
            else:
                logger.warning("Skipping failed run %d for query '%s'", run_num, query_text[:50])
            run_num += 1

        if not run_metrics:
            logger.warning("No successful runs for query '%s'", query_text[:50])
            continue

        bands = compute_tolerance_bands(run_metrics, query_metadata=q)

        # Select canonical run
        if run_outputs:
            canonical_idx, canonical_output = select_canonical_run(run_metrics, run_outputs)
            bands["canonical_run_index"] = canonical_idx + 1  # 1-indexed

            # Save canonical output separately
            canonical_path = output_dir / "canonical" / f"{qh}.json"
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_path.write_text(json.dumps(canonical_output, indent=2, default=str))

        all_bands[qh] = bands

    return all_bands
