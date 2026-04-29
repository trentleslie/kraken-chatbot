"""Structural checks for pipeline assessment.

Implements deterministic checks using tolerance bands calibrated from baseline
variance data. Each check returns a CheckResult with pass/fail/warning status.

Checks:
- Pipeline completion: all expected nodes executed
- Schema conformance: output matches DiscoveryState field types
- Entity resolution recall: resolved CURIEs contain expected set
- Finding count stability: counts within tolerance band (CV > 0.5 = warning)
- Hypothesis completeness: required fields present on each hypothesis
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CV_WARNING_THRESHOLD = 0.5


class CheckResult(BaseModel):
    """Result of a single structural check."""
    name: str = Field(..., description="Check name")
    passed: bool = Field(..., description="Whether the check passed")
    status: str = Field(..., description="pass, fail, or warning")
    metric: str = Field(..., description="What was measured")
    expected: Any = Field(None, description="Expected value or range")
    actual: Any = Field(None, description="Actual observed value")
    tolerance: Any = Field(None, description="Tolerance band if applicable")
    message: str = Field("", description="Human-readable explanation")


def check_pipeline_completion(state: dict[str, Any]) -> CheckResult:
    """Check that all expected pipeline nodes executed.

    Verifies presence of node output fields in the state dict.
    """
    node_indicators = {
        "intake": "raw_entities",
        "entity_resolution": "resolved_entities",
        "triage": "novelty_scores",
        "synthesis": "synthesis_report",
    }

    missing = []
    for node, field in node_indicators.items():
        value = state.get(field)
        if value is None or (isinstance(value, (list, str)) and len(value) == 0):
            missing.append(f"{node} ({field})")

    passed = len(missing) == 0
    return CheckResult(
        name="pipeline_completion",
        passed=passed,
        status="pass" if passed else "fail",
        metric="nodes_executed",
        expected=list(node_indicators.keys()),
        actual=[n for n in node_indicators if n not in [m.split(" ")[0] for m in missing]],
        message=f"Missing nodes: {', '.join(missing)}" if missing else "All core nodes executed",
    )


def check_schema_conformance(state: dict[str, Any]) -> CheckResult:
    """Check that key output fields have the correct types."""
    type_checks = {
        "resolved_entities": list,
        "novelty_scores": list,
        "synthesis_report": str,
        "hypotheses": list,
        "errors": list,
    }

    violations = []
    for field, expected_type in type_checks.items():
        value = state.get(field)
        if value is not None and not isinstance(value, expected_type):
            violations.append(f"{field}: expected {expected_type.__name__}, got {type(value).__name__}")

    passed = len(violations) == 0
    return CheckResult(
        name="schema_conformance",
        passed=passed,
        status="pass" if passed else "fail",
        metric="type_violations",
        expected=0,
        actual=len(violations),
        message="; ".join(violations) if violations else "All fields conform to expected types",
    )


def check_entity_resolution_recall(
    state: dict[str, Any],
    expected_curies: list[str] | None = None,
) -> CheckResult:
    """Check that resolved CURIEs contain the expected set (recall, not exact match).

    Extra resolved CURIEs beyond expected are acceptable.
    """
    if not expected_curies:
        return CheckResult(
            name="entity_resolution_recall",
            passed=True,
            status="pass",
            metric="recall",
            expected="no expected CURIEs specified",
            actual="skipped",
            message="No expected CURIEs to check — skipping",
        )

    resolved = state.get("resolved_entities", [])
    resolved_curies = set()
    for entity in resolved:
        curie = entity.get("curie") if isinstance(entity, dict) else getattr(entity, "curie", None)
        if curie:
            resolved_curies.add(curie)

    expected_set = set(expected_curies)
    found = expected_set & resolved_curies
    missing = expected_set - resolved_curies

    recall = len(found) / len(expected_set) if expected_set else 1.0
    passed = len(missing) == 0

    return CheckResult(
        name="entity_resolution_recall",
        passed=passed,
        status="pass" if passed else "fail",
        metric="recall",
        expected=sorted(expected_set),
        actual=sorted(resolved_curies),
        tolerance=None,
        message=f"Recall: {recall:.2f}. Missing: {sorted(missing)}" if missing
        else f"All {len(expected_set)} expected CURIEs found",
    )


def check_finding_count_stability(
    state: dict[str, Any],
    baseline_stats: dict[str, Any] | None = None,
) -> CheckResult:
    """Check finding counts are within tolerance bands from baseline variance.

    Queries with CV > 0.5 produce 'warning' status instead of 'fail'.
    """
    if baseline_stats is None:
        return CheckResult(
            name="finding_count_stability",
            passed=True,
            status="pass",
            metric="finding_counts",
            expected="no baseline stats",
            actual="skipped",
            message="No baseline stats provided — skipping",
        )

    direct = len(state.get("direct_findings", []))
    cold = len(state.get("cold_start_findings", []))
    total = direct + cold

    # Check against tolerance band
    metric_band = baseline_stats.get("metric_bands", {}).get("direct_finding_count", {})
    if not metric_band or metric_band.get("status") == "insufficient_data":
        return CheckResult(
            name="finding_count_stability",
            passed=True,
            status="warning",
            metric="finding_counts",
            expected="insufficient baseline data",
            actual={"direct": direct, "cold_start": cold, "total": total},
            message="Insufficient baseline data for comparison",
        )

    lower = metric_band.get("lower_bound", 0)
    upper = metric_band.get("upper_bound", float("inf"))
    cv = metric_band.get("cv", 0)
    is_high_variance = cv > _CV_WARNING_THRESHOLD

    in_band = lower <= direct <= upper

    if in_band:
        status = "pass"
        passed = True
    elif is_high_variance:
        # High-variance query — produce warning, not fail
        status = "warning"
        passed = True
        logger.warning(
            "Finding count %d outside band [%.1f, %.1f] but CV=%.3f > %.1f threshold — warning only",
            direct, lower, upper, cv, _CV_WARNING_THRESHOLD,
        )
    else:
        status = "fail"
        passed = False

    return CheckResult(
        name="finding_count_stability",
        passed=passed,
        status=status,
        metric="direct_finding_count",
        expected={"lower": lower, "upper": upper},
        actual=direct,
        tolerance={"cv": cv, "high_variance": is_high_variance},
        message=f"Direct findings: {direct} (band: [{lower:.1f}, {upper:.1f}], CV={cv:.3f})",
    )


def check_hypothesis_completeness(state: dict[str, Any]) -> CheckResult:
    """Check that each hypothesis has the required fields."""
    hypotheses = state.get("hypotheses", [])

    if not hypotheses:
        return CheckResult(
            name="hypothesis_completeness",
            passed=True,
            status="pass",
            metric="hypothesis_fields",
            expected="hypotheses present",
            actual=0,
            message="No hypotheses to check",
        )

    required_fields = ["claim", "evidence", "novelty_score"]
    incomplete = []

    for i, h in enumerate(hypotheses):
        if isinstance(h, dict):
            missing = [f for f in required_fields if f not in h or h[f] is None]
        else:
            missing = [f for f in required_fields if not hasattr(h, f) or getattr(h, f) is None]

        if missing:
            incomplete.append(f"hypothesis[{i}] missing: {', '.join(missing)}")

    passed = len(incomplete) == 0
    return CheckResult(
        name="hypothesis_completeness",
        passed=passed,
        status="pass" if passed else "fail",
        metric="incomplete_hypotheses",
        expected=0,
        actual=len(incomplete),
        message="; ".join(incomplete[:5]) if incomplete else f"All {len(hypotheses)} hypotheses complete",
    )


def run_all_checks(
    state: dict[str, Any],
    baseline_stats: dict[str, Any] | None = None,
    expected_curies: list[str] | None = None,
) -> list[CheckResult]:
    """Run all structural checks and return results."""
    return [
        check_pipeline_completion(state),
        check_schema_conformance(state),
        check_entity_resolution_recall(state, expected_curies),
        check_finding_count_stability(state, baseline_stats),
        check_hypothesis_completeness(state),
    ]
