"""Assessment report generation.

Aggregates structural check results into a JSON report with per-query
and aggregate sections. Format is compatible with R10 (serializable, diffable)
and includes human_judgment placeholder fields for R13.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .checks import CheckResult, run_all_checks

logger = logging.getLogger(__name__)


class HypothesisJudgment(BaseModel):
    """Placeholder for expert-in-the-loop human judgment (R13)."""
    override_plausibility: int | None = None
    override_relevance: int | None = None
    override_novelty: int | None = None
    notes: str | None = None


class QueryReport(BaseModel):
    """Assessment report for a single query."""
    query: str
    query_hash: str
    path_type: str | None = None
    checks: list[CheckResult]
    passed: bool = Field(..., description="True if all checks passed (warnings count as pass)")
    warnings: int = Field(0, description="Number of checks with warning status")
    human_judgment: list[HypothesisJudgment] = Field(
        default_factory=list,
        description="Per-hypothesis human judgment placeholders (initially empty)",
    )


class AssessmentReport(BaseModel):
    """Full assessment report across all queries."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "v1"
    mode: str = "replay"
    query_reports: list[QueryReport] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)


def generate_report(
    assessment_results: dict[str, Any],
    tolerance_bands: dict[str, Any] | None = None,
) -> AssessmentReport:
    """Generate an assessment report from runner results and tolerance bands.

    Args:
        assessment_results: Output from run_assessment()
        tolerance_bands: Tolerance band data keyed by query_hash

    Returns:
        AssessmentReport with per-query checks and aggregate summary.
    """
    tolerance_bands = tolerance_bands or {}
    query_reports = []

    total_passed = 0
    total_failed = 0
    total_warnings = 0
    total_checks = 0

    for result in assessment_results.get("results", []):
        state = result.get("state")
        if state is None:
            # Skipped or failed query
            continue

        qhash = result["query_hash"]
        metadata = result.get("metadata", {})
        baseline = tolerance_bands.get(qhash)

        # Run structural checks
        checks = run_all_checks(
            state=state,
            baseline_stats=baseline,
            expected_curies=metadata.get("expected_curies"),
        )

        # Compute pass/fail/warning counts
        check_passed = all(c.passed for c in checks)
        warnings = sum(1 for c in checks if c.status == "warning")

        # Create hypothesis judgment placeholders
        hypotheses = state.get("hypotheses", [])
        judgments = [HypothesisJudgment() for _ in hypotheses]

        query_reports.append(QueryReport(
            query=result["query"],
            query_hash=qhash,
            path_type=metadata.get("path_type"),
            checks=checks,
            passed=check_passed,
            warnings=warnings,
            human_judgment=judgments,
        ))

        total_checks += len(checks)
        if check_passed:
            total_passed += 1
        else:
            total_failed += 1
        total_warnings += warnings

    report = AssessmentReport(
        mode=assessment_results.get("summary", {}).get("mode", "unknown"),
        query_reports=query_reports,
        aggregate={
            "total_queries": len(query_reports),
            "passed": total_passed,
            "failed": total_failed,
            "total_warnings": total_warnings,
            "total_checks": total_checks,
            "pass_rate": round(total_passed / len(query_reports), 2) if query_reports else 0,
        },
    )

    return report


def save_report(report: AssessmentReport, path: Path) -> None:
    """Save assessment report to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2))
    logger.info("Assessment report saved to %s", path)
