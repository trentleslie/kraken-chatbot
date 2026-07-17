"""Axis E Unit 4: post-LLM Direction stamp (authoritative deterministic value).

Plan: docs/plans/2026-07-16-001-feat-tier3-falsifier-direction-awareness-plan.md

The stamp is what makes the direction value synthesis-owned rather than an LLM prompt-hope: whatever
arrow the LLM wrote (or omitted) is overwritten by the deterministic value, matched by the
hypothesis-title anchor. Seam-absent hypotheses get no line. A missing anchor or a raised stamp is
non-fatal — other blocks are still stamped and the report is never blanked.
"""

from kestrel_backend.graph.nodes.synthesis import DirectionResult, stamp_directions


def _up(conf="high"):
    return DirectionResult("up", conf, computable=True, seam_present=True)


def _down(conf="high"):
    return DirectionResult("down", conf, computable=True, seam_present=True)


def _indeterminate():
    return DirectionResult("indeterminate", None, computable=False, seam_present=True)


def _seam_absent():
    return DirectionResult("indeterminate", None, computable=False, seam_present=False)


# Execution note (plan): start from a report where the LLM wrote the WRONG arrow.
def test_stamp_overwrites_wrong_llm_arrow():
    report = (
        "#### MyPred\n"
        "**Prediction:** X drives Y\n"
        "**Direction:** ↑ (confidence: low)\n"
        "**Logic:** ...\n"
    )
    out = stamp_directions(report, {"MyPred": _down("high")})
    assert "**Direction:** ↓ (confidence: high)" in out
    assert "↑" not in out  # the wrong arrow is gone


def test_stamp_inserts_when_llm_omitted_line():
    report = "#### MyPred\n**Prediction:** X drives Y\n**Logic:** ...\n"
    out = stamp_directions(report, {"MyPred": _up("moderate")})
    lines = out.split("\n")
    assert "**Direction:** ↑ (confidence: moderate)" in lines
    # inserted immediately after the anchor heading
    assert lines[lines.index("#### MyPred") + 1].startswith("**Direction:**")


def test_stamp_renders_indeterminate():
    report = "#### MyPred\n**Prediction:** X\n"
    out = stamp_directions(report, {"MyPred": _indeterminate()})
    assert "**Direction:** indeterminate" in out


def test_seam_absent_inserts_no_line():
    report = "#### MyPred\n**Prediction:** X\n"
    out = stamp_directions(report, {"MyPred": _seam_absent()})
    assert "Direction:" not in out
    assert out == report  # untouched


def test_missing_anchor_still_stamps_other_blocks():
    report = "#### RealPred\n**Prediction:** X\n"
    out = stamp_directions(report, {"Ghost": _up(), "RealPred": _down("moderate")})
    assert "**Direction:** ↓ (confidence: moderate)" in out  # RealPred stamped despite Ghost missing


def test_two_blocks_stamped_independently():
    report = (
        "#### PredA\n**Prediction:** A\n**Direction:** ↑ (confidence: low)\n"
        "#### PredB\n**Prediction:** B\n"
    )
    out = stamp_directions(report, {"PredA": _down("high"), "PredB": _up("moderate")})
    assert "**Direction:** ↓ (confidence: high)" in out   # PredA overwritten
    assert "**Direction:** ↑ (confidence: moderate)" in out  # PredB inserted
    # PredB's inserted line stays inside PredB's block, not PredA's
    lines = out.split("\n")
    assert lines[lines.index("#### PredB") + 1].startswith("**Direction:** ↑")


def test_empty_report_or_map_is_noop():
    assert stamp_directions("", {"X": _up()}) == ""
    assert stamp_directions("some report", {}) == "some report"


def test_anchor_prefers_heading_over_summary_mention():
    # title appears first in a prose sentence, then as a heading — stamp the heading block
    report = (
        "## Executive Summary\n"
        "We highlight MyPred as the leading signal.\n"
        "#### MyPred\n"
        "**Prediction:** X\n"
    )
    out = stamp_directions(report, {"MyPred": _up("high")})
    lines = out.split("\n")
    # the Direction line sits under the heading, not injected into the summary
    assert lines[lines.index("#### MyPred") + 1] == "**Direction:** ↑ (confidence: high)"
    assert "leading signal." in out
