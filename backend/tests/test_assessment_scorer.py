"""Tests for LLM-as-judge quality scorer."""

import json

import pytest

from kestrel_backend.assessment.scorer import (
    HypothesisScore,
    ScorerResult,
    _build_scoring_context,
    _parse_scores,
    compute_stability,
    score_hypotheses,
)


SAMPLE_HYPOTHESES = [
    {
        "claim": "NAD+ decline drives cellular aging",
        "tier": 1,
        "confidence": "high",
        "supporting_entities": ["CHEBI:15422", "HP:0011462"],
        "structural_logic": "NAD+ levels decrease with age; sirtuins require NAD+",
    },
    {
        "claim": "NMN supplementation restores NAD+ via salvage pathway",
        "tier": 2,
        "confidence": "moderate",
        "supporting_entities": ["CHEBI:85990"],
        "structural_logic": "NMN is a direct precursor to NAD+ in the salvage pathway",
    },
]


class TestBuildScoringContext:
    def test_includes_query(self):
        ctx = _build_scoring_context("NAD+ and aging", SAMPLE_HYPOTHESES)
        assert "NAD+ and aging" in ctx

    def test_includes_all_hypotheses(self):
        ctx = _build_scoring_context("test", SAMPLE_HYPOTHESES)
        assert "Hypothesis 0" in ctx
        assert "Hypothesis 1" in ctx
        assert "NAD+ decline" in ctx
        assert "NMN supplementation" in ctx

    def test_includes_supporting_entities(self):
        ctx = _build_scoring_context("test", SAMPLE_HYPOTHESES)
        assert "CHEBI:15422" in ctx


class TestParseScores:
    def test_valid_json_array(self):
        response = json.dumps([
            {"hypothesis_index": 0, "plausibility": 8, "relevance": 9, "novelty": 6, "rationale": "Strong"},
            {"hypothesis_index": 1, "plausibility": 7, "relevance": 7, "novelty": 4, "rationale": "Moderate"},
        ])
        scores = _parse_scores(response, 2)
        assert len(scores) == 2
        assert scores[0].plausibility == 8
        assert scores[1].novelty == 4

    def test_scores_clamped_to_range(self):
        response = json.dumps([
            {"hypothesis_index": 0, "plausibility": 15, "relevance": 0, "novelty": -1},
        ])
        scores = _parse_scores(response, 1)
        assert scores[0].plausibility == 10  # clamped to max
        assert scores[0].relevance == 1  # clamped to min
        assert scores[0].novelty == 1  # clamped to min

    def test_markdown_code_block(self):
        response = '```json\n[{"hypothesis_index": 0, "plausibility": 7, "relevance": 8, "novelty": 5}]\n```'
        scores = _parse_scores(response, 1)
        assert scores[0].plausibility == 7

    def test_non_json_response(self):
        scores = _parse_scores("This is not valid JSON at all", 2)
        assert len(scores) == 2
        assert all(s.error is not None for s in scores)
        assert all(s.plausibility == 5 for s in scores)  # default scores

    def test_fewer_scores_than_hypotheses(self):
        response = json.dumps([
            {"hypothesis_index": 0, "plausibility": 8, "relevance": 9, "novelty": 6},
        ])
        scores = _parse_scores(response, 3)
        assert len(scores) == 3
        assert scores[0].plausibility == 8
        assert scores[1].error is not None  # missing from response
        assert scores[2].error is not None

    def test_empty_hypotheses(self):
        scores = _parse_scores("[]", 0)
        assert len(scores) == 0


class TestScoreHypotheses:
    async def test_empty_hypotheses_returns_empty(self):
        result = await score_hypotheses("test query", [])
        assert result.scores == []
        assert result.error is None

    async def test_sdk_unavailable_returns_defaults(self):
        from unittest.mock import patch
        with patch("kestrel_backend.graph.sdk_utils.HAS_SDK", False):
            result = await score_hypotheses("test", SAMPLE_HYPOTHESES)

        assert len(result.scores) == 2
        assert all(s.plausibility == 5 for s in result.scores)
        assert result.error == "SDK unavailable"


class TestComputeStability:
    def test_stable_scores(self):
        """Identical scores across runs should produce perfect correlation."""
        base_scores = [
            HypothesisScore(hypothesis_index=0, plausibility=8, relevance=9, novelty=6),
            HypothesisScore(hypothesis_index=1, plausibility=5, relevance=7, novelty=3),
            HypothesisScore(hypothesis_index=2, plausibility=9, relevance=4, novelty=8),
        ]
        # 5 identical runs
        run_scores = [base_scores] * 5
        result = compute_stability(run_scores)

        assert result["meets_threshold"]
        assert result["overall_mean"] == 1.0
        assert result["n_runs"] == 5

    def test_unstable_scores(self):
        """Reversed rankings should produce negative correlation."""
        run1 = [
            HypothesisScore(hypothesis_index=0, plausibility=10, relevance=5, novelty=5),
            HypothesisScore(hypothesis_index=1, plausibility=1, relevance=5, novelty=5),
        ]
        run2 = [
            HypothesisScore(hypothesis_index=0, plausibility=1, relevance=5, novelty=5),
            HypothesisScore(hypothesis_index=1, plausibility=10, relevance=5, novelty=5),
        ]
        result = compute_stability([run1, run2])

        plaus = result["per_dimension"]["plausibility"]
        assert plaus["mean_pairwise_spearman"] is not None
        assert plaus["mean_pairwise_spearman"] < 0  # negative correlation

    def test_single_run_error(self):
        result = compute_stability([[]])
        assert "error" in result

    def test_degenerate_scores_handled(self):
        """All-same scores should be flagged as degenerate."""
        run1 = [
            HypothesisScore(hypothesis_index=0, plausibility=5, relevance=5, novelty=5),
            HypothesisScore(hypothesis_index=1, plausibility=5, relevance=5, novelty=5),
        ]
        result = compute_stability([run1, run1])

        for dim in ["plausibility", "relevance", "novelty"]:
            assert result["per_dimension"][dim]["n_degenerate"] > 0

    def test_mixed_stability(self):
        """Some dimensions stable, others not."""
        run1 = [
            HypothesisScore(hypothesis_index=0, plausibility=8, relevance=9, novelty=3),
            HypothesisScore(hypothesis_index=1, plausibility=5, relevance=7, novelty=8),
            HypothesisScore(hypothesis_index=2, plausibility=9, relevance=4, novelty=5),
        ]
        # Same plausibility ranking, different novelty
        run2 = [
            HypothesisScore(hypothesis_index=0, plausibility=8, relevance=9, novelty=7),
            HypothesisScore(hypothesis_index=1, plausibility=5, relevance=7, novelty=2),
            HypothesisScore(hypothesis_index=2, plausibility=9, relevance=4, novelty=9),
        ]
        result = compute_stability([run1, run2])

        # Plausibility and relevance should be perfectly correlated (same rankings)
        assert result["per_dimension"]["plausibility"]["mean_pairwise_spearman"] == 1.0
        assert result["per_dimension"]["relevance"]["mean_pairwise_spearman"] == 1.0
