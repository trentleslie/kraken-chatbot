"""LLM-as-judge quality scorer for pipeline hypotheses.

Scores frozen pipeline outputs on three dimensions:
- Plausibility (1-10): biological plausibility of the hypothesis
- Relevance (1-10): relevance to the original query intent
- Novelty (1-10): non-obviousness given the input

Operates on frozen outputs (one selected baseline run per query) to isolate
scorer variance from pipeline variance. Uses temperature=0 and fixed prompts
for stability. Stability measured as per-dimension pairwise Spearman correlation
across 5 runs (target: mean >= 0.80).

Fallback: if Spearman is degenerate due to ties, reports Krippendorff's alpha.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Scoring scale anchors for LLM judge prompt
JUDGE_PROMPT = """You are an expert biomedical research evaluator. Score each hypothesis on three dimensions using a 1-10 integer scale.

## Scoring Anchors

### Plausibility (biological plausibility)
- 1-2: No biological basis; contradicts established knowledge
- 3-4: Weak basis; speculative with minimal mechanistic support
- 5-6: Moderate basis; plausible mechanism but incomplete evidence
- 7-8: Strong basis; well-supported by known biology
- 9-10: Very strong; aligns with established pathways and multiple evidence lines

### Relevance (to original query intent)
- 1-2: Unrelated to the query
- 3-4: Tangentially related; addresses adjacent topic
- 5-6: Partially relevant; addresses part of the query
- 7-8: Directly relevant; addresses the core question
- 9-10: Highly relevant; provides direct, actionable insight for the query

### Novelty (non-obviousness given input)
- 1-2: Trivially obvious; restates input or common knowledge
- 3-4: Slightly beyond obvious; minor extension of input
- 5-6: Moderately novel; connects concepts in a non-trivial way
- 7-8: Novel; reveals non-obvious connections or mechanisms
- 9-10: Highly novel; identifies surprising or counterintuitive relationships

## Instructions

For each hypothesis, return a JSON object with your scores and a brief rationale.
Always use the full 1-10 range. Avoid anchoring on 5 — differentiate clearly between hypotheses.

Return ONLY a valid JSON array of objects, one per hypothesis:
[
  {
    "hypothesis_index": 0,
    "plausibility": <1-10>,
    "relevance": <1-10>,
    "novelty": <1-10>,
    "rationale": "<brief explanation of scores>"
  }
]
"""


class HypothesisScore(BaseModel):
    """Score for a single hypothesis."""
    hypothesis_index: int
    plausibility: int = Field(..., ge=1, le=10)
    relevance: int = Field(..., ge=1, le=10)
    novelty: int = Field(..., ge=1, le=10)
    rationale: str = ""
    error: str | None = None


class ScorerResult(BaseModel):
    """Result of scoring a full query's hypotheses."""
    query: str
    scores: list[HypothesisScore]
    raw_response: str | None = None
    error: str | None = None


def _build_scoring_context(
    query: str,
    hypotheses: list[dict[str, Any]],
) -> str:
    """Build the context string for the judge prompt."""
    lines = [f"## Original Query\n{query}\n"]
    lines.append(f"## Hypotheses to Score ({len(hypotheses)} total)\n")

    for i, h in enumerate(hypotheses):
        claim = h.get("claim", h.get("title", f"Hypothesis {i}"))
        tier = h.get("tier", "unknown")
        logic = h.get("structural_logic", h.get("evidence", ""))
        entities = h.get("supporting_entities", [])
        confidence = h.get("confidence", "unknown")

        lines.append(f"### Hypothesis {i}")
        lines.append(f"**Claim:** {claim}")
        lines.append(f"**Tier:** {tier} | **Confidence:** {confidence}")
        lines.append(f"**Supporting entities:** {', '.join(str(e) for e in entities)}")
        lines.append(f"**Reasoning:** {logic}")
        lines.append("")

    return "\n".join(lines)


def _parse_scores(
    raw_response: str,
    num_hypotheses: int,
) -> list[HypothesisScore]:
    """Parse LLM judge response into HypothesisScore objects."""
    # Try to extract JSON from the response
    text = raw_response.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse judge response as JSON: %s", text[:200])
        return [
            HypothesisScore(
                hypothesis_index=i,
                plausibility=5, relevance=5, novelty=5,
                error="Failed to parse LLM response",
            )
            for i in range(num_hypotheses)
        ]

    if not isinstance(data, list):
        data = [data]

    scores = []
    for i in range(num_hypotheses):
        if i < len(data):
            entry = data[i]
            scores.append(HypothesisScore(
                hypothesis_index=i,
                plausibility=max(1, min(10, entry.get("plausibility", 5))),
                relevance=max(1, min(10, entry.get("relevance", 5))),
                novelty=max(1, min(10, entry.get("novelty", 5))),
                rationale=entry.get("rationale", ""),
            ))
        else:
            scores.append(HypothesisScore(
                hypothesis_index=i,
                plausibility=5, relevance=5, novelty=5,
                error="No score returned by judge for this hypothesis",
            ))

    return scores


async def score_hypotheses(
    query: str,
    hypotheses: list[dict[str, Any]],
) -> ScorerResult:
    """Score a set of hypotheses using the LLM judge.

    Args:
        query: The original query text
        hypotheses: List of hypothesis dicts (from serialized pipeline output)

    Returns:
        ScorerResult with per-hypothesis scores.
    """
    if not hypotheses:
        return ScorerResult(query=query, scores=[], error=None)

    from ..graph.sdk_utils import HAS_SDK, query as sdk_query, ClaudeAgentOptions

    if not HAS_SDK:
        return ScorerResult(
            query=query,
            scores=[
                HypothesisScore(
                    hypothesis_index=i,
                    plausibility=5, relevance=5, novelty=5,
                    error="SDK unavailable — default scores assigned",
                )
                for i in range(len(hypotheses))
            ],
            error="SDK unavailable",
        )

    context = _build_scoring_context(query, hypotheses)
    full_prompt = f"{context}\n\nScore each hypothesis above."

    # Build agent options — use temperature=0 if supported
    options_kwargs = {
        "system_prompt": JUDGE_PROMPT,
        "allowed_tools": [],
        "max_turns": 1,
        "permission_mode": "bypassPermissions",
    }

    options = ClaudeAgentOptions(**options_kwargs)

    # Collect LLM response
    raw_text = ""
    try:
        async for event in sdk_query(prompt=full_prompt, options=options):
            if hasattr(event, "text"):
                raw_text += event.text
            elif isinstance(event, dict) and "text" in event:
                raw_text += event["text"]
    except Exception as e:
        logger.error("Judge scoring failed: %s", e)
        return ScorerResult(
            query=query,
            scores=[
                HypothesisScore(
                    hypothesis_index=i,
                    plausibility=5, relevance=5, novelty=5,
                    error=f"Judge call failed: {e}",
                )
                for i in range(len(hypotheses))
            ],
            raw_response=None,
            error=str(e),
        )

    scores = _parse_scores(raw_text, len(hypotheses))

    return ScorerResult(
        query=query,
        scores=scores,
        raw_response=raw_text,
    )


def compute_stability(
    run_scores: list[list[HypothesisScore]],
) -> dict[str, Any]:
    """Compute per-dimension pairwise Spearman correlation across scorer runs.

    Args:
        run_scores: List of score lists, one per scorer run (all on same frozen outputs)

    Returns:
        Dict with per-dimension mean pairwise Spearman, plus Krippendorff's alpha fallback.
    """
    if len(run_scores) < 2:
        return {"error": "Need at least 2 runs for stability computation"}

    from itertools import combinations

    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {"error": "scipy not installed — cannot compute Spearman correlation"}

    dimensions = ["plausibility", "relevance", "novelty"]
    results = {}

    for dim in dimensions:
        # Extract score vectors per run
        vectors = []
        for run in run_scores:
            vectors.append([getattr(s, dim) for s in run])

        # Compute pairwise Spearman
        correlations = []
        for v1, v2 in combinations(vectors, 2):
            if len(set(v1)) <= 1 or len(set(v2)) <= 1:
                # Degenerate case — all same scores
                correlations.append(None)
            else:
                rho, _ = spearmanr(v1, v2)
                correlations.append(rho)

        valid = [c for c in correlations if c is not None]
        results[dim] = {
            "mean_pairwise_spearman": round(sum(valid) / len(valid), 4) if valid else None,
            "n_pairs": len(correlations),
            "n_valid": len(valid),
            "n_degenerate": len(correlations) - len(valid),
            "all_correlations": [round(c, 4) if c is not None else None for c in correlations],
        }

    # Overall stability assessment
    means = [r["mean_pairwise_spearman"] for r in results.values() if r["mean_pairwise_spearman"] is not None]
    overall_stable = all(m >= 0.80 for m in means) if means else False

    return {
        "per_dimension": results,
        "overall_mean": round(sum(means) / len(means), 4) if means else None,
        "meets_threshold": overall_stable,
        "threshold": 0.80,
        "n_runs": len(run_scores),
    }
