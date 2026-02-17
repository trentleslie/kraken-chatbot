"""
Literature Grounding Node: Add verified citations to synthesis hypotheses.

Uses Semantic Scholar API to find relevant papers for each hypothesis,
score relevance, and attach as literature support. This provides evidence
backing for generated hypotheses.

Position in pipeline: Runs after synthesis, before final output.
"""

import asyncio
import logging
import time
from typing import Any

from ..state import DiscoveryState, Hypothesis, LiteratureSupport
from ...semantic_scholar import (
    search_papers, score_relevance, classify_relationship,
    extract_key_passage, format_authors, extract_doi
)

logger = logging.getLogger(__name__)

# Configuration
PAPERS_PER_HYPOTHESIS = 3  # Max papers to attach per hypothesis
MAX_HYPOTHESES = 20  # Max hypotheses to ground (prioritize by tier)


async def ground_hypothesis(
    hypothesis: Hypothesis
) -> tuple[Hypothesis, list[str]]:
    """
    Find literature support for a single hypothesis.

    Searches Semantic Scholar using the hypothesis claim, scores papers
    by relevance, and attaches the top matches.

    Args:
        hypothesis: The hypothesis to ground

    Returns:
        Tuple of (updated hypothesis with literature, list of errors)
    """
    errors: list[str] = []

    # Use hypothesis claim as search query (S2 handles relevance internally)
    query = hypothesis.claim
    if not query:
        return hypothesis, ["Empty hypothesis claim"]

    # Search Semantic Scholar
    papers = await search_papers(query, limit=PAPERS_PER_HYPOTHESIS * 2)

    if not papers:
        logger.debug("No papers found for hypothesis: %s", hypothesis.title[:50])
        return hypothesis, []

    # Score and rank papers by relevance
    scored_papers: list[tuple[float, dict]] = []
    for paper in papers:
        score = score_relevance(paper, hypothesis.claim)
        scored_papers.append((score, paper))

    scored_papers.sort(key=lambda x: x[0], reverse=True)
    top_papers = scored_papers[:PAPERS_PER_HYPOTHESIS]

    # Build LiteratureSupport objects
    literature: list[LiteratureSupport] = []
    for score, paper in top_papers:
        try:
            lit_support = LiteratureSupport(
                paper_id=paper.get("paperId", ""),
                title=paper.get("title") or "Unknown",
                authors=format_authors(paper.get("authors", [])),
                year=paper.get("year") or 0,
                doi=extract_doi(paper),
                relevance_score=round(score, 3),
                relationship=classify_relationship(paper, hypothesis.claim),
                key_passage=extract_key_passage(paper, hypothesis.claim),
                citation_count=paper.get("citationCount") or 0,
            )
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing paper {paper.get('paperId', '?')}: {e}")

    # Update hypothesis with literature using model_copy (frozen model)
    if literature:
        updated = hypothesis.model_copy(update={"literature_support": literature})
        return updated, errors

    return hypothesis, errors


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Ground synthesis hypotheses with literature citations.

    Prioritizes hypotheses by tier (1 > 2 > 3) up to MAX_HYPOTHESES.
    Uses controlled concurrency to respect S2 rate limits.

    Args:
        state: Current discovery state with hypotheses

    Returns:
        Updated state with literature_support populated on hypotheses
    """
    logger.info("Starting literature_grounding")
    start = time.time()

    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        logger.info("No hypotheses to ground")
        return {
            "hypotheses": [],
            "literature_errors": [],
        }

    # Prioritize by tier (1 = highest confidence, 3 = speculative)
    sorted_hypotheses = sorted(hypotheses, key=lambda h: h.tier)
    to_ground = sorted_hypotheses[:MAX_HYPOTHESES]
    remaining = sorted_hypotheses[MAX_HYPOTHESES:]

    logger.info(
        "Grounding %d/%d hypotheses (tiers: %s)",
        len(to_ground), len(hypotheses),
        [h.tier for h in to_ground[:10]]  # Log first 10
    )

    # Process hypotheses with controlled concurrency
    all_errors: list[str] = []
    grounded: list[Hypothesis] = []

    # Process in small batches to respect S2 rate limits
    batch_size = 3
    for i in range(0, len(to_ground), batch_size):
        batch = to_ground[i:i + batch_size]
        tasks = [ground_hypothesis(h) for h in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                all_errors.append(f"Exception: {result}")
                # Keep original hypothesis if grounding failed
                grounded.append(batch[idx])
            else:
                updated_hyp, errors = result
                grounded.append(updated_hyp)
                all_errors.extend(errors)

    # Add ungrounded hypotheses (beyond MAX_HYPOTHESES)
    grounded.extend(remaining)

    # Count papers found
    papers_found = sum(len(h.literature_support) for h in grounded)

    duration = time.time() - start
    logger.info(
        "Completed literature_grounding in %.1fs — %d hypotheses, %d papers found",
        duration, len(grounded), papers_found
    )

    return {
        "hypotheses": grounded,
        "literature_errors": all_errors,
    }
