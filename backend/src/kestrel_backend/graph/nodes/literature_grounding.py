"""
Literature Grounding Node: Add verified citations to synthesis hypotheses.

Multi-source approach:
1. KG PMIDs - Collect from existing findings (free, instant)
2. OpenAlex - Free API, no rate limits, 250M+ works
3. Semantic Scholar - When API key available (rate-limited without key)

Position in pipeline: Runs after synthesis, before final output.
"""

import asyncio
import logging
import time
from typing import Any

from ..state import DiscoveryState, Hypothesis, LiteratureSupport
from ...literature_utils import pmid_to_url, doi_to_url
from ...openalex import (
    search_works, extract_pmid_from_work, extract_doi_from_work, format_authors_from_work
)
from ...semantic_scholar import (
    search_papers, score_relevance, classify_relationship,
    extract_key_passage, format_authors, extract_doi, S2RateLimitError
)

logger = logging.getLogger(__name__)

# Configuration
PAPERS_PER_HYPOTHESIS = 3  # Max papers to attach per hypothesis
MAX_HYPOTHESES = 15  # Max hypotheses to ground (prioritize by tier)


def collect_pmids_from_state(state: DiscoveryState) -> dict[str, list[str]]:
    """
    Collect PMIDs from existing findings grouped by entity.

    Returns:
        Dict mapping entity CURIE to list of PMIDs
    """
    entity_pmids: dict[str, list[str]] = {}

    # From direct_findings
    for finding in state.get("direct_findings", []):
        if hasattr(finding, 'pmids') and finding.pmids:
            entity = finding.entity
            if entity not in entity_pmids:
                entity_pmids[entity] = []
            entity_pmids[entity].extend(finding.pmids)

    # From disease_associations
    for assoc in state.get("disease_associations", []):
        if hasattr(assoc, 'pmids') and assoc.pmids:
            entity = assoc.entity_curie
            if entity not in entity_pmids:
                entity_pmids[entity] = []
            entity_pmids[entity].extend(assoc.pmids)

    # Deduplicate
    for entity in entity_pmids:
        entity_pmids[entity] = list(set(entity_pmids[entity]))

    # Log results
    total_pmids = sum(len(v) for v in entity_pmids.values())
    logger.info("KG PMIDs collected: %d PMIDs across %d entities", total_pmids, len(entity_pmids))

    return entity_pmids


def create_literature_from_pmid(pmid: str) -> LiteratureSupport:
    """Create LiteratureSupport from PMID with PubMed URL."""
    pmid_num = pmid.replace("PMID:", "").strip()

    return LiteratureSupport(
        paper_id=f"PMID:{pmid_num}",
        title=f"PubMed:{pmid_num}",  # Title unknown without fetching
        authors="",
        year=0,
        url=pmid_to_url(pmid),
        relevance_score=1.0,  # Direct KG evidence = highly relevant
        relationship="supporting",
        key_passage="",
        citation_count=0,
        source="kg",
    )


def create_literature_from_openalex(work: dict, relevance: float = 0.8) -> LiteratureSupport:
    """Create LiteratureSupport from OpenAlex work."""
    pmid = extract_pmid_from_work(work)
    doi = extract_doi_from_work(work)

    # Prefer PubMed URL, fall back to DOI
    url = None
    if pmid:
        url = pmid_to_url(pmid)
    elif doi:
        url = doi_to_url(doi)

    return LiteratureSupport(
        paper_id=pmid or work.get("id", ""),
        title=work.get("title") or "Unknown",
        authors=format_authors_from_work(work),
        year=work.get("publication_year") or 0,
        doi=doi,
        url=url,
        relevance_score=relevance,
        relationship="supporting",
        key_passage="",
        citation_count=work.get("cited_by_count") or 0,
        source="openalex",
    )


def create_literature_from_s2(paper: dict, hypothesis_claim: str) -> LiteratureSupport:
    """Create LiteratureSupport from Semantic Scholar paper."""
    doi = extract_doi(paper)
    pmid = None

    # Try to get PMID from external IDs
    external_ids = paper.get("externalIds") or {}
    if external_ids.get("PubMed"):
        pmid = external_ids["PubMed"]

    # Build URL
    url = None
    if pmid:
        url = pmid_to_url(pmid)
    elif doi:
        url = doi_to_url(doi)

    return LiteratureSupport(
        paper_id=paper.get("paperId", ""),
        title=paper.get("title") or "Unknown",
        authors=format_authors(paper.get("authors", [])),
        year=paper.get("year") or 0,
        doi=doi,
        url=url,
        relevance_score=round(score_relevance(paper, hypothesis_claim), 3),
        relationship=classify_relationship(paper, hypothesis_claim),
        key_passage=extract_key_passage(paper, hypothesis_claim),
        citation_count=paper.get("citationCount") or 0,
        source="s2",
    )


async def ground_hypothesis_openalex(
    hypothesis: Hypothesis
) -> tuple[Hypothesis, list[str]]:
    """
    Find literature support using OpenAlex (free, no rate limits).

    Args:
        hypothesis: The hypothesis to ground

    Returns:
        Tuple of (updated hypothesis with literature, list of errors)
    """
    errors: list[str] = []
    query = hypothesis.claim
    if not query:
        return hypothesis, ["Empty hypothesis claim"]

    # Search OpenAlex
    works = await search_works(query, limit=PAPERS_PER_HYPOTHESIS * 2)

    if not works:
        logger.debug("No OpenAlex papers found for: %s", hypothesis.title[:50])
        return hypothesis, []

    # Build LiteratureSupport objects from top results
    literature: list[LiteratureSupport] = []
    for i, work in enumerate(works[:PAPERS_PER_HYPOTHESIS]):
        try:
            # Score decreases slightly for lower-ranked results
            relevance = 0.9 - (i * 0.05)
            lit_support = create_literature_from_openalex(work, relevance)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing OpenAlex work: {e}")

    # Update hypothesis with literature
    if literature:
        updated = hypothesis.model_copy(update={"literature_support": literature})
        return updated, errors

    return hypothesis, errors


async def ground_hypothesis_s2(
    hypothesis: Hypothesis
) -> tuple[Hypothesis, list[str]]:
    """
    Find literature support using Semantic Scholar.

    Args:
        hypothesis: The hypothesis to ground

    Returns:
        Tuple of (updated hypothesis with literature, list of errors)
    """
    errors: list[str] = []
    query = hypothesis.claim
    if not query:
        return hypothesis, ["Empty hypothesis claim"]

    # Search Semantic Scholar
    papers = await search_papers(query, limit=PAPERS_PER_HYPOTHESIS * 2)

    if not papers:
        logger.debug("No S2 papers found for: %s", hypothesis.title[:50])
        return hypothesis, []

    # Score and rank papers
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
            lit_support = create_literature_from_s2(paper, hypothesis.claim)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing S2 paper: {e}")

    # Update hypothesis with literature
    if literature:
        updated = hypothesis.model_copy(update={"literature_support": literature})
        return updated, errors

    return hypothesis, errors


def ground_hypothesis_kg(
    hypothesis: Hypothesis,
    kg_pmids: dict[str, list[str]]
) -> tuple[Hypothesis, bool]:
    """
    Try to ground hypothesis using KG PMIDs from supporting entities.

    Returns:
        Tuple of (hypothesis with literature, True if any PMIDs found)
    """
    literature: list[LiteratureSupport] = []

    for entity in hypothesis.supporting_entities:
        if entity in kg_pmids:
            for pmid in kg_pmids[entity][:PAPERS_PER_HYPOTHESIS]:
                if len(literature) >= PAPERS_PER_HYPOTHESIS:
                    break
                literature.append(create_literature_from_pmid(pmid))

    if literature:
        updated = hypothesis.model_copy(update={"literature_support": literature})
        return updated, True

    return hypothesis, False


async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Ground synthesis hypotheses with literature citations.

    Strategy:
    1. Check KG PMIDs first (free, instant)
    2. Use OpenAlex for hypotheses without KG PMIDs (free, no rate limits)
    3. S2 as fallback when OpenAlex returns no results

    Args:
        state: Current discovery state with hypotheses

    Returns:
        Updated state with literature_support populated on hypotheses
    """
    logger.info("Starting literature_grounding (multi-source)")
    start = time.time()

    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        logger.info("No hypotheses to ground")
        return {
            "hypotheses": [],
            "literature_errors": [],
        }

    # Step 1: Collect KG PMIDs from findings
    kg_pmids = collect_pmids_from_state(state)

    # Prioritize hypotheses by tier
    sorted_hypotheses = sorted(hypotheses, key=lambda h: h.tier)
    to_ground = sorted_hypotheses[:MAX_HYPOTHESES]
    remaining = sorted_hypotheses[MAX_HYPOTHESES:]

    logger.info(
        "Grounding %d/%d hypotheses (tiers: %s)",
        len(to_ground), len(hypotheses),
        [h.tier for h in to_ground[:10]]
    )

    # Step 2: Process each hypothesis through fallback chain
    all_errors: list[str] = []
    grounded: list[Hypothesis] = []

    for hypothesis in to_ground:
        # Try KG PMIDs first (free, instant)
        updated, has_kg = ground_hypothesis_kg(hypothesis, kg_pmids)
        if has_kg:
            grounded.append(updated)
            continue

        # Fall back to OpenAlex
        updated, errors = await ground_hypothesis_openalex(hypothesis)
        all_errors.extend(errors)

        # If OpenAlex found nothing, try S2
        if not updated.literature_support:
            updated, s2_errors = await ground_hypothesis_s2(hypothesis)
            all_errors.extend(s2_errors)

        grounded.append(updated)

    # Add ungrounded hypotheses
    grounded.extend(remaining)

    # Count papers found
    papers_found = sum(len(h.literature_support) for h in grounded)
    kg_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "kg")
    openalex_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "openalex")

    duration = time.time() - start
    logger.info(
        "Completed literature_grounding in %.1fs — %d hypotheses, %d papers (kg=%d, openalex=%d)",
        duration, len(grounded), papers_found, kg_papers, openalex_papers
    )

    return {
        "hypotheses": grounded,
        "literature_errors": all_errors,
    }
