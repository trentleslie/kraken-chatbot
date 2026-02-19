"""
Literature Grounding Node: Add verified citations to synthesis hypotheses.

Parallel multi-source approach:
1. KG PMIDs - Collect from existing findings (free, instant)
2. Parallel search: OpenAlex + Exa + PubMed simultaneously
3. Merge/deduplicate by DOI → PMID → title+year
4. Semantic Scholar - Fallback only if parallel search finds nothing

Position in pipeline: Runs after synthesis, before final output.
"""

import asyncio
import logging
import re
import time
from typing import Any

from ..state import DiscoveryState, Hypothesis, LiteratureSupport
from ...literature_utils import pmid_to_url, doi_to_url
from ...openalex import (
    search_works, extract_pmid_from_work, extract_doi_from_work, format_authors_from_work
)
from ...semantic_scholar import (
    search_papers as s2_search_papers, score_relevance, classify_relationship,
    extract_key_passage, format_authors, extract_doi, S2RateLimitError
)
from ...exa_client import (
    search_papers as exa_search_papers, ExaSearchError,
    extract_doi_from_url, extract_year_from_date
)
from ...pubmed_client import (
    search_papers as pubmed_search_papers, PubMedSearchError
)

logger = logging.getLogger(__name__)

# Configuration
PAPERS_PER_HYPOTHESIS = 3  # Max papers to attach per hypothesis
MAX_HYPOTHESES = 15  # Max hypotheses to ground (prioritize by tier)
PARALLEL_FETCH_LIMIT = 6  # Fetch 2x papers per source for deduplication

# Source priority for deduplication (lower = higher priority)
SOURCE_PRIORITY = {"kg": 1, "pubmed": 2, "openalex": 3, "exa": 4, "s2": 5}

# Low-quality Exa result patterns to skip
EXA_SKIP_AUTHORS = {"Web Team", "EBI", "EMBL-EBI"}
EXA_SKIP_URL_PATTERNS = [
    "ebi.ac.uk/chebi",
    "uniprot.org",
    "omim.org",
    "ncbi.nlm.nih.gov/books",
    "ncbi.nlm.nih.gov/gene",
    "genome.jp/kegg",
    "qiagen.com",
    "genetests.org",
]
EXA_SKIP_TITLES = {
    "Gene Ontology Resource",
    "The Gene Ontology Resource",
    "STRING database",
    "STRING: functional protein association networks",
}

# Filler phrases to remove from search queries
FILLER_PATTERNS = [
    r"\(inferred via[^)]*\)",      # "(inferred via semantic similarity)"
    r"\(predicted[^)]*\)",          # "(predicted association)"
    r"may be associated with",
    r"is associated with",
    r"potentially linked to",
    r"might be involved in",
    r"could be related to",
]


def is_valid_exa_result(result: dict) -> bool:
    """
    Filter out database pages and low-quality Exa results.

    Rejects:
    - Results with year=0 (usually database pages without publication dates)
    - Results from known generic authors (Web Team, EBI, EMBL-EBI)
    - Database page URLs (ChEBI, UniProt)
    - Known generic titles (Gene Ontology, STRING)

    Args:
        result: Raw Exa search result dict

    Returns:
        True if result is a valid paper, False if it should be skipped
    """
    # Skip if year is 0 (database pages often have no publication date)
    year = extract_year_from_date(result.get("publishedDate"))
    if year == 0:
        return False

    # Skip if author contains known generic authors
    author = result.get("author") or ""
    if any(skip in author for skip in EXA_SKIP_AUTHORS):
        return False

    # Skip database pages by URL pattern
    url = result.get("url") or ""
    if any(pattern in url for pattern in EXA_SKIP_URL_PATTERNS):
        return False

    # Skip known generic titles
    title = result.get("title") or ""
    if title in EXA_SKIP_TITLES:
        return False

    return True


def _get_paper_key(lit: LiteratureSupport) -> str:
    """
    Generate key for paper deduplication in references table.

    Uses DOI if available, otherwise falls back to normalized title+year.
    Different from get_unique_key() which also checks PMID for merge-time dedup.

    Args:
        lit: Literature support object

    Returns:
        Unique key string for deduplication
    """
    if lit.doi:
        return f"doi:{lit.doi.lower()}"
    return f"title:{lit.title.lower()[:100]}:{lit.year}"


def build_references_table(hypotheses: list[Hypothesis]) -> str:
    """
    Build markdown table of literature references for synthesis report.

    Deduplicates:
    - Hypotheses by title (prevents iteration over duplicates)
    - Papers by DOI/title across all hypotheses (combines hypothesis titles)

    Includes a Relevance column showing the key_passage for quick scanning.

    Args:
        hypotheses: Grounded hypotheses with literature_support

    Returns:
        Markdown string with table of citations
    """
    # Deduplicate hypotheses by title
    seen_titles: set[str] = set()
    unique_hypotheses: list[Hypothesis] = []
    for h in hypotheses:
        if h.title not in seen_titles:
            seen_titles.add(h.title)
            unique_hypotheses.append(h)

    # Filter to hypotheses with literature
    with_lit = [h for h in unique_hypotheses if h.literature_support]
    if not with_lit:
        return ""

    # Group papers by key, track which hypotheses cite each
    paper_to_hypotheses: dict[str, tuple[LiteratureSupport, list[str]]] = {}
    for hypothesis in with_lit:
        hyp_title = hypothesis.title[:80] + "..." if len(hypothesis.title) > 80 else hypothesis.title
        for lit in hypothesis.literature_support:
            key = _get_paper_key(lit)
            if key not in paper_to_hypotheses:
                paper_to_hypotheses[key] = (lit, [])
            paper_to_hypotheses[key][1].append(hyp_title)

    # Build table header
    lines = [
        "\n## Literature References\n",
        f"Papers discovered via semantic search. {len(paper_to_hypotheses)} unique papers across {len(with_lit)} hypotheses.\n",
        "| Hypothesis | Citation | Link | Relevance |",
        "|------------|----------|------|-----------|",
    ]

    # Build rows - one per unique paper
    for key, (lit, hyp_titles) in sorted(paper_to_hypotheses.items()):
        # Combine and escape hypothesis titles (dedupe within paper)
        combined_titles = "; ".join(sorted(set(hyp_titles)))
        combined_titles = combined_titles.replace("|", "\\|")

        # Format citation: "Authors (Year) Title"
        title_truncated = lit.title[:100] + "..." if len(lit.title) > 100 else lit.title
        citation = f'{lit.authors} ({lit.year}) "{title_truncated}"'
        citation = citation.replace("|", "\\|")

        # Prefer DOI link, fall back to url
        if lit.doi:
            link = f"[DOI](https://doi.org/{lit.doi})"
        elif lit.url:
            link = f"[Link]({lit.url})"
        else:
            link = "—"

        # Sanitize relevance - strip everything that could break markdown table
        raw_passage = lit.key_passage or ""
        was_truncated = len(raw_passage) > 120
        relevance = raw_passage[:120]
        relevance = relevance.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        relevance = re.sub(r'\[.*?\]\(.*?\)', '', relevance)  # Strip markdown links
        relevance = re.sub(r'<[^>]+>', '', relevance)  # Strip HTML tags
        relevance = re.sub(r'[#*>`~]', '', relevance)  # Strip markdown formatting chars
        relevance = re.sub(r'[âÂ€™""—–]+', '', relevance)  # Strip UTF-8 mojibake artifacts
        relevance = re.sub(r'\s+', ' ', relevance).strip()  # Collapse whitespace
        if not relevance:
            relevance = "—"
        elif was_truncated:
            relevance += "..."

        lines.append(f"| {combined_titles} | {citation} | {link} | {relevance} |")

    return "\n".join(lines)


def build_search_query(hypothesis: Hypothesis, disease_context: str = "") -> str:
    """
    Extract concise search terms from hypothesis.

    Strategy:
    1. Start with title (more concise than claim)
    2. Fall back to claim for generic titles (Bridge:, Inferred role of)
    3. Strip filler phrases and parenthetical metadata
    4. Append disease context to anchor results
    5. Truncate to 200 chars for API limits

    Args:
        hypothesis: The hypothesis to build query for
        disease_context: Optional disease focus to append (e.g., "type 2 diabetes")
    """
    # Prefer title, fall back to claim
    text = hypothesis.title or hypothesis.claim

    # Generic titles have no searchable content — use claim instead
    if text.startswith(("Bridge:", "Inferred role of")):
        text = hypothesis.claim

    # Remove filler phrases
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Append disease context to anchor search results
    if disease_context:
        text = f"{text} {disease_context}"

    # Truncate for API limits
    return text[:200]


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


def create_literature_from_exa(result: dict) -> LiteratureSupport:
    """Create LiteratureSupport from Exa AI result."""
    doi = extract_doi_from_url(result.get("url", ""))
    year = extract_year_from_date(result.get("publishedDate"))
    highlights = result.get("highlights") or []
    key_passage = highlights[0] if highlights else ""

    return LiteratureSupport(
        paper_id=result.get("id", ""),
        title=result.get("title") or "Unknown",
        authors=result.get("author") or "",
        year=year,
        doi=doi,
        url=result.get("url"),
        relevance_score=min(result.get("score", 0.85), 1.0),  # Use Exa's actual score
        relationship="supporting",
        key_passage=key_passage[:300] if key_passage else "",
        citation_count=0,
        source="exa",
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


def create_literature_from_pubmed(paper: dict, relevance: float = 0.85) -> LiteratureSupport:
    """Create LiteratureSupport from PubMed ESummary result."""
    pmid = paper.get("pmid", "")
    doi = paper.get("doi")

    # Build URL - prefer PubMed URL
    url = pmid_to_url(pmid) if pmid else None
    if not url and doi:
        url = doi_to_url(doi)

    return LiteratureSupport(
        paper_id=f"PMID:{pmid}" if pmid else "",
        title=paper.get("title") or "Unknown",
        authors=paper.get("authors") or "Unknown",
        year=paper.get("year") or 0,
        doi=doi,
        url=url,
        relevance_score=relevance,
        relationship="supporting",
        key_passage="",
        citation_count=paper.get("citation_count") or 0,
        source="pubmed",
    )


def get_unique_key(lit: LiteratureSupport) -> str:
    """
    Generate unique key for deduplication.

    Priority: DOI > PMID > title+year
    """
    if lit.doi:
        return f"doi:{lit.doi.lower()}"
    if lit.paper_id.startswith("PMID:"):
        pmid = lit.paper_id.replace("PMID:", "")
        return f"pmid:{pmid}"
    # Normalize title: lowercase, first 100 chars
    title_norm = lit.title.lower()[:100].strip()
    return f"title:{title_norm}:{lit.year}"


def merge_literature(
    all_results: list[LiteratureSupport],
    limit: int = PAPERS_PER_HYPOTHESIS
) -> list[LiteratureSupport]:
    """
    Deduplicate literature by unique key, prefer higher-priority sources.

    When duplicates found:
    - Keep the paper from the higher-priority source
    - Merge complementary fields (citation_count, key_passage, doi) from lower-priority

    Args:
        all_results: Combined literature from all sources
        limit: Maximum results to return

    Returns:
        Deduplicated list sorted by relevance_score, limited to `limit` items
    """
    if not all_results:
        return []

    # Group by unique key
    by_key: dict[str, list[LiteratureSupport]] = {}
    for lit in all_results:
        key = get_unique_key(lit)
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(lit)

    # For each group, pick best source and merge complementary fields
    merged: list[LiteratureSupport] = []
    for key, papers in by_key.items():
        # Sort by source priority (lowest = best)
        papers.sort(key=lambda p: SOURCE_PRIORITY.get(p.source, 99))
        best = papers[0]

        # Merge complementary fields from lower-priority sources
        merged_citation_count = best.citation_count
        merged_key_passage = best.key_passage
        merged_doi = best.doi

        for other in papers[1:]:
            if other.citation_count > merged_citation_count:
                merged_citation_count = other.citation_count
            if not merged_key_passage and other.key_passage:
                merged_key_passage = other.key_passage
            if not merged_doi and other.doi:
                merged_doi = other.doi

        # Create merged result if any fields changed
        if (merged_citation_count != best.citation_count or
            merged_key_passage != best.key_passage or
            merged_doi != best.doi):
            best = LiteratureSupport(
                paper_id=best.paper_id,
                title=best.title,
                authors=best.authors,
                year=best.year,
                doi=merged_doi,
                url=best.url,
                relevance_score=best.relevance_score,
                relationship=best.relationship,
                key_passage=merged_key_passage,
                citation_count=merged_citation_count,
                source=best.source,
            )

        merged.append(best)

    # Sort by relevance score descending, return top `limit`
    merged.sort(key=lambda p: p.relevance_score, reverse=True)
    return merged[:limit]


async def ground_hypothesis_openalex_raw(
    hypothesis: Hypothesis,
    limit: int = PARALLEL_FETCH_LIMIT,
    disease_context: str = "",
) -> tuple[list[LiteratureSupport], list[str]]:
    """
    Search OpenAlex and return raw literature results (no hypothesis update).

    Args:
        hypothesis: The hypothesis to ground
        limit: Maximum results to fetch
        disease_context: Disease focus to append to query (e.g., "type 2 diabetes")

    Returns:
        Tuple of (list of literature, list of errors)
    """
    errors: list[str] = []
    query = build_search_query(hypothesis, disease_context)
    if not query:
        return [], ["Empty search query"]

    works = await search_works(query, limit=limit)

    if not works:
        logger.debug("No OpenAlex papers found for: %s", hypothesis.title[:50])
        return [], []

    literature: list[LiteratureSupport] = []
    for i, work in enumerate(works):
        try:
            relevance = max(0.0, 0.9 - (i * 0.03))  # Slight decay for lower-ranked
            lit_support = create_literature_from_openalex(work, relevance)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing OpenAlex work: {e}")

    return literature, errors


async def ground_hypothesis_exa_raw(
    hypothesis: Hypothesis,
    limit: int = PARALLEL_FETCH_LIMIT,
    disease_context: str = "",
) -> tuple[list[LiteratureSupport], list[str]]:
    """
    Search Exa and return raw literature results (no hypothesis update).

    Args:
        hypothesis: The hypothesis to ground
        limit: Maximum results to fetch
        disease_context: Disease focus to append to query (e.g., "type 2 diabetes")

    Returns:
        Tuple of (list of literature, list of errors)
    """
    errors: list[str] = []
    query = build_search_query(hypothesis, disease_context)
    if not query:
        return [], ["Empty search query"]

    try:
        results = await exa_search_papers(query, limit=limit)
    except ExaSearchError as e:
        logger.warning("Exa error for hypothesis: %s — %s", hypothesis.title[:50], e)
        return [], [f"Exa error: {e}"]

    if not results:
        logger.debug("No Exa papers found for: %s", hypothesis.title[:50])
        return [], []

    # Filter out low-quality results before processing
    valid_results = [r for r in results if is_valid_exa_result(r)]
    if len(valid_results) < len(results):
        logger.debug(
            "Exa quality filter: %d/%d results kept for '%s'",
            len(valid_results), len(results), hypothesis.title[:40]
        )

    literature: list[LiteratureSupport] = []
    for result in valid_results:
        try:
            lit_support = create_literature_from_exa(result)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing Exa result: {e}")

    return literature, errors


async def ground_hypothesis_pubmed_raw(
    hypothesis: Hypothesis,
    limit: int = PARALLEL_FETCH_LIMIT,
    disease_context: str = "",
) -> tuple[list[LiteratureSupport], list[str]]:
    """
    Search PubMed and return raw literature results (no hypothesis update).

    Args:
        hypothesis: The hypothesis to ground
        limit: Maximum results to fetch
        disease_context: Disease focus to append to query (e.g., "type 2 diabetes")

    Returns:
        Tuple of (list of literature, list of errors)
    """
    errors: list[str] = []
    query = build_search_query(hypothesis, disease_context)
    if not query:
        return [], ["Empty search query"]

    try:
        papers = await pubmed_search_papers(query, limit=limit)
    except PubMedSearchError as e:
        logger.warning("PubMed error for hypothesis: %s — %s", hypothesis.title[:50], e)
        return [], [f"PubMed error: {e}"]

    if not papers:
        logger.debug("No PubMed papers found for: %s", hypothesis.title[:50])
        return [], []

    literature: list[LiteratureSupport] = []
    for i, paper in enumerate(papers):
        try:
            relevance = 0.88 - (i * 0.03)  # Slight decay for lower-ranked
            lit_support = create_literature_from_pubmed(paper, relevance)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing PubMed paper: {e}")

    return literature, errors


async def parallel_search_hypothesis(
    hypothesis: Hypothesis,
    disease_context: str = "",
) -> tuple[list[LiteratureSupport], list[str]]:
    """
    Run OpenAlex + Exa + PubMed searches in parallel, merge results.

    Args:
        hypothesis: The hypothesis to ground
        disease_context: Disease focus to append to queries (e.g., "type 2 diabetes")

    Returns:
        Tuple of (merged/deduped literature, list of errors)
    """
    # Run all three searches in parallel
    results = await asyncio.gather(
        ground_hypothesis_openalex_raw(hypothesis, disease_context=disease_context),
        ground_hypothesis_exa_raw(hypothesis, disease_context=disease_context),
        ground_hypothesis_pubmed_raw(hypothesis, disease_context=disease_context),
        return_exceptions=True,
    )

    all_literature: list[LiteratureSupport] = []
    all_errors: list[str] = []

    source_names = ["OpenAlex", "Exa", "PubMed"]
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            all_errors.append(f"{source_names[i]} exception: {result}")
            continue
        literature, errors = result
        all_literature.extend(literature)
        all_errors.extend(errors)

    # Merge and deduplicate
    merged = merge_literature(all_literature)

    logger.debug(
        "Parallel search for '%s': %d total → %d merged (openalex=%d, exa=%d, pubmed=%d)",
        hypothesis.title[:40],
        len(all_literature),
        len(merged),
        sum(1 for l in all_literature if l.source == "openalex"),
        sum(1 for l in all_literature if l.source == "exa"),
        sum(1 for l in all_literature if l.source == "pubmed"),
    )

    return merged, all_errors


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
    query = build_search_query(hypothesis)
    if not query:
        return hypothesis, ["Empty search query"]

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


async def ground_hypothesis_exa(
    hypothesis: Hypothesis
) -> tuple[Hypothesis, list[str]]:
    """
    Find literature support using Exa AI semantic search.

    Args:
        hypothesis: The hypothesis to ground

    Returns:
        Tuple of (updated hypothesis with literature, list of errors)
    """
    errors: list[str] = []
    query = build_search_query(hypothesis)
    if not query:
        return hypothesis, ["Empty search query"]

    try:
        results = await exa_search_papers(query, limit=PAPERS_PER_HYPOTHESIS * 2)
    except ExaSearchError as e:
        logger.warning("Exa error for hypothesis: %s — %s", hypothesis.title[:50], e)
        return hypothesis, [f"Exa error: {e}"]

    if not results:
        logger.debug("No Exa papers found for: %s", hypothesis.title[:50])
        return hypothesis, []

    # Build LiteratureSupport objects
    literature: list[LiteratureSupport] = []
    for result in results[:PAPERS_PER_HYPOTHESIS]:
        try:
            lit_support = create_literature_from_exa(result)
            literature.append(lit_support)
        except Exception as e:
            errors.append(f"Error processing Exa result: {e}")

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
    query = build_search_query(hypothesis)
    if not query:
        return hypothesis, ["Empty search query"]

    # Search Semantic Scholar with rate limit handling
    try:
        papers = await s2_search_papers(query, limit=PAPERS_PER_HYPOTHESIS * 2)
    except S2RateLimitError:
        logger.warning("S2 rate limited, skipping for hypothesis: %s", hypothesis.title[:50])
        return hypothesis, ["S2 rate limited - skipped"]

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
    2. Parallel hybrid search: OpenAlex + Exa + PubMed simultaneously
    3. Merge/deduplicate by DOI → PMID → title+year
    4. S2 as fallback only if parallel search finds nothing (rate-limited)

    Args:
        state: Current discovery state with hypotheses

    Returns:
        Updated state with literature_support populated on hypotheses
    """
    logger.info("Starting literature_grounding (parallel hybrid search)")
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

    # Extract disease focus to anchor search queries
    # disease_focus is nested inside study_context dict from intake node
    study_context = state.get("study_context", {})
    disease_focus = study_context.get("disease_focus", "") if isinstance(study_context, dict) else ""
    if disease_focus:
        logger.info("Using disease context for queries: %s", disease_focus)

    # Prioritize hypotheses by tier
    sorted_hypotheses = sorted(hypotheses, key=lambda h: h.tier)
    to_ground = sorted_hypotheses[:MAX_HYPOTHESES]
    remaining = sorted_hypotheses[MAX_HYPOTHESES:]

    logger.info(
        "Grounding %d/%d hypotheses (tiers: %s)",
        len(to_ground), len(hypotheses),
        [h.tier for h in to_ground[:10]]
    )

    # Step 2: Process each hypothesis
    all_errors: list[str] = []
    grounded: list[Hypothesis] = []

    for hypothesis in to_ground:
        # Try KG PMIDs first (free, instant)
        updated, has_kg = ground_hypothesis_kg(hypothesis, kg_pmids)
        if has_kg:
            grounded.append(updated)
            continue

        # Parallel search: OpenAlex + Exa + PubMed
        literature, errors = await parallel_search_hypothesis(hypothesis, disease_context=disease_focus)
        all_errors.extend(errors)

        if literature:
            updated = hypothesis.model_copy(update={"literature_support": literature})
        else:
            # S2 fallback only if parallel search found nothing
            updated, s2_errors = await ground_hypothesis_s2(hypothesis)
            all_errors.extend(s2_errors)

        grounded.append(updated)

    # Add ungrounded hypotheses
    grounded.extend(remaining)

    # Count papers found by source
    papers_found = sum(len(h.literature_support) for h in grounded)
    kg_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "kg")
    pubmed_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "pubmed")
    openalex_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "openalex")
    exa_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "exa")
    s2_papers = sum(1 for h in grounded for lit in h.literature_support if lit.source == "s2")

    duration = time.time() - start
    logger.info(
        "Completed literature_grounding in %.1fs — %d hypotheses, %d papers "
        "(kg=%d, pubmed=%d, openalex=%d, exa=%d, s2=%d)",
        duration, len(grounded), papers_found,
        kg_papers, pubmed_papers, openalex_papers, exa_papers, s2_papers
    )

    # Build literature references table
    references_table = build_references_table(grounded)

    # Append to synthesis report if exists
    synthesis_report = state.get("synthesis_report", "")
    if references_table:
        synthesis_report = synthesis_report + "\n" + references_table

    return {
        "hypotheses": grounded,
        "literature_errors": all_errors,
        "synthesis_report": synthesis_report,
    }
