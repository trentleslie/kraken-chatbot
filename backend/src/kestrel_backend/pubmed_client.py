"""PubMed E-utilities API client for literature search.

NCBI E-utilities provides free access to PubMed's 35M+ citations.
https://www.ncbi.nlm.nih.gov/books/NBK25500/

Rate limits:
- Without API key: 3 requests/second
- With API key: 10 requests/second

We use ESearch to find PMIDs, then ESummary for metadata (JSON-friendly).
"""

import asyncio
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Rate limiting: 3 req/s without key, 10 req/s with key
PUBMED_SEMAPHORE = asyncio.Semaphore(2)  # Conservative concurrent requests
PUBMED_DELAY = 0.35 if not NCBI_API_KEY else 0.1  # Delay between requests


class PubMedSearchError(Exception):
    """PubMed API error."""
    pass


async def search_papers(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Search PubMed via ESearch → ESummary.

    Args:
        query: Search query (hypothesis or keywords)
        limit: Maximum results to return

    Returns:
        List of paper dicts with title, authors, year, pmid, doi
    """
    async with PUBMED_SEMAPHORE:
        pmids = await _esearch(query, limit)
        if not pmids:
            return []
        return await _esummary(pmids)


async def _esearch(query: str, limit: int) -> list[str]:
    """
    Get PMIDs matching query via ESearch.

    Args:
        query: Search query
        limit: Maximum PMIDs to return

    Returns:
        List of PMID strings
    """
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query[:500],  # Truncate long queries
        "retmode": "json",
        "retmax": limit,
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await asyncio.sleep(PUBMED_DELAY)  # Rate limit
            response = await client.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            # Extract PMIDs from response
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])

            logger.info("PubMed ESearch: %d PMIDs for query: %s", len(pmids), query[:50])
            return pmids

        except httpx.HTTPStatusError as e:
            logger.error("PubMed ESearch HTTP error: %s", e)
            raise PubMedSearchError(f"ESearch error {e.response.status_code}")
        except Exception as e:
            logger.error("PubMed ESearch failed: %s", e)
            return []


async def _esummary(pmids: list[str]) -> list[dict[str, Any]]:
    """
    Get paper metadata for PMIDs via ESummary.

    Args:
        pmids: List of PMID strings

    Returns:
        List of paper dicts with normalized fields
    """
    if not pmids:
        return []

    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await asyncio.sleep(PUBMED_DELAY)  # Rate limit
            response = await client.get(
                f"{EUTILS_BASE}/esummary.fcgi",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            # Parse ESummary result
            result = data.get("result", {})
            papers = []

            for pmid in pmids:
                if pmid not in result:
                    continue
                paper_data = result[pmid]
                if "error" in paper_data:
                    continue

                papers.append(_normalize_esummary(paper_data, pmid))

            logger.info("PubMed ESummary: %d papers retrieved", len(papers))
            return papers

        except httpx.HTTPStatusError as e:
            logger.error("PubMed ESummary HTTP error: %s", e)
            raise PubMedSearchError(f"ESummary error {e.response.status_code}")
        except Exception as e:
            logger.error("PubMed ESummary failed: %s", e)
            return []


def _normalize_esummary(paper: dict, pmid: str) -> dict[str, Any]:
    """
    Normalize ESummary response to standard paper dict.

    Args:
        paper: Raw ESummary paper object
        pmid: The PMID for this paper

    Returns:
        Normalized dict with title, authors, year, pmid, doi
    """
    return {
        "pmid": pmid,
        "title": paper.get("title", "Unknown"),
        "authors": format_authors_from_summary(paper),
        "year": extract_year_from_pubdate(paper.get("pubdate", "")),
        "doi": extract_doi_from_summary(paper),
        "source": paper.get("source", ""),  # Journal name
        "citation_count": 0,  # ESummary doesn't provide this
    }


def extract_doi_from_summary(paper: dict) -> str | None:
    """
    Extract DOI from ESummary articleids array.

    Args:
        paper: ESummary paper object

    Returns:
        DOI string or None
    """
    articleids = paper.get("articleids", [])
    for aid in articleids:
        if aid.get("idtype") == "doi":
            return aid.get("value")
    return None


def extract_year_from_pubdate(pubdate: str) -> int:
    """
    Parse year from PubMed pubdate string.

    Formats: "2024 Jan 15", "2024 Jan", "2024", "2024 Spring"

    Args:
        pubdate: PubMed date string

    Returns:
        Year as integer, or 0 if unparseable
    """
    if not pubdate:
        return 0
    match = re.match(r"(\d{4})", pubdate)
    if match:
        return int(match.group(1))
    return 0


def format_authors_from_summary(paper: dict) -> str:
    """
    Format author list from ESummary.

    Args:
        paper: ESummary paper object

    Returns:
        Author string like "Smith J" or "Smith J et al."
    """
    authors = paper.get("authors", [])
    if not authors:
        return "Unknown"

    first_author = authors[0].get("name", "Unknown")
    if len(authors) > 1:
        return f"{first_author} et al."
    return first_author
