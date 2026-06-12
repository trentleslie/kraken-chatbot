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
# defusedxml hardens against XXE / billion-laughs / quadratic-blowup in the
# EFetch XML, which arrives over the network. Same API as xml.etree.ElementTree.
import defusedxml.ElementTree as ET
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import Element, ParseError

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Rate limiting: 3 req/s without key, 10 req/s with key.
PUBMED_SEMAPHORE = asyncio.Semaphore(2)  # Conservative concurrent requests


def _ncbi_api_key() -> str:
    """Read NCBI_API_KEY at call time, not import time — so a `.env` loaded after this
    module is first imported (tests, scripts) is still honored."""
    return os.getenv("NCBI_API_KEY", "")


def _pubmed_delay() -> float:
    """0.1s (~10 req/s) with an API key, else 0.35s (~3 req/s)."""
    return 0.1 if _ncbi_api_key() else 0.35


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
    api_key = _ncbi_api_key()
    if api_key:
        params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await asyncio.sleep(_pubmed_delay())  # Rate limit
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
    api_key = _ncbi_api_key()
    if api_key:
        params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await asyncio.sleep(_pubmed_delay())  # Rate limit
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


def normalize_pmid(pmid: str) -> str:
    """Strip a 'PMID:' prefix and surrounding whitespace, returning the bare digits."""
    return pmid.replace("PMID:", "").strip()


def _parse_efetch_abstracts(xml_text: str) -> dict[str, str]:
    """
    Parse an EFetch PubMed XML payload into {pmid: abstract_text}.

    Structured abstracts split the body across multiple <AbstractText> elements,
    each optionally carrying a Label (e.g. BACKGROUND, METHODS). We join the
    segments in document order, prefixing labels when present. PMIDs with no
    abstract are omitted from the result.

    Args:
        xml_text: Raw EFetch XML response body.

    Returns:
        Dict mapping bare PMID string to concatenated abstract text.
    """
    abstracts: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except (ParseError, DefusedXmlException) as e:
        logger.error("PubMed EFetch XML parse/safety error: %s", e)
        return abstracts

    for article in root.iter("PubmedArticle"):
        pmid_el: Element | None = article.find("./MedlineCitation/PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            continue

        segments: list[str] = []
        for at in article.iter("AbstractText"):
            text = "".join(at.itertext()).strip()  # flatten inline markup (e.g. <i>, <sup>)
            if not text:
                continue
            label = (at.get("Label") or "").strip()
            segments.append(f"{label}: {text}" if label else text)

        if segments:
            abstracts[pmid] = "\n".join(segments)

    return abstracts


async def _efetch_batch(pmids: list[str]) -> dict[str, str]:
    """Fetch one batch of abstracts via EFetch (XML). Returns {} on failure."""
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    api_key = _ncbi_api_key()
    if api_key:
        params["api_key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            await asyncio.sleep(_pubmed_delay())  # Rate limit (same policy as ESearch/ESummary)
            response = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
            response.raise_for_status()
            return _parse_efetch_abstracts(response.text)
        except httpx.HTTPStatusError as e:
            logger.error("PubMed EFetch HTTP error: %s", e)
            return {}
        except Exception as e:
            logger.error("PubMed EFetch failed: %s", e)
            return {}


async def fetch_abstracts(pmids: list[str], batch_size: int = 200) -> dict[str, str]:
    """
    Fetch abstract bodies for a list of PMIDs via EFetch.

    ESummary (used by ``search_papers``) returns metadata only; abstract bodies
    require EFetch. PMIDs may be passed with or without a 'PMID:' prefix; the
    returned dict is keyed by the bare numeric PMID. PMIDs without an abstract
    (or that fail to fetch) are simply absent from the result — callers treat a
    missing key as "no abstract available".

    Reuses the module's NCBI rate-limit policy (PUBMED_SEMAPHORE / _pubmed_delay() /
    _ncbi_api_key()). Input is deduplicated so each PMID is fetched at most once.

    Args:
        pmids: PMID strings (bare or 'PMID:'-prefixed).
        batch_size: Max PMIDs per EFetch request (NCBI tolerates a few hundred).

    Returns:
        Dict mapping bare PMID string to abstract text.
    """
    # Normalize, validate (numeric only), and dedupe while preserving order.
    seen: set[str] = set()
    clean: list[str] = []
    for raw in pmids:
        pmid = normalize_pmid(str(raw))
        if pmid.isdigit() and pmid not in seen:
            seen.add(pmid)
            clean.append(pmid)

    if not clean:
        return {}

    # Acquire the semaphore per batch (one EFetch request = one logical unit, as
    # search_papers does) rather than holding it across the whole loop — otherwise
    # a large PMID set starves concurrent pipeline invocations sharing this
    # module-level semaphore for the full duration.
    results: dict[str, str] = {}
    for i in range(0, len(clean), batch_size):
        batch = clean[i:i + batch_size]
        async with PUBMED_SEMAPHORE:
            results.update(await _efetch_batch(batch))

    logger.info(
        "PubMed EFetch: %d abstracts retrieved for %d PMIDs", len(results), len(clean)
    )
    return results
