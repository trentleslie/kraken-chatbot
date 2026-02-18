"""OpenAlex API client for literature search.

OpenAlex is free, has no rate limits, and covers 250M+ works.
https://docs.openalex.org/

Using polite pool: Requests with valid email get priority routing.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENALEX_API = "https://api.openalex.org"

# Polite pool access - OpenAlex routes these to faster servers
USER_AGENT = "KRAKEN/1.0 (mailto:trent.leslie@phenomehealth.org)"


async def search_works(
    query: str,
    limit: int = 5
) -> list[dict[str, Any]]:
    """Search OpenAlex for works matching query.

    Args:
        query: Search query (hypothesis claim or keywords)
        limit: Maximum results to return

    Returns:
        List of work objects with title, authors, year, DOI, PMID, etc.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{OPENALEX_API}/works",
                params={
                    "search": query[:200],  # Truncate long queries
                    "per_page": limit,
                    "select": "id,title,authorships,publication_year,doi,ids,cited_by_count",
                },
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            logger.info("OpenAlex search: %d results for query: %s", len(results), query[:50])
            return results
        except httpx.HTTPStatusError as e:
            logger.error("OpenAlex HTTP error: %s", e)
            return []
        except Exception as e:
            logger.error("OpenAlex search failed: %s", e)
            return []


def extract_pmid_from_work(work: dict) -> str | None:
    """Extract PMID from OpenAlex work object.

    Args:
        work: OpenAlex work object

    Returns:
        PMID string (numeric only) or None
    """
    ids = work.get("ids", {})
    if not ids:
        return None

    pmid = ids.get("pmid")
    if pmid:
        # Format is "https://pubmed.ncbi.nlm.nih.gov/12345678"
        if "/" in pmid:
            return pmid.split("/")[-1]
        return pmid

    return None


def extract_doi_from_work(work: dict) -> str | None:
    """Extract DOI from OpenAlex work object.

    Args:
        work: OpenAlex work object

    Returns:
        DOI string or None
    """
    doi = work.get("doi")
    if doi:
        # Format is "https://doi.org/10.1234/example"
        if doi.startswith("https://doi.org/"):
            return doi[16:]
        return doi
    return None


def format_authors_from_work(work: dict) -> str:
    """Format author list from OpenAlex work.

    Args:
        work: OpenAlex work object

    Returns:
        Author string like "Smith et al." or "Smith"
    """
    authorships = work.get("authorships", [])
    if not authorships:
        return "Unknown"

    first_author = authorships[0].get("author", {}).get("display_name", "Unknown")
    if len(authorships) > 1:
        return f"{first_author} et al."
    return first_author
