"""Exa AI client for semantic literature search.

Exa uses neural search to find semantically relevant content.
Docs: https://exa.ai/docs/reference/search
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

EXA_API = "https://api.exa.ai/search"
EXA_API_KEY = os.getenv("EXA_API_KEY", "")


class ExaSearchError(Exception):
    """Exa API error."""
    pass


async def search_papers(
    query: str,
    limit: int = 5
) -> list[dict[str, Any]]:
    """
    Search Exa for research papers matching query.

    Args:
        query: Search query (hypothesis or keywords)
        limit: Maximum results to return

    Returns:
        List of paper objects with title, url, publishedDate, highlights
    """
    if not EXA_API_KEY:
        logger.debug("EXA_API_KEY not set, skipping Exa search")
        return []

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                EXA_API,
                headers={"x-api-key": EXA_API_KEY},
                json={
                    "query": query[:500],
                    "type": "neural",
                    "category": "research paper",
                    "numResults": limit,
                    "contents": {
                        "highlights": True,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            logger.info("Exa search: %d results for query: %s", len(results), query[:50])
            return results
        except httpx.HTTPStatusError as e:
            logger.error("Exa HTTP error: %s", e)
            raise ExaSearchError(f"Exa API error {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            logger.error("Exa search failed: %s", e)
            return []


def extract_doi_from_url(url: str) -> str | None:
    """Extract DOI from URL if present."""
    if not url:
        return None
    if "doi.org/" in url:
        return url.split("doi.org/")[-1]
    return None


def extract_year_from_date(date_str: str | None) -> int:
    """Extract year from ISO date string."""
    if date_str and len(date_str) >= 4:
        try:
            return int(date_str[:4])
        except ValueError:
            pass
    return 0
