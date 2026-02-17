"""
Semantic Scholar API client for literature grounding.

Provides functions to search for papers and score their relevance to hypotheses.
Uses the S2 Academic Graph API: https://api.semanticscholar.org/
"""

import asyncio
import math
import os
import logging
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_ENDPOINT = f"{S2_API_BASE}/paper/search"

# Rate limiting configuration
# Free tier: 100 requests per 5 minutes
# With API key: 1 request per second sustained
S2_SEMAPHORE = asyncio.Semaphore(5)  # Max concurrent requests

# Check for API key (enables higher rate limits)
S2_API_KEY = os.environ.get("S2_API_KEY")
S2_DELAY = 0.3 if S2_API_KEY else 0.5  # Faster with API key


async def search_papers(
    query: str,
    limit: int = 5,
    fields: list[str] | None = None
) -> list[dict[str, Any]]:
    """
    Search Semantic Scholar for papers matching query.

    Args:
        query: Search query (hypothesis claim or keywords)
        limit: Maximum papers to return
        fields: Paper fields to retrieve

    Returns:
        List of paper dictionaries with requested fields
    """
    if fields is None:
        fields = [
            "paperId", "title", "authors", "year", "abstract",
            "citationCount", "externalIds"
        ]

    # Truncate query to 200 chars (S2 handles relevance ranking internally)
    truncated_query = query[:200] if len(query) > 200 else query

    async with S2_SEMAPHORE:
        await asyncio.sleep(S2_DELAY)  # Rate limiting

        headers = {}
        if S2_API_KEY:
            headers["x-api-key"] = S2_API_KEY

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    S2_SEARCH_ENDPOINT,
                    params={
                        "query": truncated_query,
                        "limit": limit,
                        "fields": ",".join(fields),
                    },
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except httpx.HTTPStatusError as e:
                logger.error("S2 API HTTP error: %s %s", e.response.status_code, e)
                return []
            except httpx.RequestError as e:
                logger.error("S2 API request error: %s", e)
                return []
            except Exception as e:
                logger.error("S2 search failed: %s", e)
                return []


def score_relevance(paper: dict, hypothesis_claim: str) -> float:
    """
    Score paper relevance to hypothesis (0-1).

    Uses:
    - Keyword overlap between abstract and claim (70% weight)
    - Citation count on log10 scale (30% weight) - 10k citations = 1.0

    Args:
        paper: Paper dict with abstract and citationCount
        hypothesis_claim: The hypothesis claim text

    Returns:
        Relevance score between 0 and 1
    """
    abstract = (paper.get("abstract") or "").lower()
    claim_words = set(hypothesis_claim.lower().split())
    abstract_words = set(abstract.split())

    # Keyword overlap (Jaccard-like)
    if claim_words and abstract_words:
        overlap = len(claim_words & abstract_words)
        total = len(claim_words | abstract_words)
        keyword_score = overlap / total if total > 0 else 0
    else:
        keyword_score = 0

    # Citation credibility (log10 scale, 10k citations = 1.0)
    citations = paper.get("citationCount") or 0
    citation_score = min(1.0, math.log10(citations + 1) / 4)

    # Combined score
    return 0.7 * keyword_score + 0.3 * citation_score


def classify_relationship(
    paper: dict,
    hypothesis_claim: str
) -> Literal["supporting", "contradicting", "nuancing"]:
    """
    Classify how paper relates to hypothesis.

    TODO (v2): Use cross-encoder or LLM for accurate classification.
    Keyword-based classification has high false positive rate for "contradicting",
    which is worse than no classification. For v1, return "supporting" for all.

    Args:
        paper: Paper dict with abstract
        hypothesis_claim: The hypothesis claim text

    Returns:
        Always "supporting" in v1 (see TODO above)
    """
    # v1: Return "supporting" for all papers
    # False "contradicting" labels are worse than no classification
    return "supporting"


def extract_key_passage(paper: dict, hypothesis_claim: str) -> str:
    """
    Extract most relevant sentence from abstract.

    Finds the sentence with highest keyword overlap with hypothesis claim.
    Falls back to title if no abstract.

    Args:
        paper: Paper dict with abstract and title
        hypothesis_claim: The hypothesis claim text

    Returns:
        Most relevant sentence (truncated to 300 chars)
    """
    abstract = paper.get("abstract") or ""
    if not abstract:
        return paper.get("title") or ""

    # Split into sentences
    sentences = abstract.replace(". ", ".|").split("|")
    claim_words = set(hypothesis_claim.lower().split())

    best_sentence = sentences[0] if sentences else ""
    best_overlap = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(claim_words & sentence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    # Truncate if too long
    if len(best_sentence) > 300:
        best_sentence = best_sentence[:297] + "..."

    return best_sentence.strip()


def format_authors(authors: list[dict]) -> str:
    """
    Format author list as "First Author et al." or just "First Author".

    Args:
        authors: List of author dicts with 'name' field

    Returns:
        Formatted author string
    """
    if not authors:
        return "Unknown"

    first_author = authors[0].get("name", "Unknown")
    if len(authors) > 1:
        return f"{first_author} et al."
    return first_author


def extract_doi(paper: dict) -> str | None:
    """
    Extract DOI from paper external IDs.

    Args:
        paper: Paper dict with externalIds

    Returns:
        DOI string or None
    """
    external_ids = paper.get("externalIds") or {}
    return external_ids.get("DOI")
