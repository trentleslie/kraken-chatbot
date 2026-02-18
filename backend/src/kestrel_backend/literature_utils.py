"""Literature utilities for PMID handling and URL generation."""

import re
from typing import Optional

PUBMED_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov"


def pmid_to_url(pmid: str) -> str:
    """Convert PMID to PubMed URL.

    Args:
        pmid: PMID in format "PMID:12345678" or just "12345678"

    Returns:
        PubMed URL like https://pubmed.ncbi.nlm.nih.gov/12345678
    """
    # Extract numeric ID
    if pmid.startswith("PMID:"):
        pmid = pmid[5:]
    pmid = pmid.strip()
    return f"{PUBMED_BASE_URL}/{pmid}"


def doi_to_url(doi: str) -> str:
    """Convert DOI to URL.

    Args:
        doi: DOI string (with or without https://doi.org/ prefix)

    Returns:
        DOI URL
    """
    if doi.startswith("http"):
        return doi
    return f"https://doi.org/{doi}"


def extract_pmid_number(pmid: str) -> str:
    """Extract numeric PMID from PMID string.

    Args:
        pmid: PMID in format "PMID:12345678" or just "12345678"

    Returns:
        Numeric PMID string
    """
    if pmid.startswith("PMID:"):
        return pmid[5:].strip()
    return pmid.strip()


def extract_pmid_from_string(text: str) -> Optional[str]:
    """Extract PMID from a string if present.

    Args:
        text: Text that may contain a PMID

    Returns:
        Numeric PMID or None
    """
    match = re.search(r'PMID:?\s*(\d{7,8})', text)
    if match:
        return match.group(1)
    return None


def format_pmid_link(pmid: str) -> str:
    """Format PMID as clickable markdown link.

    Args:
        pmid: PMID string

    Returns:
        Markdown link like [PMID:12345678](https://pubmed.ncbi.nlm.nih.gov/12345678)
    """
    pmid_num = extract_pmid_number(pmid)
    url = pmid_to_url(pmid_num)
    return f"[PMID:{pmid_num}]({url})"
