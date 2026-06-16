"""Per-leg co-occurrence retrieval for the bridge_grounding scorer (U2).

For each leg (A–B, B–C) we retrieve a relevance-sorted PubMed co-occurrence pool by ANDing the
two entity names, capped at a per-leg limit. Co-mention PMIDs (in both legs) are kept-first into
one leg (matching skimgpt's shared-seen dedup), never dropped or double-counted.
"""

import logging
import re

from ..pubmed_client import _ncbi_api_key, search_pmids

logger = logging.getLogger(__name__)

# A bare CURIE looks like "NCBIGene:7132" / "CHEBI:4167": a prefix token followed by a colon.
# The KG name lookup falls back to the raw CURIE when it omits a human label (see plan O4);
# we must NOT PubMed-search a CURIE (it returns junk), so treat such "names" as missing.
_CURIE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]*:\S")

# Hard ceiling on PMIDs retrieved per leg (cost guard).
MAX_RETMAX = 50


def is_curie_like(name: str) -> bool:
    """True if ``name`` looks like a bare CURIE rather than a human-readable entity name."""
    return bool(name) and bool(_CURIE_RE.match(name.strip()))


def cooccurrence_query(name_x: str, name_y: str) -> str:
    """Quoted AND co-occurrence query for two entity names."""
    return f'"{name_x}" AND "{name_y}"'


async def cooccurrence_pmids(name_x: str, name_y: str, limit: int = MAX_RETMAX) -> list[str]:
    """PMIDs co-mentioning both names (relevance-sorted, capped at ``min(limit, MAX_RETMAX)``).

    Returns [] when either name is missing or CURIE-like (no usable human name → the leg
    will be floored to insufficient_literature upstream).
    """
    if not name_x or not name_y or is_curie_like(name_x) or is_curie_like(name_y):
        return []
    if not _ncbi_api_key():
        logger.warning(
            "bridge_grounding retrieval: NCBI_API_KEY unset — slower rate (0.35s/req); "
            "set NCBI_API_KEY for 10 req/s")
    return await search_pmids(cooccurrence_query(name_x, name_y), retmax=min(limit, MAX_RETMAX))


def dedupe_co_mention(
    pool_a: list[str], pool_b: list[str]
) -> tuple[list[str], list[str], int]:
    """Keep-first co-mention dedup: PMIDs present in both pools stay in A and are removed from B.

    Matches skimgpt's shared-``seen_pmids`` behavior (kept once, not dropped from both, not
    double-counted). Returns ``(a_kept, b_kept, dropped_from_b)``.
    """
    a_set = set(pool_a)
    b_kept = [p for p in pool_b if p not in a_set]
    dropped = len(pool_b) - len(b_kept)
    return list(pool_a), b_kept, dropped
