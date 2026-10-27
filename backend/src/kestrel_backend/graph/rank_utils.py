"""Rank inference for the entity-resolution rank-collapse guard (Axis D, Guard 1).

v1 is **taxa-only**. A label's requested taxonomic rank is derived purely from its *shape*
(binomial → species; single capitalized token → genus; ``-aceae`` → family; ``-ales`` → order).
Non-taxa labels (metabolites, genes, diseases) and ambiguous/garbage input return ``None`` so the
guard stays inert and the resolution path is byte-identical. No "leaf-vs-broad ontology specificity"
logic in v1 — deferred until a concrete non-taxa rank-collapse case is documented.

Detection is **name-derived**: the same ``infer_requested_rank`` function is applied to both the
requested label and the resolved node's name. Kestrel's node dicts carry no taxonomic lineage/rank
metadata (grep confirms zero ``taxon``/``lineage``/``rank`` handling in ``kestrel_backend``), so a
rank collapse is inferred by comparing the two name-derived ranks plus a leading-token (genus) match.
"""

from __future__ import annotations

import re
from enum import IntEnum


class Rank(IntEnum):
    """Ordered taxonomic ranks; **larger == finer** so "requested finer than resolved" is ``>``."""

    ORDER = 1
    FAMILY = 2
    GENUS = 3
    SPECIES = 4


# A single taxon token: a Titlecase alphabetic word (leading upper, remainder lower). This excludes
# all-caps gene symbols (APOE), mixed-case/alnum symbols (KIF6, IL6), and lowercase common names.
_TAXON_TOKEN = re.compile(r"^[A-Z][a-z]+$")


def _is_taxon_token(tok: str) -> bool:
    return bool(_TAXON_TOKEN.match(tok))


def infer_requested_rank(label: str | None) -> Rank | None:
    """Infer the taxonomic rank a label *requests* from its shape, or ``None`` when not taxa.

    - ``Genus species`` (Titlecase + lowercase, both alphabetic) → ``SPECIES``
    - single Titlecase token ending ``-aceae`` → ``FAMILY``
    - single Titlecase token ending ``-ales`` → ``ORDER``
    - any other single Titlecase token → ``GENUS``
    - everything else (metabolites, genes, diseases, empty, garbage) → ``None``
    """
    if not label:
        return None
    text = label.strip()
    if not text:
        return None
    tokens = text.split()

    if len(tokens) == 2:
        genus, species = tokens
        if _is_taxon_token(genus) and species.isalpha() and species.islower():
            return Rank.SPECIES
        return None

    if len(tokens) == 1:
        tok = tokens[0]
        if not _is_taxon_token(tok):
            return None
        low = tok.lower()
        if low.endswith("aceae"):
            return Rank.FAMILY
        if low.endswith("ales"):
            return Rank.ORDER
        return Rank.GENUS

    return None


def is_rank_collapse(requested_label: str | None, resolved_name: str | None) -> bool:
    """True when ``resolved_name`` is a *proper ancestor* (coarser rank) of ``requested_label``.

    A rank collapse requires all of:
      1. both names yield a taxonomic rank (guard inert otherwise),
      2. the requested rank is **finer** than the resolved-name rank, and
      3. a leading-token (genus) match — the requested and resolved names share their first token
         (case-insensitive). This distinguishes a genuine rank collapse
         (``Ruminococcus gnavus`` → ``Ruminococcus``) from an unrelated mis-resolution
         (``Ruminococcus gnavus`` → ``Blautia``), which is a different concern.
    """
    if not requested_label or not resolved_name:
        return False
    req = infer_requested_rank(requested_label)
    res = infer_requested_rank(resolved_name)
    if req is None or res is None:
        return False
    if req <= res:  # requested not strictly finer than resolved
        return False
    req_first = requested_label.strip().split()[0].lower()
    res_first = resolved_name.strip().split()[0].lower()
    return req_first == res_first
