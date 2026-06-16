"""Async Biomapper pre-resolver wrapper for the entity_resolution node.

Maps a single ``(name, entity_type_hint)`` to a normalized resolution dict using the
``biomapper`` PyPI client over the BioMapper2 HTTP API. The module is import-safe with no
side effects; ``biomapper`` itself is **lazy-imported inside the call** so flag-off runs and
environments without the dependency/credentials are unaffected.

Responsibilities (kept here so the node stays thin):
  - map the intake ``entity_type_hint`` -> a Biolink class (the namespace/species lever);
  - gate on the confidence tier, read **via attribute** (``confidence_tier`` is a computed
    @property — ``model_dump()`` drops it);
  - apply the error taxonomy (``BioMapperAuthError`` is fatal → re-raise; everything else →
    ``None`` so the caller falls back to Kestrel);
  - validate the returned CURIE shape (BioMapper2 is a third-party service in the resolution
    trust boundary);
  - never log or echo the API key.

See docs/plans/2026-06-11-001-feat-biomapper-entity-resolution-plan.md (Unit 2).
"""

from __future__ import annotations

import logging
import re

from .config import get_settings

logger = logging.getLogger(__name__)

# intake ``entity_type_hint`` -> Biolink class passed to Biomapper as the namespace/species
# lever. Standard Biolink model classes; validate against ``client.list_entity_types()`` if they
# ever drift. A hint outside this map (or None) means "no class signal" → skip Biomapper.
_CLASS_TO_BIOLINK: dict[str, str] = {
    "gene": "biolink:Gene",
    "protein": "biolink:Protein",
    "metabolite": "biolink:SmallMolecule",
}

# Minimum confidence tier to accept a Biomapper map; below this the caller falls back to Kestrel.
# Tiers are derived by the SDK from an (unbounded) relevance score: >=2.0 high, >=1.0 medium,
# else low; None score -> "unknown".
_MIN_CONFIDENCE_TIER = "medium"
_TIER_RANK = {"unknown": -1, "low": 0, "medium": 1, "high": 2}

# CURIE shape guard: a malformed/crafted CURIE must not flow into a Kestrel query or the report.
_CURIE_RE = re.compile(r"^[A-Za-z0-9_.]+:[A-Za-z0-9._\-]+$")


def biolink_class_for(entity_type_hint: str | None) -> str | None:
    """Return the Biolink class for an intake hint, or None if there is no usable signal."""
    if not entity_type_hint:
        return None
    return _CLASS_TO_BIOLINK.get(entity_type_hint.strip().lower())


async def resolve_entity(
    name: str, entity_type_hint: str | None, base_url: str | None = None
) -> dict | None:
    """Resolve a single entity name via Biomapper, or return None to fall back to Kestrel.

    Returns a normalized dict on a confident, well-formed map:
        {"curie", "tier", "confidence", "resolved_name", "category", "xrefs"}
    Returns None when: no/unknown hint, Biomapper did not resolve, the tier is below the gate,
    the CURIE is malformed, or any non-auth Biomapper/transport error occurred.
    Raises BioMapperAuthError (fatal misconfig — never silently degrade a bad key).

    ``base_url`` overrides the configured biomapper2 endpoint per call (the prod/dev toggle);
    None falls back to Settings, then to the client default.
    """
    biolink = biolink_class_for(entity_type_hint)
    if biolink is None:
        return None  # no class signal → don't guess; caller uses Kestrel

    # Lazy import: keeps module import side-effect-free and flag-off/dep-absent paths unaffected.
    from biomapper import (  # noqa: PLC0415 — intentional lazy import
        BioMapperAuthError,
        BioMapperClient,
        BioMapperError,
    )

    settings = get_settings()
    client_kwargs: dict[str, object] = {}
    if settings.biomapper_api_key:
        client_kwargs["api_key"] = settings.biomapper_api_key
    effective_base_url = base_url or settings.biomapper_base_url
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url

    try:
        async with BioMapperClient(**client_kwargs) as client:
            result = await client.map_entity(name=name, entity_type=biolink)
    except BioMapperAuthError:
        # Bad/absent key is a deployment misconfig — fail loud, do not fall through silently.
        raise
    except BioMapperError:
        # Rate-limit / config / server / timeout (all subclass BioMapperError) → fall back.
        logger.info("FALLBACK_EVENT reason=biomapper_error name=%r class=%s", name, biolink)
        return None
    except Exception:  # noqa: BLE001 — transport/unknown: never break resolution, fall back
        logger.info("FALLBACK_EVENT reason=biomapper_transport name=%r class=%s", name, biolink)
        return None

    if not getattr(result, "resolved", False):
        return None

    # confidence_tier is a computed @property — read via attribute, NEVER via model_dump().
    tier = getattr(result, "confidence_tier", "unknown")
    if _TIER_RANK.get(tier, -1) < _TIER_RANK[_MIN_CONFIDENCE_TIER]:
        return None

    curie = getattr(result, "primary_curie", None)
    if not curie or not _CURIE_RE.match(curie):
        # Missing or malformed CURIE from a third-party service → treat as unmapped.
        return None

    xrefs = getattr(result, "kg_equivalent_ids", None) or {}
    return {
        "curie": curie,
        "tier": tier,
        "confidence": getattr(result, "confidence_score", None),
        # Biomapper has no canonical display-name field (only query_name echo + chosen_kg_id);
        # Unit 3/4 prefer the confirmed Kestrel node's name. Echo the query name as a fallback.
        "resolved_name": getattr(result, "query_name", name),
        "category": biolink,  # provisional; Unit 4 overrides with the Kestrel-native category
        "xrefs": dict(xrefs),  # dict[str, list[str]] keyed by namespace prefix
    }
