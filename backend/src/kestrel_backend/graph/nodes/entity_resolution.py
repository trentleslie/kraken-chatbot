"""
Entity Resolution Node: Two-tier entity name resolution using Kestrel KG.

This node resolves raw entity names to knowledge graph identifiers (CURIEs)
using a two-tier approach:

Tier 1 (API): Direct hybrid_search calls via Kestrel HTTP client (~100ms each)
  - Fast, reliable - uses Kestrel's ranking which is typically accurate
  - Top result with score > 0.6 is accepted

Tier 1.5 (Alias): For Tier 1 failures, try known aliases before falling back to LLM
  - Uses entity_aliases extracted by intake node
  - Avoids expensive LLM calls for entities with known alternative names

Tier 2 (LLM): Falls back to Claude Agent SDK for remaining ambiguous cases
  - Handles complex synonyms, abbreviations, partial matches
  - More expensive but can reason about alternatives

This hybrid approach optimizes for speed while maintaining accuracy.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ...biomapper_client import resolve_entity as biomapper_resolve, biolink_class_for
from ...config import get_settings, resolve_biomapper_base_url
from ..state import DiscoveryState, EntityResolution
from ..sdk_utils import HAS_SDK, query_with_usage, ClaudeAgentOptions, chunk
from ..pipeline_config import get_pipeline_config
from ..state_contracts import validate_state, EntityResolutionInput, EntityResolutionOutput

logger = logging.getLogger(__name__)

_config = get_pipeline_config().entity_resolution

# Semaphore to serialize SDK calls and prevent concurrent CLI spawn issues
SDK_SEMAPHORE = asyncio.Semaphore(_config.sdk_semaphore)

# Selection prompt (#61): the broken stdio MCP search tools are gone. We now do the
# variant searches over HTTP ourselves and hand the model a CANDIDATE list to SELECT
# from — it no longer searches the KG.
RESOLUTION_PROMPT = """You are an expert biomedical entity resolver for the Kestrel knowledge graph.

## Your Task
You are given an entity name and a list of CANDIDATE nodes already retrieved from the
knowledge graph. SELECT the single best candidate CURIE for the entity, or return null
if none of the candidates actually refer to the entity.

## Matching guidance
- Match on meaning, not just string equality: synonyms, spelling/hyphenation variants,
  gene symbols, and chemical-name variants all count.
  - "N-lactoyl phenylalanine" matches a "lactoylphenylalanine" node
  - "16-hydroxypalmitate" matches "16-hydroxypalmitic acid"
  - A gene symbol (all caps, 2-6 chars, e.g. KIF6) matches its NCBIGene node
- Prefer the most specific, highest-scoring candidate that genuinely refers to the entity.

## Hard rules
- You MUST pick a `curie` value from the provided candidates verbatim, or return null.
- Do NOT invent, complete, or correct a CURIE. If nothing fits, return null — never guess.

## Output Format
Return ONLY valid JSON (no other text):
{"curie": "PREFIX:ID", "confidence": 0.95}

If no candidate fits:
{"curie": null, "confidence": 0.0}"""

# Retry uses the same selection task with a more permissive matching posture (the entity
# already failed the score threshold, so accept good synonym/variant matches).
RETRY_PROMPT = """You are an expert biomedical entity resolver. This entity was hard to resolve.

You are given an entity name and a list of CANDIDATE nodes retrieved from the knowledge
graph via several spelling/synonym variant searches. SELECT the single best candidate, being
generous about synonym, hyphenation, numeric-prefix, IUPAC/common-name, and gene-symbol
variants — but only if the candidate genuinely refers to the same entity.

## Hard rules
- You MUST pick a `curie` value from the provided candidates verbatim, or return null.
- Do NOT invent, complete, or correct a CURIE. If nothing genuinely matches, return null.

## Output
Return ONLY valid JSON:
{"curie": "PREFIX:ID", "confidence": 0.95}

If no candidate fits: {"curie": null, "confidence": 0.0}"""



async def resolve_via_api(
    entity: str, category: str | None = None
) -> EntityResolution | None:
    """
    Tier 1: Attempt to resolve entity via direct Kestrel API call.

    Uses hybrid_search and takes the top-scored result if confidence is high enough.
    Returns None if resolution fails or confidence is too low (triggering Tier 2).

    ``category`` (a Biolink class, e.g. "biolink:Disease") constrains hybrid_search to that
    class so a same-text node from another namespace cannot win (the Tier 1 wrong-namespace
    fix). When set: an in-category hit at/below ``tier1_min_score`` or an empty result returns
    None (routes to Tier 2); an ``isError`` falls back to the unconstrained call at
    ``tier1_fallback_confidence``. When None, behavior is byte-identical to the legacy path.

    Confidence mapping from hybrid_search score:
    - score > 1.5 → confidence 0.95 (exact + vector match)
    - score > 1.0 → confidence 0.90
    - score > 0.8 → confidence 0.80
    - score > 0.6 → confidence 0.70
    - score < 0.6 → fall through to Tier 2 (returns None)
    """
    try:
        # Call hybrid_search directly - parameter is 'search_text' not 'query'
        search_args: dict = {
            "search_text": entity,
            "limit": 1,  # Only need top result
        }
        if category is not None:
            # Constrain to the expected Biolink class. Matching is list-membership (a node
            # matches if the class is anywhere in its categories array) — verified live 2026-06-17.
            search_args["category"] = category
        result = await call_kestrel_tool("hybrid_search", search_args)

        # Debug logging - show raw API response
        is_error = result.get("isError", False)
        content = result.get("content", [])
        logger.info(
            "Tier 1 '%s': isError=%s, content_len=%d",
            entity, is_error, len(content)
        )

        if is_error:
            if category is not None:
                # The category-filter mechanism itself failed (e.g. a future arg drift). Degrade
                # gracefully to today's unconstrained behavior, flagged at a reduced confidence so
                # the degraded path is visible. (R5; mirrors the direction-param cold-start lesson.)
                logger.info(
                    "FALLBACK_EVENT node=entity_resolution reason=category_iserror entity=%s",
                    entity,
                )
                fallback = await resolve_via_api(entity, category=None)
                if fallback is None:
                    return None
                return fallback.model_copy(update={
                    "confidence": _config.tier1_fallback_confidence,
                    "method": "category-fallback",
                })
            logger.debug("Tier 1 '%s': API returned error", entity)
            return None

        # Parse the search results
        if not content:
            logger.info("Tier 1 '%s': No content in response", entity)
            return None

        # Extract JSON from content
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.info("Tier 1 '%s': Could not parse JSON response", entity)
            return None

        # Response format is {search_text: [results]} - extract our entity's results
        if isinstance(data, dict):
            # Try exact key match first, then case-insensitive
            results = data.get(entity) or data.get(entity.lower()) or []
            if not results and len(data) == 1:
                # Single result dict - take the first (and only) value
                results = list(data.values())[0]
        else:
            results = data

        if not results:
            logger.info("Tier 1 '%s': No search results", entity)
            return None

        # Take top result
        top = results[0]
        score = float(top.get("score", 0))
        curie = top.get("id") or top.get("curie")
        name = top.get("name") or top.get("label")
        # categories is a list in the API response
        categories = top.get("categories", [])
        category = categories[0] if categories else top.get("category")

        # Map score to confidence
        if score > 1.5:
            confidence = 0.95
        elif score > 1.0:
            confidence = 0.90
        elif score > 0.8:
            confidence = 0.80
        elif score > _config.tier1_min_score:
            confidence = 0.70
        else:
            # Score too low - fall through to Tier 2
            logger.info(
                "Tier 1 '%s': Score %.2f below threshold %.2f, falling through to Tier 2",
                entity, score, _config.tier1_min_score
            )
            return None

        # Map confidence to method (same logic as LLM parser)
        if confidence >= 0.9:
            method = "exact"
        elif confidence >= 0.7:
            method = "fuzzy"
        else:
            method = "semantic"

        logger.info(
            "Tier 1 '%s': resolved to %s (score=%.2f, confidence=%.2f, method=%s)",
            entity, curie, score, confidence, method
        )

        return EntityResolution(
            raw_name=entity,
            curie=curie,
            resolved_name=name,
            category=category,
            confidence=confidence,
            method=method,
        )

    except Exception as e:
        logger.warning("Tier 1 '%s': Exception - %s", entity, str(e))
        return None


def parse_resolution_result(entity: str, result_text: str) -> EntityResolution:
    """
    Parse LLM response into EntityResolution object.

    Handles JSON extraction from potentially noisy LLM output.
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]+\}', result_text)
        if json_match:
            data = json.loads(json_match.group())

            curie = data.get("curie")
            name = data.get("name")
            category = data.get("category")
            confidence = float(data.get("confidence", 0.0))

            # Determine resolution method based on confidence
            if curie and confidence >= 0.9:
                method = "exact"
            elif curie and confidence >= 0.7:
                method = "fuzzy"
            elif curie:
                method = "semantic"
            else:
                method = "failed"

            return EntityResolution(
                raw_name=entity,
                curie=curie,
                resolved_name=name,
                category=category,
                confidence=confidence,
                method=method,
            )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        pass

    # Fallback for parse failures
    return EntityResolution(
        raw_name=entity,
        curie=None,
        resolved_name=None,
        category=None,
        confidence=0.0,
        method="failed",
    )


# Max candidates shown to the selector (bounds prompt size; membership is checked
# against exactly this shown set).
_CANDIDATE_CAP = 25


def _resolution_variants(entity: str) -> list[str]:
    """Spelling/synonym variants to broaden the candidate search.

    Approximates (server-side) the variant reformulation the old max_turns=5 SDK loop
    did: raw + de-hyphenated + hyphen-removed + numeric/short-prefix-stripped +
    gene-symbol form. NOTE: this is a fixed, finite list — a strict subset of the live
    loop's open-ended reformulation — so recall must be gated against a hand-labeled
    fixture before merge (see plan #61).
    """
    variants: list[str] = []

    def add(v: str) -> None:
        v = v.strip()
        if v and v not in variants:
            variants.append(v)

    add(entity)
    add(entity.replace("-", " "))   # de-hyphenated
    add(entity.replace("-", ""))    # hyphen-removed
    # strip a leading "N-"/"16-"/"3-" prefix, but skip degenerate stubs (e.g. "IL-6" -> "6")
    # that would just burn a result slot with noise — the raw name is always searched anyway.
    stripped = re.sub(r"^[0-9A-Za-z]{1,3}-", "", entity)
    if len(stripped) > 3:
        add(stripped)
    if 2 <= len(entity) <= 6 and entity.isalnum():
        add(entity.upper())         # gene-symbol form
    return variants


def _canonical_curie(curie: str | None) -> str | None:
    """Canonical form for the R1a membership check: uppercase prefix + trim.

    Fuzzy/equivalent-namespace matching is intentionally NOT done — a lenient match
    could admit a different node that merely normalizes alike.
    """
    if not curie:
        return None
    c = curie.strip()
    if ":" in c:
        prefix, _, local = c.partition(":")
        return f"{prefix.upper()}:{local}"
    return c.upper()


def _extract_search_results(data: Any, search_text: str) -> list:
    """Pull the results list out of a Kestrel ``{search_text: [results]}`` envelope.

    Same envelope hybrid_search and text_search return (both route through
    call_kestrel_tool; see resolve_via_api).
    """
    if isinstance(data, dict):
        results = data.get(search_text) or data.get(search_text.lower()) or []
        if not results and len(data) == 1:
            results = list(data.values())[0]
        return results or []
    return data or []


# ============================ Biomapper pre-resolver helpers (Unit 3) ============================
# HGNC is human-only by construction; an HGNC equivalent on a gene/protein KG node is the human
# marker. biomapper2 now prefers the human candidate upstream (PR #69), so this gate is
# defense-in-depth: it catches any residual wrong-species CURIE before it enters the pipeline.
_HGNC_MARKER = "HGNC:"
_HGNC_GATED_CLASSES = {"gene", "protein"}


def _tier_to_confidence(tier: str | None) -> float:
    """Map a Biomapper confidence tier to the EntityResolution 0–1 confidence (provenance-cosmetic)."""
    return {"high": 0.95, "medium": 0.8}.get(tier or "", 0.7)


def _parse_get_nodes(result: Any, curie: str) -> dict | None:
    """Return the Kestrel node dict for `curie` from a get_nodes MCP envelope, or None.

    Envelope (confirmed live 2026-06-16): present → ``{"<curie>": {node...}}`` (top-level key IS the
    queried CURIE; value is the node **dict** — some tool versions wrap it as a single-element list,
    so both are accepted); invalid prefix → ``{"error": true, ...}``; absent → empty value.
    """
    if not isinstance(result, dict) or result.get("isError"):
        return None
    content = result.get("content", [])
    if not content:
        return None
    text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(body, dict) or body.get("error"):
        return None
    payload = body.get(curie)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None


def _node_has_human_marker(node: dict) -> bool:
    """True if the Kestrel node carries an HGNC equivalent id (human-only marker)."""
    eq = node.get("equivalent_ids") or []
    return any(str(x).upper().startswith(_HGNC_MARKER) for x in eq)


def _node_category(node: dict) -> str | None:
    """The node's Kestrel-native category (so the biomapper path emits Kestrel spellings)."""
    cats = node.get("categories")
    if isinstance(cats, list) and cats:
        return cats[0]
    return node.get("category")


def _biomapper_candidate_curies(biomapper_result: dict, hint: str | None) -> list[str]:
    """Ordered CURIE candidates: primary_curie first, then xrefs by per-class namespace_preference.

    ``xrefs`` is ``dict[prefix -> list[local_id]]`` (verified live: ``{"NCBIGene": ["7132"], ...}``),
    so a candidate CURIE is built as ``f"{prefix}:{local_id}"``.
    """
    prefs = get_pipeline_config().entity_resolution.biomapper.namespace_preference.get(
        (hint or "").lower(), []
    )
    xrefs: dict[str, list] = biomapper_result.get("xrefs") or {}
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(curie: str | None) -> None:
        canon = _canonical_curie(curie)
        if curie and canon and canon not in seen:
            seen.add(canon)
            ordered.append(curie)

    _add(biomapper_result.get("curie"))
    # xrefs in configured preference order, then any remaining namespaces (lowest priority).
    pref_upper = [p.upper() for p in prefs]
    for ns in sorted(xrefs, key=lambda n: pref_upper.index(n.upper()) if n.upper() in pref_upper else 999):
        for local in (xrefs[ns] if isinstance(xrefs[ns], list) else [xrefs[ns]]):
            local_s = str(local)
            _add(local_s if ":" in local_s else f"{ns}:{local_s}")
    return ordered


async def reconcile_to_kestrel(
    biomapper_result: dict, hint: str | None
) -> tuple[str, str | None] | None:
    """Confirm a Biomapper result against the Kestrel KG; return (confirmed_curie, kestrel_category).

    Walks the candidate CURIEs (primary first, then namespace-preferred xrefs), accepting the first
    that ``get_nodes`` confirms. For gene/protein, the confirmed node must carry the HGNC human
    marker (defense-in-depth) or the candidate is skipped. Returns the node's canonical id +
    Kestrel-native category, or None if nothing confirms (caller falls back to Kestrel tiers).
    """
    gated = (hint or "").lower() in _HGNC_GATED_CLASSES
    for candidate in _biomapper_candidate_curies(biomapper_result, hint):
        try:
            result = await call_kestrel_tool("get_nodes", {"curies": candidate})
        except Exception as e:  # noqa: BLE001 — transport failure for this candidate; try next
            logger.debug("Biomapper reconcile: get_nodes('%s') failed: %s", candidate, e)
            continue
        node = _parse_get_nodes(result, candidate)
        if node is None:
            continue
        if gated and not _node_has_human_marker(node):
            # Confirmed in Kestrel but no HGNC marker → non-human ortholog; reject (defense-in-depth).
            logger.info("FALLBACK_EVENT node=entity_resolution reason=biomapper_non_human curie=%s", candidate)
            continue
        return node.get("id") or candidate, _node_category(node)
    return None


async def prefetch_resolution_candidates(entity: str, limit: int = 10) -> list[dict]:
    """Tier-2 prefetch (#61): issue multiple HTTP variant searches and dedup into a
    candidate set. Replaces the broken MCP variant-search loop.

    Each variant is searched via BOTH hybrid_search and text_search (``limit > 1``);
    results are deduped by canonical CURIE, keeping the highest-scoring instance.
    Returns ``[{curie, name, category, score}]`` sorted best-score first.
    """
    candidates: dict[str, dict] = {}  # canonical curie -> candidate
    for variant in _resolution_variants(entity):
        for tool in ("hybrid_search", "text_search"):
            try:
                result = await call_kestrel_tool(tool, {"search_text": variant, "limit": limit})
            except Exception as e:
                logger.debug("Prefetch '%s' via %s failed: %s", variant, tool, e)
                continue
            if result.get("isError"):
                continue
            content = result.get("content", [])
            if not content:
                continue
            text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            for row in _extract_search_results(data, variant):
                if not isinstance(row, dict):
                    continue
                curie = row.get("id") or row.get("curie")
                canon = _canonical_curie(curie)
                if not canon:
                    continue
                score = float(row.get("score", 0) or 0)
                cats = row.get("categories", [])
                existing = candidates.get(canon)
                if existing is None or score > existing["score"]:
                    candidates[canon] = {
                        "curie": curie,
                        "name": row.get("name") or row.get("label"),
                        "category": cats[0] if cats else row.get("category"),
                        "score": score,
                    }
    return sorted(candidates.values(), key=lambda c: c["score"], reverse=True)


async def resolve_single_entity(entity: str, is_retry: bool = False) -> tuple[EntityResolution, Any]:
    """
    Tier 2 (#61): resolve via HTTP candidate prefetch + SDK selection.

    Replaces the broken stdio-MCP variant-search loop: we issue the variant searches
    over HTTP ourselves (``prefetch_resolution_candidates``), then ask the SDK — with
    NO tools — to SELECT the best candidate.

    Correctness (R1a): the returned CURIE is validated against the prefetched candidate
    set (exact canonical match); a CURIE not in the set → method="failed" (never a
    fabricated CURIE). Empty candidate set or transport failure → method="failed"
    WITHOUT invoking the SDK. On a valid selection we surface the CANDIDATE's own
    curie/name/category — never the model's emitted strings.

    Returns tuple of (EntityResolution, ModelUsageRecord | None).
    """
    failed = EntityResolution(
        raw_name=entity, curie=None, resolved_name=None,
        category=None, confidence=0.0, method="failed",
    )

    # Without the SDK we can't run the selection step, so fail fast before spending
    # any HTTP round-trips on the prefetch.
    if not HAS_SDK:
        return (failed, None)

    # HTTP prefetch — variant searches build the candidate set.
    candidates = await prefetch_resolution_candidates(entity)
    if not candidates:
        # No real data → honest failure (no fabrication, no SDK call).
        return (failed, None)

    shown = candidates[:_CANDIDATE_CAP]
    by_canon = {_canonical_curie(c["curie"]): c for c in shown}
    candidate_lines = "\n".join(
        f'- curie={c["curie"]} | name={c["name"]!r} | category={c["category"]}'
        for c in shown
    )
    select_prompt = (
        f"Entity to resolve: {entity}\n\n"
        f"Candidates (choose the best `curie` verbatim, or null):\n{candidate_lines}"
    )

    try:
        async with SDK_SEMAPHORE:
            options = ClaudeAgentOptions(
                system_prompt=RETRY_PROMPT if is_retry else RESOLUTION_PROMPT,
                allowed_tools=[],  # data-in-prompt selection; no KG tools
                max_turns=1,
                permission_mode="bypassPermissions",
                max_buffer_size=10 * 1024 * 1024,
            )
            result_text, usage_record = await query_with_usage(
                prompt=select_prompt,
                options=options,
                node_name="entity_resolution",
            )
    except Exception as e:
        logger.warning("Tier 2 select for '%s' failed: %s", entity, str(e))
        return (failed, None)

    # Parse the model's selection (curie + confidence → method).
    parsed = parse_resolution_result(entity, result_text)
    chosen = by_canon.get(_canonical_curie(parsed.curie)) if parsed.curie else None
    if chosen is None:
        # R1a: model returned null OR a CURIE not in the candidate set → honest failure.
        if parsed.curie:
            logger.warning(
                "Tier 2 '%s': model selected out-of-candidate CURIE %s — rejected (R1a)",
                entity, parsed.curie,
            )
        return (failed, usage_record)

    # Surface the CANDIDATE's own fields (never the model's emitted strings).
    return (EntityResolution(
        raw_name=entity,
        curie=chosen["curie"],
        resolved_name=chosen["name"],
        category=chosen["category"],
        confidence=parsed.confidence,
        method=parsed.method,
    ), usage_record)



@validate_state(EntityResolutionInput, EntityResolutionOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Resolve all raw entities to knowledge graph identifiers.

    Implements three-tier resolution:

    Tier 1 (API): Try direct hybrid_search for all entities in parallel
      - Fast (~100ms each), uses Kestrel's reliable ranking
      - Accepts top result if score > 0.6

    Tier 1.5 (Alias): For Tier 1 failures, try known aliases from intake
      - Uses entity_aliases dict to find alternative names
      - Avoids expensive LLM calls for entities with known aliases

    Tier 2 (LLM): HTTP prefetch + SDK selection for remaining failed entities (#61)
      - Multi-variant HTTP searches (hybrid_search + text_search) build a candidate set
      - SDK selects the best candidate from the list (allowed_tools=[], max_turns=1)
      - R1a: the returned CURIE must be a member of the prefetched candidate set

    Returns resolved_entities list and any errors encountered.
    """
    entities = state.get("raw_entities", [])
    aliases = state.get("entity_aliases", {})  # NEW: Get alias mappings from intake
    logger.info("Starting entity_resolution with %d entities, %d alias mappings",
                len(entities), len(aliases))
    start = time.time()

    if not entities:
        logger.info("No entities to resolve, skipping")
        return {
            "resolved_entities": [],
            "errors": [],
        }

    all_results: list[EntityResolution | None] = [None] * len(entities)
    errors: list[str] = []

    # ========== BIOMAPPER PRE-RESOLVER (flag-gated; Unit 4) ==========
    # Read the flag fresh (not the import-time _config) so get_pipeline_config.cache_clear() in
    # tests takes effect. Flag-off → this whole block is skipped and behavior is byte-identical.
    biomapper_cfg = get_pipeline_config().entity_resolution.biomapper
    entity_type_hints: dict[str, str] = state.get("entity_type_hints", {})
    biomapper_confirmed: set[int] = set()
    biomapper_resolved = 0
    if biomapper_cfg.enabled:
        # Resolve the prod/dev biomapper2 endpoint from the per-run toggle (state["biomapper_env"]).
        # An invalid/unconfigured env → log and skip biomapper this run (safe fall back to Kestrel),
        # rather than failing the whole discovery run.
        try:
            biomapper_base_url = resolve_biomapper_base_url(state.get("biomapper_env"), get_settings())
            env_ok = True
        except ValueError as e:
            logger.warning("Biomapper env toggle error (%s); skipping biomapper, using Kestrel", e)
            biomapper_base_url = None
            env_ok = False
        # Only attempt entities whose intake hint maps to a Biolink class (else fall back to Kestrel).
        targets = [
            (i, e) for i, e in enumerate(entities)
            if env_ok and biolink_class_for(entity_type_hints.get(e)) is not None
        ]
        if targets:
            sem = asyncio.Semaphore(max(1, biomapper_cfg.http_concurrency))
            node_timeout = biomapper_cfg.node_timeout_seconds

            async def _biomapper_one(idx: int, name: str) -> tuple[int, EntityResolution | None]:
                hint = entity_type_hints.get(name)
                async with sem:
                    # BioMapperAuthError intentionally propagates out of run() (fail loud, R6).
                    try:
                        r = await asyncio.wait_for(
                            biomapper_resolve(name, hint, base_url=biomapper_base_url),
                            timeout=node_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.info(
                            "FALLBACK_EVENT node=entity_resolution reason=biomapper_timeout entity=%s",
                            name,
                        )
                        return idx, None
                if r is None:
                    logger.info(
                        "FALLBACK_EVENT node=entity_resolution reason=biomapper_miss entity=%s", name
                    )
                    return idx, None
                reconciled = await reconcile_to_kestrel(r, hint)
                if reconciled is None:
                    logger.info(
                        "FALLBACK_EVENT node=entity_resolution reason=biomapper_unconfirmed entity=%s",
                        name,
                    )
                    return idx, None
                curie, category = reconciled
                return idx, EntityResolution(
                    raw_name=name,
                    curie=curie,
                    resolved_name=r.get("resolved_name") or name,
                    category=category,
                    confidence=_tier_to_confidence(r.get("tier")),
                    method="biomapper",
                )

            prepass = await asyncio.gather(*[_biomapper_one(i, e) for (i, e) in targets])
            for idx, res in prepass:
                if res is not None:
                    all_results[idx] = res
                    biomapper_confirmed.add(idx)
                    biomapper_resolved += 1
            logger.info(
                "Biomapper pre-resolver confirmed %d/%d hinted entities",
                biomapper_resolved, len(targets),
            )

    # ========== TIER 1: API Resolution (skips Biomapper-confirmed indices) ==========
    tier1_start = time.time()
    tier1_targets = [(i, e) for i, e in enumerate(entities) if i not in biomapper_confirmed]
    logger.info("Tier 1 (API): Attempting direct resolution for %d entities", len(tier1_targets))

    # Run all API calls in parallel (they're fast and independent). Pass the Biolink category
    # the intake hint maps to (None when no/unknown hint) so hybrid_search resolves to the right
    # namespace instead of a same-text node from another category (Tier 1 wrong-namespace fix).
    tier1_results = await asyncio.gather(
        *[
            resolve_via_api(e, category=biolink_class_for(entity_type_hints.get(e)))
            for (_i, e) in tier1_targets
        ],
        return_exceptions=True,
    )

    tier1_resolved = 0
    tier1_failed_indices = []

    for (i, entity), result in zip(tier1_targets, tier1_results):
        if isinstance(result, Exception):
            logger.debug("Tier 1 '%s': Exception - %s", entity, str(result))
            tier1_failed_indices.append(i)
        elif result is not None:
            # Successfully resolved via API
            all_results[i] = result
            tier1_resolved += 1
        else:
            # Returned None - needs Tier 1.5 or Tier 2
            tier1_failed_indices.append(i)

    tier1_duration = time.time() - tier1_start
    logger.info(
        "Tier 1 (API) resolved %d/%d entities in %.1fs",
        tier1_resolved, len(entities), tier1_duration
    )

    # ========== TIER 1.5: ALIAS Resolution ==========
    tier15_start = time.time()
    tier15_resolved = 0
    tier2_needed_indices = []

    for idx in tier1_failed_indices:
        entity = entities[idx]
        
        # Check if this entity has known aliases
        if entity in aliases:
            entity_aliases = aliases[entity]
            logger.info("Tier 1.5: Trying %d aliases for '%s': %s",
                       len(entity_aliases), entity, entity_aliases)
            
            resolved_via_alias = False
            # Constrain the alias lookup to the PARENT entity's category (hints are keyed on the
            # original name, not the alias). A type-mismatched alias yields an empty in-category
            # result and routes to Tier 2, exactly as an over-fired hint does.
            alias_category = biolink_class_for(entity_type_hints.get(entity))
            for alias in entity_aliases:
                alias_result = await resolve_via_api(alias, category=alias_category)
                if alias_result is not None:
                    # Use alias resolution but keep original raw_name
                    all_results[idx] = EntityResolution(
                        raw_name=entity,  # Keep original name
                        curie=alias_result.curie,
                        resolved_name=alias_result.resolved_name,
                        category=alias_result.category,
                        confidence=alias_result.confidence,
                        method=f"alias:{alias}",  # Track that alias was used
                    )
                    tier15_resolved += 1
                    resolved_via_alias = True
                    logger.info("Tier 1.5 '%s': resolved via alias '%s' to %s",
                               entity, alias, alias_result.curie)
                    break
            
            if not resolved_via_alias:
                tier2_needed_indices.append(idx)
        else:
            # No aliases for this entity
            tier2_needed_indices.append(idx)

    tier15_duration = time.time() - tier15_start
    if tier1_failed_indices:
        logger.info(
            "Tier 1.5 (Alias) resolved %d/%d failed entities in %.1fs",
            tier15_resolved, len(tier1_failed_indices), tier15_duration
        )

    # ========== TIER 2: LLM Resolution ==========
    model_usages: list = []
    if tier2_needed_indices and HAS_SDK:
        tier2_start = time.time()
        failed_entities = [entities[i] for i in tier2_needed_indices]
        logger.info(
            "Tier 2 (LLM): Processing %d entities that failed Tier 1 and 1.5",
            len(failed_entities)
        )
        for idx in tier2_needed_indices:
            logger.info(
                "FALLBACK_EVENT node=entity_resolution entity=%s reason=tier1_failed tier=2",
                entities[idx],
            )

        # First pass: Standard resolution in batches
        tier2_results = []
        for batch in chunk(failed_entities, _config.batch_size):
            batch_results = await asyncio.gather(
                *[resolve_single_entity(e) for e in batch],
                return_exceptions=True,
            )
            tier2_results.extend(batch_results)

        # Map results back to all_results
        for idx, result in zip(tier2_needed_indices, tier2_results):
            if isinstance(result, Exception):
                errors.append(f"Resolution failed for '{entities[idx]}': {str(result)}")
                all_results[idx] = EntityResolution(
                    raw_name=entities[idx],
                    curie=None,
                    resolved_name=None,
                    category=None,
                    confidence=0.0,
                    method="failed",
                )
            else:
                resolution, usage_record = result
                all_results[idx] = resolution
                if usage_record is not None:
                    model_usages.append(usage_record)

        # Second pass: Aggressive retry for still-failed entities
        still_failed_indices = [i for i in tier2_needed_indices if all_results[i] and not all_results[i].curie]

        if still_failed_indices:
            still_failed_entities = [entities[i] for i in still_failed_indices]
            logger.info(
                "Tier 2 retry: %d entities still unresolved, trying aggressive prompt",
                len(still_failed_entities)
            )

            retry_batch_size = max(2, _config.batch_size // 2)
            retry_results = []

            for batch in chunk(still_failed_entities, retry_batch_size):
                batch_results = await asyncio.gather(
                    *[resolve_single_entity(e, is_retry=True) for e in batch],
                    return_exceptions=True,
                )
                retry_results.extend(batch_results)

            # Merge successful retries back
            for idx, retry_result in zip(still_failed_indices, retry_results):
                if isinstance(retry_result, Exception):
                    continue
                resolution, usage_record = retry_result
                if isinstance(resolution, EntityResolution) and resolution.curie:
                    all_results[idx] = resolution
                if usage_record is not None:
                    model_usages.append(usage_record)

        tier2_duration = time.time() - tier2_start
        tier2_resolved = sum(1 for i in tier2_needed_indices if all_results[i] and all_results[i].curie)
        logger.info(
            "Tier 2 (LLM) resolved %d/%d entities in %.1fs",
            tier2_resolved, len(tier2_needed_indices), tier2_duration
        )
    elif tier2_needed_indices:
        # SDK not available - mark remaining as failed
        for idx in tier2_needed_indices:
            all_results[idx] = EntityResolution(
                raw_name=entities[idx],
                curie=None,
                resolved_name=None,
                category=None,
                confidence=0.0,
                method="failed",
            )

    # Ensure no None values remain
    final_results = []
    for i, r in enumerate(all_results):
        if r is None:
            final_results.append(EntityResolution(
                raw_name=entities[i],
                curie=None,
                resolved_name=None,
                category=None,
                confidence=0.0,
                method="failed",
            ))
        else:
            final_results.append(r)

    # Calculate final stats
    resolved = [r for r in final_results if r.curie]
    failed = [r for r in final_results if not r.curie]
    duration = time.time() - start
    rate = 100 * len(resolved) / len(final_results) if final_results else 0

    # Count by tier. Subtract biomapper (pre-pass) too, else the tier2 count is inflated by the
    # pre-resolver's hits once the flag is on.
    alias_resolved = sum(1 for r in final_results if r.method and r.method.startswith("alias:"))
    llm_resolved = len(resolved) - tier1_resolved - alias_resolved - biomapper_resolved

    if failed:
        failed_names = [r.raw_name for r in failed[:5]]
        logger.warning(
            "Failed to resolve %d entities: %s%s",
            len(failed), failed_names, "..." if len(failed) > 5 else ""
        )

    logger.info(
        "Completed entity_resolution in %.1fs — resolved=%d (tier1=%d, biomapper=%d, alias=%d, tier2=%d), failed=%d (%.0f%%)",
        duration, len(resolved), tier1_resolved, biomapper_resolved, alias_resolved, llm_resolved,
        len(failed), rate
    )

    result = {
        "resolved_entities": final_results,
        "errors": errors,
    }
    if model_usages:
        result["model_usages"] = model_usages
    return result
