"""
Triage & Route Node: Classify entities by KG connectivity for routing.

This node performs two-tier edge counting and classification:

Tier 1 (API): Direct one_hop_query with mode="preview" (~100ms each)
  - Fast, returns results_count which is the edge count
  - Runs all entities in parallel

Tier 2 (LLM): Falls back to Claude Agent SDK for failures
  - Handles cases where API returns errors
  - More expensive but can retry with different parameters

Classification thresholds:
- cold_start: 0 edges (no KG presence)
- sparse: 1-19 edges (limited connectivity)
- moderate: 20-199 edges (reasonable coverage)
- well_characterized: >=200 edges (rich KG representation)

The route_after_triage function in builder.py uses these classifications
to route entities to the appropriate analysis branches (direct_kg or cold_start).
"""

import asyncio
import json
import logging
import math
import time
from typing import Any

from ...kestrel_client import call_kestrel_tool
from ..pipeline_config import get_pipeline_config
from ..state import DiscoveryState, NoveltyScore, EntityResolution
from ..state_contracts import validate_state, TriageInput, TriageOutput

logger = logging.getLogger(__name__)


# Classification thresholds
THRESHOLD_WELL_CHARACTERIZED = 200
THRESHOLD_MODERATE = 20
THRESHOLD_SPARSE = 1

# Edge-count retry: a couple of attempts with a short backoff cover transient Kestrel hiccups
# (server isError / timeout) without re-firing instantly into the same load (plan 2026-06-23-001).
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 0.5


async def count_edges_via_api(entity: EntityResolution, sem: asyncio.Semaphore) -> NoveltyScore | None:
    """
    Tier 1: Count edges via direct Kestrel API call.

    Uses one_hop_query with mode="preview" which returns results_count (edge count).
    Runs under the caller's concurrency semaphore (bounds in-flight Kestrel calls). Returns a
    NoveltyScore for a measured count (including a genuine ``results_count == 0`` → cold_start),
    or **None when the count could not be MEASURED** (server ``isError`` / empty content /
    unparseable JSON / exhausted retries). The caller treats None as a measurement failure and
    routes the entity to the direct-KG path (``moderate``) with a visible ``errors`` marker —
    NOT silently to cold_start (plan 2026-06-23-001).

    Transient failures (server ``isError`` / exception) are retried with backoff up to
    ``_MAX_ATTEMPTS``. *Deterministic* per-CURIE failures (empty content, unparseable JSON) are
    NOT retried — an identical-args retry would re-fail — and return None immediately.
    """
    curie = entity.curie
    raw_name = entity.raw_name

    # Skip entities that failed resolution (deterministic — no query, no retry)
    if not curie or entity.method == "failed":
        return NoveltyScore(
            curie=curie or raw_name,
            raw_name=raw_name,
            edge_count=0,
            classification="cold_start",
        )

    max_attempts = _MAX_ATTEMPTS
    for attempt in range(max_attempts):
        try:
            # Call one_hop_query with preview mode - returns counts instead of full data.
            # The semaphore bounds in-flight Kestrel calls; it wraps only the call (not the retry
            # backoff) so a sleeping retry releases its slot to other entities.
            async with sem:
                result = await call_kestrel_tool("one_hop_query", {
                    "start_node_ids": curie,
                    "mode": "preview",
                    "limit": 10000,  # High limit to get accurate count
                })

            is_error = result.get("isError", False)
            content = result.get("content", [])

            if is_error:
                # Transient server-side error — back off, retry, then give up.
                logger.debug(
                    "Tier 1 triage '%s': API isError (attempt %d/%d)",
                    curie, attempt + 1, max_attempts,
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                continue

            if not content:
                # Deterministic empty response for this CURIE — do not retry.
                logger.debug("Tier 1 triage '%s': no content", curie)
                return None

            # Parse response
            text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Deterministic malformed response — do not retry.
                logger.debug("Tier 1 triage '%s': Could not parse JSON", curie)
                return None

            # results_count is the edge count
            edge_count = int(data.get("results_count", 0))
            classification = classify_by_edge_count(edge_count)

            logger.info(
                "Tier 1 triage '%s': edges=%d, classification=%s",
                curie, edge_count, classification
            )

            return NoveltyScore(
                curie=curie,
                raw_name=raw_name,
                edge_count=edge_count,
                classification=classification,
            )

        except Exception as e:
            # Transient (timeout / connection) — back off, retry, then give up.
            logger.warning(
                "Tier 1 triage '%s': Exception (attempt %d/%d) - %s",
                curie, attempt + 1, max_attempts, str(e),
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(_RETRY_BACKOFF_S)
            continue

    # All attempts exhausted on transient failures → cold_start default (None).
    return None


def classify_by_edge_count(edge_count: int) -> str:
    """Classify entity by edge count thresholds."""
    if edge_count >= THRESHOLD_WELL_CHARACTERIZED:
        return "well_characterized"
    elif edge_count >= THRESHOLD_MODERATE:
        return "moderate"
    elif edge_count >= THRESHOLD_SPARSE:
        return "sparse"
    else:
        return "cold_start"


# =============================================================================
# Axis B: intramodular-centrality hubs + inverted routing (consumes axis A ModuleSpine)
# Plan: docs/plans/2026-07-16-001-feat-triage-intramodular-centrality-inverted-routing-plan.md
#
# Rationale (validation memo 20260716-212442): KG edge-count degree is study bias, not biology
# (PNAS 2025 10.1073/pnas.2416646122; arXiv 2405.14985) — a degree-bias-only link predictor beats
# sophisticated models, so the edge-count hub is an artifact. The biologically meaningful hub is the
# one central WITHIN its WGCNA module (intramodular connectivity). And in validation cold_start beat
# direct_kg, so high-degree entities we currently trust onto the fast path should get MORE scrutiny.
# This pass marks the top-k% |kME| members of each module as hubs and inverts their routing.
# =============================================================================


def _normalize_name(name: str) -> str:
    """Normalized join key: case-insensitive, trimmed. The ModuleSpine ``members`` map is keyed by
    the canonical analyte name the user uploaded, which is also ``NoveltyScore.raw_name`` — but the
    spine predates resolution, so we join on the name, not the CURIE (mirror the analyte-upload dedup
    identity, R10)."""
    return (name or "").strip().lower()


def _spine_get(obj: Any, key: str, default: Any = None) -> Any:
    """Duck-typed attribute/item access. Axis A owns the ModuleSpine Pydantic type (not importable
    here); in-process it is a model (attribute access), but after any JSON round-trip it is a dict —
    so read both. (Q1: coded against L1's shape, joined by name.)"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _module_top_k(n_members: int, top_k_pct: float) -> int:
    """Relative top-k%: ceil(pct/100 * module_size), at least 1. Percentile, never an absolute
    |kME| threshold — module |kME| distributions vary, so a relative cut is cohort-portable."""
    return max(1, math.ceil((top_k_pct / 100.0) * n_members))


def compute_hub_members(module_spine: Any, cfg: Any) -> dict[str, dict[str, Any]]:
    """Per-module |kME| top-k% hub detection with kIM veto → ``{normalized_name: entry}``.

    ``entry`` = ``{"kme": float, "kim": float|None, "crowned": bool, "group": str}``. For each module
    (group), members are ranked by ``abs(kme)`` descending and the top ``ceil(k% * size)`` are hub
    candidates. A candidate is vetoed (``crowned=False``) only when a ``kim_floor`` is configured AND
    the member's ``kim`` is present AND below the floor (the low-n kME caveat, R4); a missing ``kim``
    skips the veto (|kME| alone decides, L4). A member appearing in several modules is ``crowned`` if
    crowned in ANY; its reported ``kme``/``kim`` are from the module where ``|kME|`` is largest (most
    representative). Members without a ``kme`` are ignored (they carry no centrality signal).
    """
    agg: dict[str, dict[str, Any]] = {}
    if not module_spine:
        return agg
    kim_floor = getattr(cfg, "intramodular_kim_floor", None)
    top_k_pct = getattr(cfg, "intramodular_hub_top_k_pct", 10.0)

    for group, module in module_spine.items():
        members = _spine_get(module, "members", {}) or {}
        parsed: list[tuple[str, float, float | None]] = []
        for name, mw in members.items():
            kme = _spine_get(mw, "kme")
            if kme is None:
                continue
            kim = _spine_get(mw, "kim")
            parsed.append((name, float(kme), None if kim is None else float(kim)))
        if not parsed:
            continue
        parsed.sort(key=lambda t: abs(t[1]), reverse=True)
        top_k = _module_top_k(len(parsed), top_k_pct)
        crowned_names = {parsed[i][0] for i in range(min(top_k, len(parsed)))}

        for name, kme, kim in parsed:
            crowned = name in crowned_names
            if crowned and kim_floor is not None and kim is not None and kim < kim_floor:
                crowned = False  # kIM veto: high |kME|, low connectivity = false hub at low n
            norm = _normalize_name(name)
            entry = agg.get(norm)
            if entry is None:
                agg[norm] = {"kme": kme, "kim": kim, "crowned": crowned, "group": group}
            else:
                entry["crowned"] = entry["crowned"] or crowned
                if abs(kme) > abs(entry["kme"]):
                    entry.update(kme=kme, kim=kim, group=group)
    return agg


def expected_hub_count(module_spine: Any, cfg: Any) -> int:
    """Sum over modules of ceil(k% * weighted_module_size) — the count of hubs the top-k% rule would
    crown before the kIM veto and the name-join. Compared against the actual hub count in the R11
    hook so a silent name-join failure (actual << expected) is caught, not read as 'a small hub set'.
    """
    if not module_spine:
        return 0
    top_k_pct = getattr(cfg, "intramodular_hub_top_k_pct", 10.0)
    total = 0
    for _group, module in module_spine.items():
        members = _spine_get(module, "members", {}) or {}
        n = sum(1 for mw in members.values() if _spine_get(mw, "kme") is not None)
        if n:
            total += _module_top_k(n, top_k_pct)
    return total


@validate_state(TriageInput, TriageOutput)
async def run(state: DiscoveryState) -> dict[str, Any]:
    """
    Triage resolved entities by KG connectivity using Tier-1 HTTP only (#61).

    Tier 1 (API): bounded-concurrency one_hop_query (mode="preview") for all entities, retried
      with backoff on transient isError/exception. A genuine 0-edge count → cold_start; an entity
      whose count cannot be MEASURED (None) is routed to ``moderate`` (the direct-KG path) with a
      visible ``errors`` marker, never silently to cold_start (plan 2026-06-23-001). The broken
      stdio-MCP Tier-2 LLM fallback was removed (#61).

    Returns:
        novelty_scores: List of NoveltyScore objects
        well_characterized_curies: CURIEs with >=200 edges
        moderate_curies: CURIEs with 20-199 edges (incl. measurement-failed entities)
        sparse_curies: CURIEs with 1-19 edges
        cold_start_curies: CURIEs with 0 edges (genuine) + failed-resolution names
        errors: degraded markers for entities whose edge count could not be measured
    """
    logger.info("Starting triage")
    start = time.time()

    resolved = state.get("resolved_entities", [])

    # Filter to only entities with valid CURIEs
    valid_entities = [e for e in resolved if e.curie and e.method != "failed"]

    if not valid_entities:
        # No valid entities - everything goes to cold_start
        failed_names = [e.raw_name for e in resolved if e.method == "failed"]
        return {
            "novelty_scores": [],
            "well_characterized_curies": [],
            "moderate_curies": [],
            "sparse_curies": [],
            "cold_start_curies": failed_names,
            "errors": [],
        }

    all_scores: list[NoveltyScore | None] = [None] * len(valid_entities)
    errors: list[str] = []

    # ========== TIER 1: API Edge Counting ==========
    tier1_start = time.time()
    logger.info("Tier 1 (API): Counting edges for %d entities", len(valid_entities))

    # Run API calls concurrently but BOUNDED — an unbounded fan-out over hundreds of entities
    # thundering-herds Kestrel into timeouts (plan 2026-06-23-001). Per-invocation semaphore.
    sem = asyncio.Semaphore(get_pipeline_config().triage.kestrel_concurrency)
    tier1_results = await asyncio.gather(
        *[count_edges_via_api(e, sem) for e in valid_entities],
        return_exceptions=True,
    )

    tier1_success = 0
    tier1_failed_indices = []

    for i, (entity, result) in enumerate(zip(valid_entities, tier1_results)):
        if isinstance(result, Exception):
            logger.debug("Tier 1 triage '%s': Exception - %s", entity.curie, str(result))
            tier1_failed_indices.append(i)
        elif result is not None:
            all_scores[i] = result
            tier1_success += 1
        else:
            tier1_failed_indices.append(i)

    tier1_duration = time.time() - tier1_start
    logger.info(
        "Tier 1 (API) counted edges for %d/%d entities in %.1fs",
        tier1_success, len(valid_entities), tier1_duration
    )

    # ========== Measurement failures reroute to direct-KG, never silent cold_start ==========
    # count_edges_via_api retries transient failures (server isError / exception) with backoff;
    # an entity whose count still cannot be MEASURED (None) is NOT a genuine 0-edge entity. A real
    # 0 returns a NoveltyScore (edge_count=0 -> cold_start); only a measurement failure is None.
    # Routing a measurement failure to cold_start would make a Kestrel overload read as "no KG
    # presence" (the silent-degradation anti-pattern, cf. synthesis overflow, PR #85). Instead we
    # route it to the direct-KG path via the `moderate` bucket and emit a visible marker.
    model_usages: list = []
    if tier1_failed_indices:
        for idx in tier1_failed_indices:
            entity = valid_entities[idx]
            logger.info(
                "FALLBACK_EVENT node=triage entity=%s curie=%s "
                "reason=tier1_edge_count_failed action=reroute_direct_kg_moderate",
                getattr(entity, "raw_name", str(entity)),
                getattr(entity, "curie", "unknown"),
            )

    # Ensure no None values; a None means the count FAILED (not measured 0) -> moderate + marker.
    final_scores = []
    for i, s in enumerate(all_scores):
        if s is None:
            ent = valid_entities[i]
            cur = ent.curie or ent.raw_name
            final_scores.append(NoveltyScore(
                curie=cur,
                raw_name=ent.raw_name,
                edge_count=0,
                classification="moderate",
            ))
            errors.append(
                f"triage: edge-count failed for {cur} ({ent.raw_name}); "
                "routed to direct-KG (moderate)"
            )
        else:
            final_scores.append(s)

    # ========== Axis B: intramodular-centrality hub detection + inverted routing ==========
    # Flag-gated and consumes axis A's ModuleSpine (read-only). When disabled, or no ModuleSpine, or
    # no member carries kME → is_intramodular_hub stays False everywhere and bucketing below is
    # byte-identical to the edge-count baseline (fallback identity). The pass runs AFTER the
    # measurement-failure→moderate backfill, so a measurement failure is never rescued — it only
    # gains hub status if it is genuinely a top-k% kME module member (then cold_start = more
    # scrutiny, the intended safe direction).
    cfg = get_pipeline_config().triage
    module_spine = state.get("module_spine")
    centrality_active = bool(cfg.intramodular_centrality_enabled and module_spine)
    hub_members: dict[str, dict[str, Any]] = {}
    if centrality_active:
        hub_members = compute_hub_members(module_spine, cfg)
        if hub_members:
            promoted = []
            for s in final_scores:
                m = hub_members.get(_normalize_name(s.raw_name))
                if m is not None:
                    s = s.model_copy(update={
                        "is_intramodular_hub": bool(m["crowned"]),
                        "kme": m["kme"],
                        "kim": m["kim"],
                    })
                promoted.append(s)
            final_scores = promoted

    # Classify into routing buckets. An intramodular hub is kept OUT of its edge-count bucket and
    # placed into cold_start (inversion, R6/R8) — routing keys on the boolean, not the classification,
    # which is preserved for synthesis/display. When centrality is off, no score is a hub, so this
    # reduces exactly to the original bucketing.
    well_characterized = [s.curie for s in final_scores
                          if s.classification == "well_characterized" and not s.is_intramodular_hub]
    moderate = [s.curie for s in final_scores
                if s.classification == "moderate" and not s.is_intramodular_hub]
    sparse = [s.curie for s in final_scores
              if s.classification == "sparse" and not s.is_intramodular_hub]
    cold_start = [s.curie for s in final_scores
                  if s.classification == "cold_start" and not s.is_intramodular_hub]
    hub_curies = [s.curie for s in final_scores if s.is_intramodular_hub]
    cold_start.extend(hub_curies)  # inverted routing: all intramodular hubs → cold_start

    # Add failed resolutions to cold_start bucket
    failed_names = [e.raw_name for e in resolved if e.method == "failed"]
    cold_start.extend(failed_names)

    duration = time.time() - start
    logger.info(
        "Completed triage in %.1fs — well_char=%d, moderate=%d, sparse=%d, cold_start=%d "
        "(tier1_ok=%d, tier1_failed=%d)",
        duration, len(well_characterized), len(moderate), len(sparse), len(cold_start),
        tier1_success, len(valid_entities) - tier1_success
    )

    # Unit 0b — forward triage-outcome counter (Decision 3). The human log line above is for
    # operators; this machine-parseable record lets us size the production query mix (what fraction
    # of runs produce speculative sparse/cold_start hypotheses vs. well-characterized-only) once real
    # traffic accrues, so the well-characterized latency cost of ground-before-synthesis becomes an
    # evidenced choice. `produces_speculative` is the routing-relevant predicate (sparse OR cold_start
    # entities are what generate cold-start hypotheses). One structured line per run; no new table.
    outcome: dict[str, Any] = {
        "event": "triage_outcome",
        "well_characterized": len(well_characterized),
        "moderate": len(moderate),
        "sparse": len(sparse),
        "cold_start": len(cold_start),
        "tier1_ok": tier1_success,
        "tier1_failed": len(valid_entities) - tier1_success,
        "produces_speculative": bool(sparse or cold_start),
        "duration_seconds": round(duration, 2),
    }
    # Axis B measurement hooks (R10-R11) — added ONLY when the centrality pass ran, so the flag-off
    # line is unchanged. They quantify how the |kME| hub notion diverges from the edge-degree hub
    # notion (finding #2: KG degree = study bias) and the routing blast radius. `expected_hub_n` vs
    # `kme_hub_n` is the join-integrity check: a silent name-join failure collapses `kme_hub_n` toward
    # 0 while `expected_hub_n` stays positive, so the shift-toward-0 is not mistaken for "few hubs".
    if centrality_active:
        edge_degree_hub_set = {s.curie for s in final_scores
                               if s.edge_count >= THRESHOLD_WELL_CHARACTERIZED}
        kme_hub_set = {s.curie for s in final_scores if s.is_intramodular_hub}
        union = edge_degree_hub_set | kme_hub_set
        intersection = edge_degree_hub_set & kme_hub_set
        outcome.update({
            "hub_set_jaccard": round(len(intersection) / len(union), 4) if union else 0.0,
            "edge_degree_hub_n": len(edge_degree_hub_set),                # edge_count >= 200
            "edge_degree_hub_n_1000": sum(1 for s in final_scores if s.edge_count > 1000),
            "kme_hub_n": len(kme_hub_set),
            "hub_set_only_edge_degree_n": len(edge_degree_hub_set - kme_hub_set),
            "hub_set_only_kme_n": len(kme_hub_set - edge_degree_hub_set),
            # entities edge-count would have sent to direct_kg (well_characterized/moderate) but are
            # now hub → cold_start
            "routing_shift_direct_to_cold": sum(
                1 for s in final_scores if s.is_intramodular_hub
                and s.classification in ("well_characterized", "moderate")),
            # 0 under pure inversion; emitted for auditing (no mechanism moves cold→direct)
            "routing_shift_cold_to_direct": 0,
            "expected_hub_n": expected_hub_count(module_spine, cfg),
        })
    logger.info("triage_outcome %s", json.dumps(outcome))

    result = {
        "novelty_scores": final_scores,
        "well_characterized_curies": well_characterized,
        "moderate_curies": moderate,
        "sparse_curies": sparse,
        "cold_start_curies": cold_start,
        "errors": errors,
    }
    if model_usages:
        result["model_usages"] = model_usages
    return result
