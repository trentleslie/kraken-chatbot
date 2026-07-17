"""Axis C — bridge specificity scoring by intermediate-node degree (DWPC).

Score each cross-entity bridge by the KG degree of its intermediate node(s) using
Degree-Weighted Path Count (DWPC) damping. Bridges through generic, high-degree intermediates
(e.g. "blood", "placenta", "cancer") are penalized; bridges through specific, low-degree
intermediates are rewarded.

The 1-MNA hypothesis ("TNFRSF10A -> 1-MNA via blood, placenta, cancer") INVERTED — its
intermediates are near-universal, extremely high-degree KG nodes and "both found in blood" is
true of almost everything. This module produces a deterministic STRUCTURAL-genericity signal
(NOT a mechanism-confidence claim) that synthesis (axis E) can use to discount such bridges.

Scoring is a pure, deterministic transform (this ``bridge_specificity`` fn — no I/O). The KG
degree behind each intermediate is resolved by :func:`make_degree_provider` (bounded + per-run
cached, best-effort). Emit-only: nothing here filters, re-tiers, or mutates a bridge.

References:
- Himmelstein & Baranzini, Heterogeneous Network Edge Prediction (DWPC), PLOS Comput Biol 2015,
  10.1371/journal.pcbi.1004259. w ~= 0.4 empirically optimal for disease-gene prediction.
- Himmelstein et al., Rephetio, eLife 2017, 26726.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from ...kestrel_client import call_kestrel_tool
from ..state import BridgeSpecificity

logger = logging.getLogger(__name__)

# --- Cited / tuned module constants (NOT runtime config) -----------------------------------
# These are calibrated once from a real intermediate-degree distribution (Unit 4 probe), then
# frozen — they are cited constants, not a per-run config surface (there is no consumer for
# runtime-tunable cut points). The values below are placeholders pending the Unit 4 calibration
# and are known to be mis-skewed relative to real KG degrees; they are chosen to (a) keep the
# `specific` reward bucket reachable and (b) align the per-node generic cutoff with the pipeline's
# existing "hub" notion (SharedNeighbor.is_hub = degree > 1000; direct_kg/integration hub guard).

# DWPC damping exponent (Himmelstein 2015: w ~= 0.4 optimal for disease-gene prediction).
DAMPING_W: float = 0.4

# Aggregate-score cut points (applied only when the bridge is not condemned by a generic hub):
#   score >= SPECIFIC_CUT -> "specific"
#   score >= MODERATE_CUT -> "moderate"
#   else                  -> "generic"
# With w=0.4 and a single intermediate, score = degree**-0.4, so SPECIFIC_CUT=0.20 admits
# degree <= ~55 and MODERATE_CUT=0.10 admits degree <= ~316. Placeholders (see note above).
SPECIFIC_CUT: float = 0.20
MODERATE_CUT: float = 0.10

# Per-node degree cutoff: an intermediate whose OWN known degree strictly exceeds this is
# "generic" and lands in `generic_intermediates`. Aligned with the existing is_hub definition
# (degree > 1000). Placeholder pending the Unit 4 quantile calibration.
GENERIC_CUTOFF: int = 1000


def _bucket(score: float) -> str:
    """Map an aggregate DWPC score to a label (inclusive on the upper bucket boundary)."""
    if score >= SPECIFIC_CUT:
        return "specific"
    if score >= MODERATE_CUT:
        return "moderate"
    return "generic"


def bridge_specificity(
    intermediate_curies: list[str],
    intermediate_degrees: list[int | None],
) -> BridgeSpecificity:
    """Pure DWPC scorer: intermediate CURIEs + degrees -> ``BridgeSpecificity``. No I/O.

    Length-normalized (geometric-mean) DWPC: ``score = (prod known_degree_i) ** (-w / n_known)``,
    so 1-, 2-, and 3-intermediate scaffolds are comparable under one cut point (a raw degree
    product would penalize a longer path through several moderately-specific nodes purely for
    length). Each degree is clamped to >= 1 (an isolated node, results_count == 0, is maximally
    specific, and 0 ** -0.4 is infinite), so the score stays in (0, 1].

    Rules:
    - Zero intermediates (2-node bridge): maximally specific (score 1.0, label "specific").
    - ``generic_intermediates`` = scaffold CURIEs whose OWN known degree strictly exceeds
      ``GENERIC_CUTOFF``.
    - Condemn-on-known: if any known intermediate is generic -> label "generic" regardless of any
      missing degrees (a partially-unknown scaffold cannot become more specific by hiding a degree).
    - ``label == "unknown"`` is reserved for the all-degrees-missing case only; ``score`` is then
      None. The score is otherwise computed over the known-degree subset.

    ``intermediate_curies`` and ``intermediate_degrees`` are parallel; ``intermediate_degrees``
    preserves the raw values (including ``None`` and 0) on the returned model.
    """
    curies = list(intermediate_curies)
    degrees = list(intermediate_degrees)

    # 2-node bridge: no scaffold to discount -> maximally specific.
    if not curies:
        return BridgeSpecificity(
            score=1.0,
            label="specific",
            intermediate_curies=[],
            intermediate_degrees=[],
            generic_intermediates=[],
        )

    # Generic hubs are decided on KNOWN degrees (strict >), independent of the score.
    generic_intermediates = [
        c for c, d in zip(curies, degrees) if d is not None and d > GENERIC_CUTOFF
    ]

    known = [d for d in degrees if d is not None]
    if not known:
        # All degrees missing: no evidence either way -> unknown (score None).
        return BridgeSpecificity(
            score=None,
            label="unknown",
            intermediate_curies=curies,
            intermediate_degrees=degrees,
            generic_intermediates=generic_intermediates,
        )

    # Length-normalized geometric-mean DWPC over the known subset; clamp each degree to >= 1.
    product = 1.0
    for d in known:
        product *= max(int(d), 1)
    score = product ** (-DAMPING_W / len(known))

    # Condemn-on-known: a generic hub forces "generic" regardless of any missing degrees.
    label = "generic" if generic_intermediates else _bucket(score)

    return BridgeSpecificity(
        score=score,
        label=label,
        intermediate_curies=curies,
        intermediate_degrees=degrees,
        generic_intermediates=generic_intermediates,
    )


# --- Hybrid, bounded, per-run cached degree provider (R2, ledger L9) ------------------------
# NOTE (R6): this reads RAW KG connectivity (an intermediate node's edge count), which is
# distinct from axis B's intramodular centrality. It is a candidate future shared "KG degree for
# a CURIE" helper across triage / pathway_enrichment / this provider — deliberately NOT coupled
# here (that convergence is a separate refactor).

# one_hop_query count limit. Preview mode returns results_count (the edge count == degree);
# a high limit yields an accurate count, matching triage's read.
_DEGREE_QUERY_LIMIT = 10000


def _inline_degree(node: Any) -> int | None:
    """Opportunistic per-node degree from a KG envelope's ``nodes`` entry.

    The documented multi_hop/subgraph ``nodes`` entry is ``{name, categories}`` — NO degree — so
    this almost always returns None and the provider falls back to the bounded fetch. Kept only as
    an opportunistic bonus IF a live envelope is confirmed to carry an integer ``degree``.
    """
    if isinstance(node, dict):
        d = node.get("degree")
        if isinstance(d, int) and not isinstance(d, bool):
            return d
    return None


async def _fetch_degree(curie: str) -> int | None:
    """Best-effort KG degree via one_hop_query preview (``results_count``). None on any failure."""
    try:
        resp = await call_kestrel_tool(
            "one_hop_query",
            {"start_node_ids": curie, "mode": "preview", "limit": _DEGREE_QUERY_LIMIT},
        )
    except Exception as e:  # best-effort: a Kestrel failure -> no degree, never propagates
        logger.warning("bridge_specificity: degree fetch failed for %s: %s", curie, e)
        return None
    if not isinstance(resp, dict) or resp.get("isError"):
        return None
    content = resp.get("content") or []
    if not content:
        return None
    try:
        data = json.loads(content[0].get("text", ""))
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    rc = data.get("results_count")
    if rc is None:
        return None
    try:
        return int(rc)
    except (TypeError, ValueError):
        return None


def make_degree_provider(
    inline_nodes: dict[str, Any] | None = None,
    concurrency: int = 8,
) -> Callable[[str], Awaitable[int | None]]:
    """Build a single-flight, concurrency-bounded, per-run cached CURIE->degree resolver.

    Fetch-dominant: for each CURIE, read an opportunistic inline degree from ``inline_nodes`` if
    present, else acquire the semaphore and fetch the KG degree once (``one_hop_query`` preview,
    ``results_count``). The result — including ``None`` — is cached per run so a hub intermediate
    shared across bridges is fetched at most once, and concurrent callers await the same in-flight
    task. Never raises; returns ``int | None``.

    Mirrors ``bridge_grounding.cached_leg_fetcher`` (per-run dedup cache + bounded fetch). Bounding
    is mandatory: an unbounded per-bridge one_hop fan-out once exhausted Kestrel's LMDB readers
    (MDB_READERS_FULL incident, 2026-06-24). A fresh cache is built per call — never stale across runs.
    """
    inline = inline_nodes or {}
    sem = asyncio.Semaphore(concurrency)
    cache: dict[str, "asyncio.Future[int | None]"] = {}

    async def get(curie: str) -> int | None:
        task = cache.get(curie)
        if task is None:
            async def _go() -> int | None:
                # Opportunistic inline read is free (no Kestrel call, outside the semaphore).
                d = _inline_degree(inline.get(curie))
                if d is not None:
                    return d
                async with sem:
                    return await _fetch_degree(curie)

            task = asyncio.ensure_future(_go())
            cache[curie] = task
        return await task

    return get
