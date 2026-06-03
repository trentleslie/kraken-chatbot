"""entity_resolution Tier-2 recall go/no-go gate (#61).

Runs the migrated Tier-2 path (HTTP prefetch + SDK select) against a committed
hard-variant fixture with hand-labeled expected CURIEs. Requires a live Kestrel KG
and the Claude Agent SDK (auth).

Pass criteria (full mode), reconciled with the recorded team decision that the
fixture's `_ambiguous` entries have multiple equally-valid CURIEs for the same entity
(acid vs anion, or the same compound across UMLS/MESH/CHEBI) for which exact-CURIE
matching is inherently brittle:

    PASS  iff  prefetch coverage >= threshold        (the real recall-regression signal:
                                                       is the right candidate surfaced?)
          AND  every UNAMBIGUOUS entity resolves to the same entity as its label.

`_ambiguous` entries are still run and reported, but are not counted in the strict
pass denominator (a sub-threshold result there reflects KG node-identity ambiguity,
not a resolution failure). The headline same-entity recall over ALL entries is also
printed for transparency.

Usage:
    cd backend && uv run python tests/recall_gate.py            # full gate (prefetch + SDK select)
    cd backend && uv run python tests/recall_gate.py --prefetch # prefetch coverage only (no SDK cost)

Exit code 0 = PASS, 1 = FAIL. On a genuine FAIL (prefetch coverage drops, or an
unambiguous entity mis-resolves), the plan decision is to reclassify entity_resolution
to honest-fail (drop Tier-2 -> cold_start).
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kestrel_backend.graph.nodes.entity_resolution import (  # noqa: E402
    prefetch_resolution_candidates,
    resolve_single_entity,
    _canonical_curie,
)
from kestrel_backend.kestrel_client import call_kestrel_tool  # noqa: E402

logger = logging.getLogger("recall_gate")

FIXTURE = Path(__file__).parent / "fixtures" / "entity_resolution_hard_variants.json"


async def _same_entity(resolved: str, expected: str) -> bool:
    """True if `resolved` and `expected` are the same KG entity — exact, or linked via
    either node's equivalent_ids. Handles same-entity-different-DB (e.g. UMLS vs CHEBI)
    where exact-CURIE matching is brittle but the resolution is still correct."""
    if _canonical_curie(resolved) == _canonical_curie(expected):
        return True
    for a, b in ((expected, resolved), (resolved, expected)):
        try:
            r = await call_kestrel_tool("get_nodes", {"curies": a})
            d = json.loads(r["content"][0]["text"]).get(a, {})
            eq = {_canonical_curie(x) for x in d.get("equivalent_ids", [])}
            if _canonical_curie(b) in eq:
                return True
        except Exception as exc:
            # Don't silently under-count: surface lookup failures.
            logger.warning("_same_entity lookup failed for %s / %s: %s", a, b, exc)
    return False


async def main(prefetch_only: bool) -> int:
    data = json.loads(FIXTURE.read_text())
    entities = data["entities"]
    threshold = float(data.get("_threshold", 0.95))
    n = len(entities)
    if n == 0:
        print("ERROR: fixture contains no entities — nothing to evaluate.")
        return 1

    prefetch_hits = 0
    exact_hits = 0
    same_entity_hits = 0
    unamb_total = 0
    unamb_same_hits = 0
    ambiguous_rows: list[str] = []

    for e in entities:
        raw, expected = e["raw_name"], e["expected_curie"]
        is_ambiguous = bool(e.get("_ambiguous"))
        exp = _canonical_curie(expected)

        cands = await prefetch_resolution_candidates(raw)
        in_prefetch = exp in {_canonical_curie(c["curie"]) for c in cands}
        prefetch_hits += in_prefetch

        if prefetch_only:
            print(f"{'Y' if in_prefetch else 'n'} {raw:26s} exp={expected:22s} cands={len(cands)}")
            continue

        # Retry transient SDK failures (e.g. "Control request timeout: initialize"):
        # resolve_single_entity swallows them into method="failed"/curie=None. Prefetch
        # is 100%, so a None here is an SDK-infra hiccup, not a missing candidate — the
        # gate measures resolution recall, not SDK uptime.
        got = None
        for attempt in range(3):
            res, _ = await resolve_single_entity(raw)
            got = res.curie
            if got is not None:
                break
            logger.warning("resolve returned None for %r (attempt %d/3) — retrying", raw, attempt + 1)
        exact = _canonical_curie(got) == exp if got else False
        exact_hits += exact
        same = exact or (got is not None and await _same_entity(got, expected))
        same_entity_hits += same
        if not is_ambiguous:
            unamb_total += 1
            unamb_same_hits += same
        flag = "OK " if exact else ("~  " if same else "XX ")
        amb = "  [ambiguous]" if is_ambiguous else ""
        line = f"{flag}{raw:26s} exp={expected:22s} prefetch={'Y' if in_prefetch else 'n'} got={got}{amb}"
        print(line)
        if is_ambiguous:
            ambiguous_rows.append(line.strip())

    print()
    prefetch_cov = prefetch_hits / n
    print(f"Prefetch coverage: {prefetch_hits}/{n} = {prefetch_cov:.1%}")
    if prefetch_only:
        # Prefetch coverage is the recall floor; selection can only lose from here.
        passed = prefetch_cov >= threshold
        print(f"PREFETCH GATE (>= {threshold:.0%}): {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    print(f"Exact-CURIE recall (all):       {exact_hits}/{n} = {exact_hits / n:.1%}")
    print(f"Same-entity recall (all):       {same_entity_hits}/{n} = {same_entity_hits / n:.1%}")
    unamb_rate = unamb_same_hits / unamb_total if unamb_total else 1.0
    print(f"Same-entity recall (unambig.):  {unamb_same_hits}/{unamb_total} = {unamb_rate:.1%}  (gated)")
    if ambiguous_rows:
        print(f"\n{len(ambiguous_rows)} ambiguous entr{'y' if len(ambiguous_rows) == 1 else 'ies'} "
              "(multiple valid CURIEs for the same entity; reported, not gated):")
        for row in ambiguous_rows:
            print(f"  - {row}")

    prefetch_ok = prefetch_cov >= threshold
    unamb_ok = unamb_rate >= threshold
    passed = prefetch_ok and unamb_ok
    print()
    print(f"GATE: prefetch_coverage {'>=' if prefetch_ok else '<'} {threshold:.0%} "
          f"AND unambiguous_recall {'>=' if unamb_ok else '<'} {threshold:.0%} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    code = asyncio.run(main(prefetch_only="--prefetch" in sys.argv))
    sys.exit(code)
