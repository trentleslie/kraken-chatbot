"""entity_resolution Tier-2 recall go/no-go gate (#61).

Runs the migrated Tier-2 path (HTTP prefetch + SDK select) against a committed
hard-variant fixture with hand-labeled expected CURIEs and checks recall against
the >=95% threshold. Requires a live Kestrel KG and the Claude Agent SDK (auth).

Usage:
    cd backend && uv run python tests/recall_gate.py            # full gate (prefetch + SDK select)
    cd backend && uv run python tests/recall_gate.py --prefetch # prefetch coverage only (no SDK cost)

Exit code 0 = PASS (>= threshold), 1 = FAIL. On a FAIL, the plan decision is to
reclassify entity_resolution to honest-fail (drop Tier-2 -> cold_start).
"""

import asyncio
import json
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
        except Exception:
            pass
    return False


async def main(prefetch_only: bool) -> int:
    data = json.loads(FIXTURE.read_text())
    entities = data["entities"]
    threshold = float(data.get("_threshold", 0.95))
    n = len(entities)

    prefetch_hits = 0
    exact_hits = 0
    same_entity_hits = 0
    for e in entities:
        raw, expected = e["raw_name"], e["expected_curie"]
        exp = _canonical_curie(expected)

        cands = await prefetch_resolution_candidates(raw)
        in_prefetch = exp in {_canonical_curie(c["curie"]) for c in cands}
        prefetch_hits += in_prefetch

        if prefetch_only:
            print(f"{'Y' if in_prefetch else 'n'} {raw:26s} exp={expected:22s} cands={len(cands)}")
            continue

        res, _ = await resolve_single_entity(raw)
        got = res.curie
        exact = _canonical_curie(got) == exp if got else False
        exact_hits += exact
        same = exact or (got is not None and await _same_entity(got, expected))
        same_entity_hits += same
        flag = "OK " if exact else ("~  " if same else "XX ")
        print(f"{flag}{raw:26s} exp={expected:22s} prefetch={'Y' if in_prefetch else 'n'} got={got}")

    print()
    print(f"Prefetch coverage: {prefetch_hits}/{n} = {prefetch_hits / n:.1%}")
    if prefetch_only:
        # Prefetch coverage is the recall floor; selection can only lose from here.
        passed = prefetch_hits / n >= threshold
        print(f"PREFETCH GATE (>= {threshold:.0%}): {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    print(f"Exact-CURIE recall:  {exact_hits}/{n} = {exact_hits / n:.1%}")
    print(f"Same-entity recall:  {same_entity_hits}/{n} = {same_entity_hits / n:.1%}  (resolved == expected or in its equivalent_ids)")
    passed = same_entity_hits / n >= threshold
    print(f"GATE (>= {threshold:.0%}, same-entity): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    code = asyncio.run(main(prefetch_only="--prefetch" in sys.argv))
    sys.exit(code)
