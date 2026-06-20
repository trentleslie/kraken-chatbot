"""L1 — bridge evidence-provenance classifier + per-leg labeler.

Deterministic: per leg, the best evidence tier from the leg's KG edges'
knowledge_level + agent_type + Biolink predicate class. No score, no LLM.

Run with: uv run python -m pytest tests/test_bridge_provenance.py -v
"""

import asyncio
import json

import pytest

from src.kestrel_backend.bridge_grounding import provenance
from src.kestrel_backend.bridge_grounding.provenance import (
    bridge_label,
    cached_leg_fetcher,
    evidence_tier,
    is_curated,
    leg_tier,
    predicate_class,
)


def _edge(predicate, knowledge_level, agent_type, subject="A", obj="B", **extra):
    e = {"subject": subject, "object": obj, "predicate": predicate,
         "knowledge_level": knowledge_level, "agent_type": agent_type}
    e.update(extra)
    return e


# --- predicate_class -------------------------------------------------------------------

@pytest.mark.parametrize("predicate,expected", [
    ("biolink:causes", "causal"),
    ("biolink:contributes_to", "causal"),
    ("biolink:regulates", "causal"),
    ("biolink:directly_physically_interacts_with", "causal"),
    ("biolink:associated_with", "associative"),
    ("biolink:correlated_with", "associative"),
    ("biolink:gene_associated_with_condition", "associative"),
    ("biolink:has_adverse_event", "associative"),
    # treats/applied_to_treat are TREATMENT assertions, NOT A->B->C mechanism (v2 finding)
    ("biolink:treats", "associative"),
    ("biolink:applied_to_treat", "associative"),
    ("biolink:related_to", "neutral"),
    ("biolink:interacts_with", "neutral"),
    ("biolink:some_unknown_predicate", "neutral"),
])
def test_predicate_class(predicate, expected):
    assert predicate_class(predicate) == expected


# --- is_curated ------------------------------------------------------------------------

@pytest.mark.parametrize("kl,agent,expected", [
    ("knowledge_assertion", "manual_agent", True),
    ("logical_entailment", "manual_agent", True),
    ("knowledge_assertion", "manual_validation_of_automated_agent", True),
    ("knowledge_assertion", "text_mining_agent", False),   # asserted but text-mined agent
    ("not_provided", "manual_agent", False),               # manual but no KL
    ("statistical_association", "manual_agent", False),
    ("not_provided", "text_mining_agent", False),
])
def test_is_curated(kl, agent, expected):
    assert is_curated(kl, agent) is expected


# --- evidence_tier (per edge) ----------------------------------------------------------

def test_evidence_tier_curated_causal():
    assert evidence_tier(_edge("biolink:causes", "knowledge_assertion", "manual_agent")) == "curated-causal"


def test_evidence_tier_text_mined_causes_is_not_curated_causal():
    # A text-mined `causes` is text-mined, NOT curated-causal (the v1/v2 failure mode).
    assert evidence_tier(_edge("biolink:causes", "not_provided", "text_mining_agent")) == "text-mined"


def test_evidence_tier_curated_associative_and_neutral():
    assert evidence_tier(_edge("biolink:associated_with", "knowledge_assertion", "manual_agent")) == "curated-associative"
    assert evidence_tier(_edge("biolink:related_to", "knowledge_assertion", "manual_agent")) == "curated-neutral"


# --- leg_tier (async fetch + best-of-candidates) ---------------------------------------

class _FakeKestrel:
    """Records the one_hop_query args and returns a canned full-mode edge response."""

    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    async def __call__(self, tool, args):
        self.calls.append((tool, args))
        import json
        body = json.dumps({"results": [], "nodes": {}, "edges": {str(i): e for i, e in enumerate(self.edges)}})
        return {"content": [{"type": "text", "text": body}], "isError": False}


async def test_leg_tier_best_of_candidates(monkeypatch):
    # leg A-B has a text-mined causes AND a curated causes -> best (curated-causal) wins.
    fake = _FakeKestrel([
        _edge("biolink:causes", "not_provided", "text_mining_agent", "A", "B"),
        _edge("biolink:causes", "knowledge_assertion", "manual_agent", "A", "B"),
        _edge("biolink:related_to", "not_provided", "text_mining_agent", "A", "Z"),  # not A-B, ignored
    ])
    monkeypatch.setattr(provenance, "call_kestrel_tool", fake)
    tier = await leg_tier("A", "B")
    assert tier == "curated-causal"
    # uses one_hop_query, start-only (NOT end_node_ids), full mode
    tool, args = fake.calls[0]
    assert tool == "one_hop_query"
    assert args.get("start_node_ids") == "A" and "end_node_ids" not in args
    assert args.get("mode") == "full"


async def test_leg_tier_no_edges_is_none(monkeypatch):
    monkeypatch.setattr(provenance, "call_kestrel_tool", _FakeKestrel([]))
    assert await leg_tier("A", "B") == "none"


async def test_leg_tier_filters_to_target(monkeypatch):
    # Only edges touching B count; an A-Z edge must not set the tier.
    fake = _FakeKestrel([_edge("biolink:causes", "knowledge_assertion", "manual_agent", "A", "Z")])
    monkeypatch.setattr(provenance, "call_kestrel_tool", fake)
    assert await leg_tier("A", "B") == "none"


async def test_leg_tier_retries_from_other_endpoint_on_truncation(monkeypatch):
    # Hub guard: X's edge list is TRUNCATED (hits the cap) with no A-B edge, so the true A-B edge
    # may lie beyond the cap. The retry from B (lower-degree) finds the curated edge -> labeled,
    # not a false 'none'. (Greptile PR #79 fix.)
    import json
    monkeypatch.setattr(provenance, "_LEG_EDGE_LIMIT", 2)
    ab_edge = _edge("biolink:causes", "knowledge_assertion", "manual_agent", "A", "B")

    class _DirectionalFake:
        def __init__(self):
            self.calls = []

        async def __call__(self, tool, args):
            start = args.get("start_node_ids")
            self.calls.append(start)
            if start == "A":  # hub: truncated (== cap) and no A-B edge
                edges = [
                    _edge("biolink:related_to", "not_provided", "text_mining_agent", "A", "Z1"),
                    _edge("biolink:related_to", "not_provided", "text_mining_agent", "A", "Z2"),
                ]
            else:  # retry from B finds the real A-B edge
                edges = [ab_edge]
            body = json.dumps(
                {"results": [], "nodes": {}, "edges": {str(i): e for i, e in enumerate(edges)}})
            return {"content": [{"type": "text", "text": body}], "isError": False}

    fake = _DirectionalFake()
    monkeypatch.setattr(provenance, "call_kestrel_tool", fake)
    assert await leg_tier("A", "B") == "curated-causal"
    assert fake.calls == ["A", "B"]  # retried from the other endpoint


async def test_leg_tier_no_retry_when_not_truncated(monkeypatch):
    # Below the cap (not truncated) with no A-B edge -> 'none', and NO second probe call.
    monkeypatch.setattr(provenance, "_LEG_EDGE_LIMIT", 100)
    fake = _FakeKestrel([_edge("biolink:causes", "knowledge_assertion", "manual_agent", "A", "Z")])
    monkeypatch.setattr(provenance, "call_kestrel_tool", fake)
    assert await leg_tier("A", "B") == "none"
    assert len(fake.calls) == 1  # no retry


# --- bridge_label (compose two legs) ---------------------------------------------------

def test_bridge_label_both_curated_causal():
    assert bridge_label("curated-causal", "curated-causal") == "both legs curated-causal"


def test_bridge_label_one_leg_unsupported():
    assert bridge_label("curated-causal", "none") == "one leg unsupported"
    assert bridge_label("none", "curated-associative") == "one leg unsupported"


def test_bridge_label_no_kg_edge():
    assert bridge_label("none", "none") == "no KG edge"


def test_bridge_label_weakest_leg():
    # Mixed tiers (both present) -> summarized by the weaker leg.
    assert bridge_label("curated-causal", "text-mined") == "weakest leg text-mined"
    assert bridge_label("curated-associative", "curated-neutral") == "weakest leg curated-neutral"


# --- cached_leg_fetcher: single-flight + concurrency bound (perf optimization) ----------

def _kestrel_resp(edges_by_id):
    return {"content": [{"text": json.dumps({"edges": edges_by_id})}]}


async def test_cached_leg_fetcher_single_flight(monkeypatch):
    # Concurrent fetches for the SAME curie hit the underlying Kestrel call exactly once.
    calls = []

    async def fake_kestrel(tool, args):
        calls.append(args["start_node_ids"])
        await asyncio.sleep(0.01)  # keep the first call in-flight while the others arrive
        return _kestrel_resp({})

    monkeypatch.setattr(provenance, "call_kestrel_tool", fake_kestrel)
    fetch = cached_leg_fetcher(concurrency=8)
    await asyncio.gather(fetch("HGNC:1"), fetch("HGNC:1"), fetch("HGNC:1"))
    assert calls == ["HGNC:1"]  # deduped to a single in-flight fetch


async def test_cached_leg_fetcher_caches_repeat_fetches_distinct_curies(monkeypatch):
    # A repeated curie is fetched once; distinct curies are each fetched once.
    calls = []

    async def fake_kestrel(tool, args):
        calls.append(args["start_node_ids"])
        return _kestrel_resp({})

    monkeypatch.setattr(provenance, "call_kestrel_tool", fake_kestrel)
    fetch = cached_leg_fetcher()
    await fetch("A")
    await fetch("B")
    await fetch("A")
    assert calls == ["A", "B"]  # A cached on the second call


async def test_cached_leg_fetcher_bounds_concurrency(monkeypatch):
    # No more than `concurrency` underlying Kestrel calls run at once.
    inflight = 0
    peak = 0

    async def fake_kestrel(tool, args):
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return _kestrel_resp({})

    monkeypatch.setattr(provenance, "call_kestrel_tool", fake_kestrel)
    fetch = cached_leg_fetcher(concurrency=2)
    await asyncio.gather(*[fetch(f"N:{i}") for i in range(10)])
    assert peak <= 2


# --- leg_tier: injectable fetch (so the node can pass a cached fetcher) -----------------

async def test_leg_tier_uses_injected_fetch():
    # leg_tier classifies using the injected fetch's edges (no real Kestrel call).
    async def fetch(curie):
        return ([_edge("biolink:causes", "knowledge_assertion", "manual_agent",
                       subject="A", obj="B")], False)

    assert await leg_tier("A", "B", fetch=fetch) == "curated-causal"


async def test_leg_tier_injected_fetch_hub_retry():
    # When the start node is truncated with no X-Y edge, leg_tier retries from Y via the same fetch.
    async def fetch(curie):
        if curie == "HUB":
            return ([], True)  # truncated, no edge to Y found
        return ([_edge("biolink:causes", "knowledge_assertion", "manual_agent",
                       subject="Y", obj="HUB")], False)

    assert await leg_tier("HUB", "Y", fetch=fetch) == "curated-causal"
