---
title: "Discovery pipeline reasoning architecture: one graph, methods within nodes"
date: 2026-05-28
last_updated: 2026-06-22
category: docs/solutions/best-practices
module: discovery-pipeline
problem_type: best_practice
component: service_object
severity: medium
applies_when:
  - Adding or modifying a node in build_discovery_graph
  - Deciding whether a new reasoning approach needs its own graph or a method within a node
  - Choosing between multi_hop_query and sequential one_hop_query for a Kestrel call
  - Implementing or reviewing a node's Tier 1 direct-endpoint / Tier 2 Claude-SDK fallback
related_components:
  - assistant
  - tooling
tags:
  - langgraph
  - kestrel-client
  - multi-hop-query
  - reasoning-architecture
  - knowledge-graph
  - graph-routing
  - astabench
---

# Discovery pipeline reasoning architecture: one graph, methods within nodes

## Context

KRAKEN's discovery pipeline (`kraken-chatbot/backend`) runs as **one overarching LangGraph graph**, not as a set of separate pipelines per reasoning approach. `build_discovery_graph()` in `backend/src/kestrel_backend/graph/builder.py` builds a single `StateGraph(DiscoveryState)` with 13 nodes:

```
intake → entity_resolution → triage → [direct_kg | cold_start] → pathway_enrichment
       → integration → [temporal?] → hypothesis_extraction → bridge_grounding
       → literature_grounding → synthesis → reporting → END
```

> _Updated 2026-06-22: grew from the original 10 nodes to 13. The "ground-before-synthesis"
> reorg (PR #79) inserted `hypothesis_extraction → bridge_grounding → literature_grounding`
> **before** `synthesis` (so hypotheses are extracted, evidence-labeled, and literature-grounded
> before the report is written — `synthesis` no longer validates bridges itself), and the
> per-node performance report (PR #84) added a terminal `reporting` node after `synthesis`. The
> one-graph / methods-within-nodes teaching below is unchanged; only the node set and order moved._

Two recurring misconceptions prompted this note:

1. That each "reasoning approach" (direct KG traversal, cold-start, multi-hop pathfinding, subgraph) is its own pipeline.
2. That "multi-hop is just sequential one-hop."

Neither is true. Reasoning approaches are **methods invoked within nodes**, and `multi_hop_query` is a real server-side Kestrel endpoint, not a loop of one-hop calls. This matters most when extending the pipeline: the instinct to "add a new pipeline" is wrong — you add a method call within a node (and possibly a routing function), all inside the same graph.

## Guidance

**Approach selection happens at two distinct levels — keep them separate in your head:**

1. **Graph-level conditional routing BETWEEN nodes.** *Which* nodes run is decided by routing functions, not by the nodes themselves:
   - `route_after_triage` (`builder.py:27`) returns `["direct_kg","cold_start"]` (both, parallel), `"direct_kg"`, `"cold_start"`, or `"pathway_enrichment"` based on entity sparsity class.
   - `route_after_integration` (`builder.py:59`) returns `"temporal"` vs `"synthesis"` based on `is_longitudinal`.
   - When a router returns a list, LangGraph runs those nodes in the **same superstep** (parallel). Their concurrent writes merge safely via `operator.add` reducers (`state.py:309–341`). Both branches converge to `pathway_enrichment` (`builder.py:154–155`).

2. **Within-node API-method selection.** Each node calls whichever `kestrel_client` methods it needs (`one_hop_query`, `multi_hop_query`, `/subgraph`, `vector_search`, …). The reasoning "approach" is *which methods/presets a node picks*, not *which graph it lives in*.

**`multi_hop_query` is a real endpoint, already used in three nodes.** `backend/src/kestrel_backend/kestrel_client.py:340` wraps the server-side Kestrel endpoint (API v1.16.0; the wrapper maps `max_hops`→`max_path_length` and `predicate_filter`→`predicate` at `:393`/`:401`). Two modes:
- **Singly-pinned** (start only → explore N hops): `pathway_enrichment` (`:201`, `max_hops=2`) for shared-neighbor discovery.
- **Doubly-pinned** (start + end → connecting paths): `integration`'s `detect_bridges_via_api` (`:172`) for cross-type bridge detection, and `hypothesis_extraction`'s `validate_bridge_hypotheses` loop for Tier-3 bridge-hypothesis validation (this validation moved out of `synthesis` in the ground-before-synthesis reorg, PR #79). *(session history)*

That it is a real endpoint — and version-sensitive — is confirmed by the prior fix `3afcd99 "fix: correct multi_hop_query parameter names for Kestrel API v1.16.0"`. *(session history)*

**The two-tier node pattern — and the gotcha.** Each analysis node has two tiers:
- **Tier 1 (primary):** calls the HTTP endpoint directly via `call_kestrel_tool` / `kestrel_client`. This is what the node *actually does* in the common path.
- **Tier 2 (SDK fallback):** hands a Claude Agent SDK session a set of `allowed_tools` (e.g. `["mcp__kestrel__one_hop_query","mcp__kestrel__get_nodes"]`) and lets the LLM decide which to call. This is **LLM-directed tool use, not a hardcoded sequential loop.** In some nodes the fallback prompt *steers* the agent toward one-hop — e.g. `pathway_enrichment.py:364,368,376` instructs "Use one_hop_query for each entity … Focus on ONE-HOP neighbors only" (because that node's two-hop work is already done in Tier 1).

**To know what a node really does, read its Tier-1 path — not the Tier-2 fallback's tool grant or prompt.** Concluding "this node fakes multi-hop with one-hop" by reading the Tier-2 `one_hop_query` tooling is the misread that generates the "multi-hop == sequential one-hop" myth. The Tier-1 path of those same nodes calls the real `multi_hop_query` endpoint.

**State-field design rule.** A field written by a single node *before* the parallel fork and only read afterward should be a **plain dict field** — no reducer. Only fields written *concurrently by parallel branches* need `Annotated[list[X], operator.add]` (e.g. `direct_findings`/`cold_start_findings` at `state.py:322–323`, vs. `final_report` which deliberately omits the reducer at `:348`). The forthcoming `tool_strategies` field (see below) follows the plain-dict rule.

## Why This Matters

- **Prevents architectural sprawl.** Treating each approach as its own pipeline would fork the graph, duplicate state plumbing, and break LangGraph's single-superstep parallelism. The single-graph design keeps one source of truth for topology and lets the router decide what runs.
- **Stops a recurring misdiagnosis.** Reading the Tier-2 fallback's `one_hop_query` tooling and concluding multi-hop is faked leads engineers either to "fix" something that isn't broken or to distrust the real endpoint. Read Tier 1 first.
- **Avoids reducer bugs.** Adding an `operator.add` reducer to a single-writer field causes silent accumulation/duplication; omitting one on a concurrent-write field causes lost updates from parallel branches. The before-fork/after-read heuristic gets this right by construction.

## When to Apply

- **Extending the pipeline's reasoning.** R5/Unit 5 of the kestrel-api-depth plan (`docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`) adds multi-hop to `direct_kg` for mechanistic chains (Drug→Gene→Pathway→Disease). The right move is to add a `multi_hop_query` call **within `direct_kg`**, plus routing if a new entity-class branch is needed — **not** a parallel graph.
  - Current gap this closes: `direct_kg` today uses **only** `one_hop_query` across two presets — `PRESETS = ["established", "hidden_gems"]` (`direct_kg.py:47`) — and no multi-hop. (Tracking: GitHub issue **#27**, "Integrate Kestrel multi_hop_query API for deeper graph traversal" — reconcile its status against the API-depth plan; multi-hop is a *new use location* in `direct_kg`, not a new capability.)
- **Adding a triage-emitted strategy field.** The active plan has Triage emit a per-entity `tool_strategies: dict[CURIE → ToolStrategy]` (`ranking_presets`, `use_multi_hop`, `search_mode`). Because Triage writes it **once, before** the parallel fork and it is **read-only** downstream, model it as a **plain dict — no `operator.add`**. This is "logic within node," reinforcing the single-graph design. *(Note: `tool_strategies`/`ToolStrategy` are not yet in `state.py`/`triage.py` — they are forthcoming per the plan.)*
- **Debugging graph topology.** Run `langgraph dev` from `backend/` to open LangGraph Studio; `backend/langgraph.json` registers `kraken_discovery: kestrel_backend.graph.builder:build_discovery_graph`. (Studio config gotcha documented separately — see Related.)
- **Determining what any node does.** Always read the **Tier-1 primary** path before trusting a conclusion drawn from the Tier-2 fallback prompt.

## Examples

**The 13-node graph (single StateGraph):**

```
                    intake
                      │
                entity_resolution
                      │
                    triage
                      │  route_after_triage  (entity sparsity class)
        ┌─────────────┼─────────────┐
        ▼             ▼             │  (no entities resolved)
   direct_kg     cold_start         │
   (one_hop ×2:  (sparse/cold)      │
    established,                    │
    hidden_gems) ── parallel ──     │
        └─────────────┴─────────────┘
                      ▼
              pathway_enrichment   ← multi_hop_query, singly-pinned, max_hops=2  (Tier 1)
                      │
                 integration       ← multi_hop_query, doubly-pinned (bridge detection)
                      │  route_after_integration  (is_longitudinal?)
        ┌─────────────┴─────────────┐
        ▼ (if longitudinal)         │ (else)
    temporal ─────────────────────► ▼
                          hypothesis_extraction  ← multi_hop_query, doubly-pinned (Tier-3 validation)
                                    │
                            bridge_grounding      (deterministic evidence-provenance labels)
                                    │
                          literature_grounding    (ground hypotheses before the report)
                                    │
                                synthesis         (module-aware, token-bounded report)
                                    │
                                reporting         (per-node performance report)
                                    │
                                   END
```

**Wrong mental model vs. correct model:**

```
WRONG — "a pipeline per approach":
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ direct-KG graph  │  │ cold-start graph │  │ multi-hop graph  │
   │ (own StateGraph) │  │ (own StateGraph) │  │ (own StateGraph) │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
   → "multi-hop is just sequential one-hop in its own pipeline"

CORRECT — "methods within ONE graph":
   ONE StateGraph(DiscoveryState)
     • routing functions choose WHICH NODES run (route_after_triage / _integration)
     • each NODE chooses WHICH kestrel_client METHODS it calls
         - one_hop_query, multi_hop_query (real server endpoint), vector_search, /subgraph
     • multi_hop_query is a real Kestrel API call, NOT a one-hop loop
     • the one-hop wording lives only in a Tier-2 SDK FALLBACK prompt (LLM-directed)
```

**The two-tier gotcha, concretely (pathway_enrichment):**

```
Tier 1 (primary, what actually runs):
    multi_hop_query(start_node_ids=[curie], max_hops=2)   # kestrel_client.py:340, called at pathway_enrichment.py:201
    → real server-side multi-hop endpoint

Tier 2 (SDK fallback, only if Tier 1 path unavailable):
    Claude agent, allowed_tools=["mcp__kestrel__one_hop_query", ...]  # pathway_enrichment.py:376
    prompt steers: "Use one_hop_query for each entity ... Focus on ONE-HOP neighbors only"  # :364,:368
    → LLM-directed, NOT a hardcoded loop; this prompt is the source of the multi-hop confusion
```

**Key file anchors:**
- `backend/src/kestrel_backend/graph/builder.py:74` — `build_discovery_graph`; routers at `:27` and `:59`
- `backend/src/kestrel_backend/kestrel_client.py:340–403` — `multi_hop_query` wrapper (param remap at `:393`,`:401`)
- `backend/src/kestrel_backend/graph/nodes/pathway_enrichment.py:201` (Tier 1 multi-hop), `:364–378` (Tier 2 fallback)
- `backend/src/kestrel_backend/graph/nodes/integration.py:172` — `detect_bridges_via_api` (doubly-pinned)
- `backend/src/kestrel_backend/graph/nodes/direct_kg.py:47` — `PRESETS = ["established","hidden_gems"]` (one-hop only; the multi-hop gap)
- `backend/src/kestrel_backend/graph/state.py:309–348` — reducer vs. plain-field examples
- `backend/langgraph.json` — Studio graph registration

## Related
- `docs/discovery-pipeline.md` — canonical architecture diagram, node descriptions, routing, semaphore docs (reflects the pre-kestrel-api-depth 10-node state).
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — formalization of this same pipeline (state contracts, per-node config, variance bands); architecture context for this note.
- `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md` — concrete Tier-1/Tier-2 fallback example; the Tier-2 path needs explicit test coverage (it once shipped a NameError).
- `docs/solutions/best-practices/langgraph-json-src-layout-import-2026-05-06.md` — `langgraph.json` must use dotted-import (not file-path) form for the same `build_discovery_graph` entry point under a `src/` layout.
- `docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md` — R5/Unit 5 (multi-hop in `direct_kg`), `tool_strategies` design, and the D1–D6 deepening note.
- GitHub issue **#27** — multi_hop_query integration tracking.
