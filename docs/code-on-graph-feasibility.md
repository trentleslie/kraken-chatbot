# Code-on-Graph → Kraken Discovery Pipeline: Feasibility Assessment

**Date:** 2026-06-03
**Paper:** Code-on-Graph: Iterative Programmatic Reasoning via LLMs on Knowledge Graphs — [arXiv:2606.03705](https://arxiv.org/abs/2606.03705) (Ding et al., ICT-CAS + Shandong U.)
**Verdict:** ✅ **recommend-prototype** — but scoped honestly as a *CoG-inspired iterative query-planning loop*, not a CoG integration. Measurement-gated.

---

## TL;DR

- **No CoG code is released.** No repo, no code-availability statement in the paper. This is a *reimplementation from a paper* (with unrecoverable hyperparameters: depth `d`, top-K, retry `N`, schema→class mapper), not a library integration.
- **The obvious objection is wrong:** CoG does *not* have the LLM write Cypher/SPARQL. It generates **Python over framework-retrieved objects**. Kestrel already exposes exactly that programmatic primitive set.
- **The real catch:** CoG's headline wins (≤10.5% Hits@1, ~47× token-utilization) come mostly from *keeping the KG out of the prompt* — benchmarked against PoG, which stuffs raw triples in. **Kraken already keeps the KG out of the prompt** (`allowed_tools=[]`, fetch-then-reason). So CoG's published numbers give ≈zero predictive evidence for kraken's payoff.
- **What's genuinely testable:** does *iterative query-refinement* beat kraken's *static hand-coded query plan* for bridge discovery? That's the narrow hypothesis worth a prototype.

---

## How CoG actually works

Three-phase loop (planning → coding → executing), repeated until an evaluator deems exploration sufficient:

1. **Planning** — a generator LLM proposes the next subtask; an evaluator decides continue/stop.
2. **Coding** — framework retrieves a bounded subgraph (depth-`d`, top-K edges ranked by DistilBERT relation similarity), maps KG schema → **Python classes** (entity types → classes, relations → typed attributes), and prompts the LLM to write Python grounded in those classes.
3. **Executing** — generated Python runs in a restricted sandbox over instantiated objects; on error/empty result, execution feedback drives self-correction up to `N` retries.

Key interface assumptions: (a) introspectable schema/types for class generation, (b) a framework-controlled entity-anchored neighborhood retrieval API, (c) a Python execution environment. **Not** LLM-authored graph queries.

Evaluated only on Freebase KGQA (WebQSP/CWQ/GrailQA), single-gold-answer Hits@1.

## Why it maps onto kraken

- **The KG backend is Kestrel** (`kestrel_client.py:KestrelClient`), reached via hand-rolled HTTP/JSON-RPC + SSE parsing, exposing `hybrid_search`, `text_search`, `one_hop_query`, `similar_nodes`, `multi_hop_query`, `subgraph_query`.
- **Today every node runs a STATIC query plan**; the LLM only summarizes pre-fetched data with `allowed_tools=[]`, `max_turns=1`. **There is no generate→execute→observe→iterate loop anywhere** — this is the gap CoG addresses.
- `kestrel_tools.py:create_kestrel_mcp_server` already wraps the full Kestrel tool surface but is **dormant** ("currently unused…kept for reference") — it's the closest existing scaffold to a code-on-graph executor's callable API.

## Recommended approach (corrected by adversarial review)

1. **Build a structured-spec executor, not an arbitrary-Python sandbox.** The LLM emits a typed JSON query spec (start/end CURIEs, `max_hops` 1–5, predicate filter, degree constraints, target categories). Captures ~80% of CoG's iterate-over-graph value while preserving kraken's no-arbitrary-execution safety posture (kraken has *no* sandbox today; free code execution reopens the failure modes issues #61/#44 engineered out).
2. **Start at a DISCOVERY site, not a validation site.** ⚠️ Do **not** prototype at `synthesis.py:validate_bridge_hypotheses`: a "retry until a path returns" loop on a *validation* step is confirmation-seeking and can inflate Tier-3→Tier-2 false upgrades. The R1a CURIE-membership gate blocks *fabricated* CURIEs but **not spurious-but-real degree-hub paths**. Prototype instead at **`integration.py:detect_bridges_via_api`** (discovery — extra recall is the goal; synthesis still validates conservatively downstream).
3. **Reuse the R1a grounding contract:** any CURIE the loop emits must reference a node actually returned by an executed Kestrel call.
4. **Define success before building** (no gold answer in a discovery task): bridge-recall vs one-shot baseline; **false-confirmation rate** (spurious-but-real paths upgraded); hallucinated-CURIE rate (must stay 0); per-discovery cost **distribution (p50/p95, not mean)** under a hard turn cap.

## Integration points (by effort)

| Node / file | Change | Effort |
|---|---|---|
| `kestrel_tools.py:create_kestrel_mcp_server` (dormant) | Re-enable as the executor's callable primitive surface | S |
| `entity_resolution.py:resolve_single_entity` | No change — its prefetch-then-select-with-grounding-gate is the **output-validation template** the loop must reuse | S |
| `integration.py:detect_bridges_via_api` (`max_pairs=3` hardcoded fan-out) | **First prototype target** — replace fixed category-pair fan-out with generated, iterative bridge-discovery queries | L |
| `direct_kg.py:analyze_via_api` (6-call fan-out: 3 categories × 2 presets) | Replace with entity-adaptive generated queries driven by triage edge counts (phase 2+; highest-volume site) | M |
| `pathway_enrichment.py:find_two_hop_shared_neighbors` | Generated set-intersection-over-graph code instead of fixed 2-hop loop | M |
| NEW `graph_reasoning` node in `builder.py:build_discovery_graph` | Dedicated code-on-graph node; needs matching `@validate_state` contracts in `state_contracts.py`; emit only `operator.add`-reduced fields (`bridges`, `direct_findings`, `errors`, `model_usages`) | L+ |
| `synthesis.py:validate_bridge_hypotheses` | Keep one-shot OR add explicit anti-confirmation guards (hard retry cap, log every spec, apply hub-degree filter, measure false-confirmation rate) | M |

## Risks

- **Reimplementation-from-paper** — unrecoverable hyperparameters; reported Freebase-QA gains will not transfer to an open-ended discovery task.
- **Sandbox safety** — literal CoG runs LLM-generated Python; kraken has no sandbox and a deliberate anti-hallucination contract. Mitigation: structured-spec executor.
- **Schema-to-class abstraction is partially *blocked*, not just hard** — Kestrel is remote-only (no local subgraph object, no local embedding index; all semantic ranking is server-side). CoG's "Python over a held in-memory subgraph" + local relation-ranking cannot be replicated without first materializing subgraphs locally (an unscoped sub-project).
- **Efficiency claim can invert** — CoG's TUR win is vs prompt-stuffing baselines; kraken's baseline is already lean, so an iterate-and-refine loop *adds* turns/calls. Measure, don't assume.
- **No checkpointer** in `build_discovery_graph` (plain `compile()`) → all iteration state must stay within a single node invocation or a self-looping subgraph.
- **Cost variance** — variable iteration count makes per-node cost a *distribution* problem; AstaBench budgeting needs p95 + a hard, tested turn cap.

## Open questions

- Does Kestrel's `multi_hop_query` already do semantic relation ranking (CoG's DistilBERT step), or would kraken need a relation-ranking layer? (Embeddings are server-side only today.)
- Structured-spec executor (sufficient for subgraph mechanistic extraction?) vs full LLM-generated-Python fidelity (richer compositional reasoning, but sandbox + local-materialization cost)?
- Concrete success metric for a no-gold-answer discovery task — must be fixed before any "better" claim.
- **Will the authors release code?** If a reference impl appears, reimplementation risk + hyperparameter uncertainty largely evaporate. → Added to the weekly code-watch list; worth a short watch before committing to a from-scratch build.

---

*Generated from a 5-agent feasibility workflow (paper read + CoG repo search + kraken pipeline map → synthesis → adversarial stress-test), 2026-06-03.*
