---
title: "Verify per-edge publication-date provenance before designing a temporal holdout eval on an aggregated biomedical KG"
date: 2026-05-29
category: best-practices
module: discovery-pipeline
problem_type: best_practice
component: tooling
severity: medium
applies_when:
  - "Designing a temporal/time-sliced (Swanson ABC) holdout eval of gap or link prediction"
  - "Working against an aggregated KG derived from upstream sources (SPOKE, RTX-KG2)"
  - "Relying on an external KG-access API (REST/OpenAPI + MCP) you do not control"
  - "A plan asserts the KG is 'temporally grounded' or can be 'frozen at date T' without a verified per-edge date field"
related_components:
  - service_object
  - database
tags:
  - temporal-holdout
  - knowledge-graph
  - kestrel-api
  - provenance
  - link-prediction
  - swanson-abc
  - feasibility-analysis
  - openapi-probing
---

# Verify per-edge publication-date provenance before designing a temporal holdout eval on an aggregated biomedical KG

## Context

We wanted a temporal / time-sliced ("Swanson ABC") holdout eval for the KRAKEN discovery pipeline's **gap identification**: train on edges known before a cutoff date **T**, score the pipeline's predictions against edges published *after* T. The plan's D4 deepening note assumed the KG could be "frozen at date T" — i.e. that Kestrel edges carry per-edge literature dates.

That assumption was never verified against the live API. A prior hands-on Kestrel probe (`docs/ark_kg_investigation_report.md`, 2026-03-06) had even noted that `get_edges` returns a `publications` PMID list on some edges — **but stopped short of checking whether those PMIDs (or any field) gave a usable date, or whether temporal filtering existed** *(session history)*. So we probed it before building.

The finding: aggregated biomedical KGs carry **source attribution, not queryable per-edge publication dates**. A literature-date holdout is **not natively feasible** on Kestrel. Full evidence is in [`docs/kestrel-api-reference.md` § Temporal & provenance limitations](../../kestrel-api-reference.md) — this doc is the generalizable lesson, not a re-derivation of the numbers.

## Guidance

Before designing a temporal/time-sliced eval on an aggregated biomedical KG (SPOKE / RTX-KG2-derived), do these four things, in order:

1. **Verify per-edge date provenance exists AND is queryable — first.** Pull the OpenAPI spec, probe the constraint/traversal-options endpoint, and inspect a *real* edge with full attributes. Treat "temporally-grounded" and "has provenance" as unproven claims until you see a queryable date on an actual edge. Aggregated KGs almost always mean source attribution (`knowledge_source`, `agent_type`), not dates. A 15-minute probe settles it.

2. **Validate the eval harness on a natively-dated KG before porting.** Use a graph with built-in chronology — `ogbl-collab` / TGB 2.0 — to prove the *scoring code* is correct: chronological split, leakage-safe negatives, **AUC-PR as the primary metric** (ROC misleads under the ~99% sparsity of link-prediction graphs; add MRR / Hits@k). Then run the *actual pipeline* against a biomedical **dated** KG: SemMedDB (native per-PMID dates) or the pharmacogenomic KG (medRxiv 2025.09.24, which already does chronological splits + publication-date verification).

3. **Swap only the KG backend; keep the pipeline and reasoning model.** Stand up a parallel MCP server that mirrors Kestrel's tool contract (`one_hop_query`/`multi_hop_query`/`subgraph`/`get_edges`/search) plus publication dates and a cutoff, then repoint `KESTREL_MCP_URL`. Nodes run unchanged. Sequence this *after* the one/multi/subgraph reasoning is finalized — that reasoning *is* the gap-identifier under evaluation. (Active workstream: [`AGENT-TASK-temporal-eval-port.md`](../../../AGENT-TASK-temporal-eval-port.md).)

4. **Score under the Open World Assumption.** Recall against held-out post-T edges is trustworthy; an unpublished prediction is OWA-uncertain, not a false positive. Time-slice the literature corpus too (`literature_grounding` must not read post-T papers) to avoid circular evidence.

**Partial fallback if you must stay on the undated KG:** you can still get a *literature-recall* temporal signal without freezing the KG — date-filter the `literature_grounding` node's own sources (PubMed `pubdate`, OpenAlex `publication_year`) at T and score post-T literature. This tests literature recall, **not** KG-derived structural prediction, so it's a weaker proxy — document the distinction *(session history)*. Where edges carry only PMIDs, resolve PMID→date via PubMed eutils and use earliest-PMID as a documented, weakly-circular proxy for "first asserted."

## Why This Matters

The failure mode is silent and expensive: you design and partially build a temporal eval assuming dates exist, then discover mid-build that they don't. Worse, the naive fallback — restricting the holdout to the ~18% of edges that *do* carry publication info — produces a **biased benchmark**. The undatable ~82% (predictions, computed/aggregator edges) is precisely the class a discovery pipeline competes against; evaluating only on the datable subset rewards rediscovering already-published associations and hides the pipeline's real discovery value. A short API probe up front prevents both the wasted build and the misleading metric — and corrects plan assumptions before they propagate into implementation.

## When to Apply

- Designing any temporal, time-sliced, or "freeze at date T" holdout/backtest on a biomedical KG.
- The KG is *aggregated* from upstream sources (SPOKE, RTX-KG2, …) rather than a primary literature-derived store.
- A plan/design note asserts the KG is "temporally grounded," "has provenance," or "can be frozen at T" without an explicit, verified per-edge date field.
- Evaluating a discovery/gap-identification pipeline whose value claim is finding *not-yet-published* associations (so OWA scoring matters).

## Examples

The probe that settled it (Kestrel API v0.1.0, 2026-05-29). This sequence is the reusable method — adapt the host/auth:

```bash
# 1. Pull the spec — how many endpoints, what shape
curl -sL -H "X-API-Key: $KESTREL_API_KEY" \
  https://kestrel.nathanpricelab.com/api/openapi.json | jq '.paths | keys | length'   # → 22

# 2. Inspect what is actually CONSTRAINABLE (is any field temporal?)
curl -sL -H "X-API-Key: $KESTREL_API_KEY" \
  https://kestrel.nathanpricelab.com/api/traversal-options
#   → 17 fields (predicate, knowledge_level, agent_type, *_knowledge_source, provided_by,
#     qualifiers, confidence, degree, degree_percentile, edge_count, chemical_formula,
#     exact_mass, prefix, *_node_category, upstream_kg) — NONE temporal

# 3. Seed a real node, then inspect REAL edges with FULL attributes
curl -sL -H "X-API-Key: $KESTREL_API_KEY" -X POST .../api/text-search \
  -d '{"search_text":"type 2 diabetes mellitus"}'            # → MONDO:0005148
# POST /api/one-hop {"start_node_ids":["MONDO:0005148"],"limit":60}
# POST /api/get-edges {"edge_ids":[...],"slim":false}   # slim:false = full attributes
```

Result on a 120-edge sample around `MONDO:0005148`:
- **0 / 120** edges had any date field.
- **22 / 120 (~18%)** had `publications` (bare PMIDs, e.g. `["PMID:1360036"]`, undated).
- `knowledge_level`: 48 knowledge_assertion / 37 not_provided / 20 prediction / 11 logical_entailment / 4 statistical_association — the undatable ~82% is the discovery-relevant class.

Edge schema (`slim:false`): `[subject, predicate, object, qualifiers, primary_knowledge_source, supporting_sources, aggregator_knowledge_source, knowledge_level, agent_type, id]` — provenance by *source*, never by *date*.

**Verdict:** literature-date temporal holdout is not natively feasible on this aggregated KG → run the eval against a dated KG via a swapped MCP backend, or accept the biased ~18% subset and document it.

## Related

- [`docs/kestrel-api-reference.md`](../../kestrel-api-reference.md) — the live-probed evidence (§ Temporal & provenance limitations); source of the numbers above.
- [`AGENT-TASK-temporal-eval-port.md`](../../../AGENT-TASK-temporal-eval-port.md) — the active workstream implementing guidance #2–#4 (swap KG backend via parallel MCP server).
- [`docs/plans/2026-05-25-001-feat-kestrel-api-depth-plan.md`](../../plans/2026-05-25-001-feat-kestrel-api-depth-plan.md) — the **D4** deepening note ("freeze the KG/literature at date T") is **feasibility-corrected by this doc**; D6 supplies the AUC-PR-over-ROC metric; D3/D5 supply the OWA + circular-evidence cautions.
- [`docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md`](discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md) — the pipeline that runs *unchanged* against the swapped backend (distinct topic: reasoning topology, not eval feasibility).
- `docs/ark_kg_investigation_report.md` (2026-03-06) — the earlier probe that found `publications` PMIDs but did not check dates.
- ⚠️ Naming: the pipeline's existing **`temporal` node** (`backend/.../graph/nodes/temporal.py`, longitudinal causal classification) is unrelated to this **temporal/time-sliced eval**.
