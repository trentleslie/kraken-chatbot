---
title: "feat: Bridge Evidence-Provenance Labeler"
type: feat
status: active
date: 2026-06-17
deepened: 2026-06-17
origin: backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md
---

# feat: Bridge Evidence-Provenance Labeler

## Overview

Attach a **deterministic evidence-provenance label** to each 3-node `Bridge`, replacing the abandoned
mechanism-confidence scorer. For each leg (A→B, B→C) the labeler reads the leg's KG edges and reports
the **best evidence tier** — `curated-causal` / `curated-associative` / `curated-neutral` / `text-mined`
/ `none` — derived from the edges' `knowledge_level`, `agent_type`, and Biolink predicate class. Each
bridge gets a chain summary (e.g. "both legs curated-causal" … "text-mined only" … "no KG edge"). The
label tells a researcher *what kind of evidence backs each leg*; it makes **no confidence claim**, so
there is **no score, no LLM, and no calibration gate** (nothing to falsify).

This supersedes the v1 co-occurrence scorer and the v2 KG-provenance scorer, both of which failed to
support a trustworthy per-bridge mechanism-confidence number (see origin findings). It reuses the
committed v1 substrate (U1 model/state/contracts, the before-synthesis node-wiring design, default-off
enablement, synthesis rendering) and the classification logic validated by the committed probes.

## Problem Frame

Three Kestrel probes established that a per-bridge mechanism-confidence *score* is not achievable from
available signals (v1 literature AUC failed; v2 KG-provenance AUC 0.61; only 23% of real bridge legs are
curated-causal). But the KG provenance is genuinely useful as **honest transparency**. The reframe
(origin: `backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md`) drops the score and surfaces the
provenance label the data *can* support deterministically.

## Requirements Trace

- **R1.** For each ordered 3-node bridge leg, classify the best evidence tier from the leg's KG edges'
  `knowledge_level` + `agent_type` + predicate class (curated-causal / curated-associative /
  curated-neutral / text-mined / none). → L1
- **R2.** Attach a per-bridge chain-summary label composed from the two legs' tiers. → L1, L2
- **R3.** Deterministic only — no score, no LLM, no co-occurrence retrieval, no calibration gate. → all
- **R4.** Reuse U1 (`BridgeGrounding`/`grounded_bridges`/contracts) and the v1 plan U5 node-wiring design
  (before-synthesis, default-off, synthesis renders the label). → L2, L3
- **R5.** Ship `enabled=False` (eval-only) until the entity-resolution wrong-namespace fix lands (label
  accuracy depends on correct CURIEs — see the BioMapper wiki report). → L3

## Scope Boundaries

- Single-bridge labeling only; ordered 3-node multi-hop bridges (exclude subgraph "connecting" bridges).
- **No** confidence score, ranking, LLM confirmation, co-occurrence retrieval, or Tier A calibration gate.

### Deferred to Separate Tasks

- Entity-resolution wrong-namespace fix (the accuracy dependency) — separate BioMapper/kraken work; see
  `docs/wiki/entity-resolution-namespace-fix.outline.md`.
- Per-bridge UI-row rendering in `node_detail_extractors` (v1.1) — synthesis-report rendering is in scope.

## Context & Research

### Relevant Code and Patterns (verified firsthand this session)

- `backend/assessment_data/kg_bridge_leg_probe.py` / `kg_provenance_probe.py` — the **classification +
  per-leg-fetch logic to productionize** (`pred_class`, `is_curated`, the one_hop-full leg read). These
  probes ARE the labeler in script form.
- `backend/src/kestrel_backend/kestrel_client.py` — `call_kestrel_tool`. Per-leg edges via
  `call_kestrel_tool("one_hop_query", {"start_node_ids": X, "mode": "full"})` then **filter the
  returned `edges` dict client-side to those touching Y** — this is the call the working probe used
  (`kg_provenance_probe.py` `direct_edges`). **Server-side `end_node_ids` on `one_hop_query` is
  UNVERIFIED** — adopt the start-only + post-filter approach, or spike-confirm `end_node_ids` first;
  do NOT rely on it as written. Full-mode edges are rich **dicts** with `predicate`, `knowledge_level`,
  `agent_type`, `subject`, `object` (verified); `primary_knowledge_source` is expected but not yet
  exercised by the probes — confirm it is present in the dict during L1. **Do NOT use
  `multi_hop_query(max_path_length=1)`** — Kestrel rejects it (must be ≥2).
- `backend/src/kestrel_backend/graph/state.py` — `Bridge` (frozen; `predicates`/`predicate_directions`
  from U0), `BridgeGrounding`/`LegSummary` (U1), `grounded_bridges` (no-reducer), `bridge_grounding_errors`.
- `backend/src/kestrel_backend/graph/state_contracts.py` — `BridgeGroundingInput/Output` + registry (U1).
- `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md` U5 — the before-synthesis node-wiring
  design (reroute `temporal/integration → bridge_grounding → synthesis`, full graph only; default-off;
  synthesis renders; `protocol.total_nodes` 10→11; `NODE_STATUS_MESSAGES` entry). Substrate-independent.
- `backend/src/kestrel_backend/graph/nodes/integration.py` — produces ordered multi-hop bridges
  (`predicates`/`predicate_directions` populated by U0) AND subgraph "connecting" bridges. **Discriminator
  (robust, documented at `state.py` Bridge docstring): `predicate_directions` is non-empty ONLY for
  multi-hop bridges** (subgraph leaves it `[]`); use `predicate_directions != []` (or `len(entities)==3
  and predicate_directions`) to select scoreable bridges — prefer this over matching `path_description`.

### Institutional Learnings

- `[[bridge-grounding-v1-fails-calibration]]` — why the score was dropped (this reframe).
- `[[biomapper2-wraps-kestrel-hgnc-marker]]` — resolution returns wrong-namespace CURIEs; the label is
  only as good as the CURIEs, hence default-off until the resolution fix.
- `[[persist-expensive-run-artifacts]]` — any eval run over real bridges auto-persists.

## Key Technical Decisions

- **Label, not score; no LLM, no calibration gate.** The signal can't support a confidence number; it can
  support an honest provenance label. This removes the entire scoring/LLM/Tier-A surface.
- **Fetch per-leg edges fresh via `one_hop_query` full mode; classify the dict edges directly.** Avoids
  extending `parse_kestrel_response` (which handles multi_hop *compact tuples*, not full-mode dicts) — no
  change to shared parsing code. The fetch returns ALL A-B edges, so "best tier over candidates" is natural.
- **Static predicate-class set (the probe's), not `bmt`.** `bmt` is not a backend dependency; the static
  causal/associative/neutral set classified correctly in all three probes.
- **Curated = `knowledge_level ∈ {knowledge_assertion, logical_entailment}` AND `agent_type ∈
  {manual_agent, manual_validation_of_automated_agent}`.** Text-mined (`agent_type=text_mining_agent` /
  `knowledge_level=not_provided`) is its own lower tier — a text-mined `causes` is NOT curated-causal.
- **Ship default-off.** Label accuracy is gated on correct CURIEs (resolution fix); eval-only until then.

## Open Questions

### Resolved During Planning

- *Where does provenance extraction live?* — In the labeler (reads one_hop-full dict edges), not
  `parse_kestrel_response`. Confirmed by the probes.
- *How to fetch a single leg's edges?* — `one_hop_query` full mode with `start_node_ids`/`end_node_ids`
  (multi_hop max_path_length=1 is rejected).

### Deferred to Implementation

- The exact chain-summary label vocabulary (e.g. how many named buckets) — settle against real bridge
  output in L1's eval, but the per-leg tier set (R1) is fixed.
- Whether to also persist the per-hop tier on the `Bridge` (vs re-fetch) for cost — start with re-fetch
  (simpler, matches probe); optimize only if the per-run Kestrel call count is a problem.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
bridge (ordered A→B→C, multi-hop only):
  for each leg (X,Y):
     all_edges = one_hop_query(start_node_ids=X, mode=full)   # start-only (the PROVEN probe call)
     edges     = [e for e in all_edges if Y in (e.subject, e.object)]   # client-side filter to Y
     tier  = best over edges of evidence_tier(edge)           # curated-causal > curated-assoc >
                                                              #   curated-neutral > text-mined > none
  chain_label = summarize(leg1_tier, leg2_tier)               # "both legs curated-causal" … "no KG edge"
  grounded_bridges += bridge.model_copy(update={grounding: BridgeGrounding(legs=[...], label=chain_label)})
→ synthesis renders the per-bridge label (reuse v1 before-synthesis wiring)

evidence_tier(edge):                                          # deterministic; no LLM
  curated = knowledge_level ∈ {assertion, logical_entailment} AND agent_type ∈ {manual_*}
  cls     = predicate_class(predicate)   # causal | associative | neutral (static Biolink set)
  → "curated-" + cls if curated else ("text-mined" if cls != none else "none")
```

## Implementation Units

- [ ] **L1: Provenance classification + per-leg labeler**

**Goal:** Productionize the probe's logic: classify an edge's evidence tier, fetch a leg's edges and
take the best tier, and compose a bridge chain-summary label.

**Requirements:** R1, R2, R3

**Dependencies:** None (reuses `call_kestrel_tool`).

**Files:**
- Create: `backend/src/kestrel_backend/bridge_grounding/provenance.py` — `predicate_class`,
  `is_curated`, `evidence_tier(edge)`, `leg_tier(curie_a, curie_b)` (async, one_hop-full fetch),
  `bridge_label(legs)`.
- Test: `backend/tests/test_bridge_provenance.py`

**Approach:**
- Lift `pred_class`/`is_curated`/the one_hop-full leg read from `assessment_data/kg_bridge_leg_probe.py`.
  **Use that probe's predicate-class set** (the two probes differ — `biolink:treats`/`applied_to_treat`
  are **associative** there, causal in `kg_provenance_probe.py`; treat them as associative so a
  curated drug→disease `treats` edge is NOT counted as a curated-causal *mechanism* — see the v2
  finding that `treats`-as-causal inflated coverage).
- `evidence_tier` is a pure function over an edge dict; `leg_tier` does the start-only one_hop-full
  fetch + client-side filter to the target curie + best-of-candidates; `bridge_label` maps the two leg
  tiers to a named chain summary (vocabulary settled here).
- Best-tier ordering: curated-causal > curated-associative > curated-neutral > text-mined > none.

**Execution note:** Implement the pure classifiers (`predicate_class`/`evidence_tier`/`bridge_label`)
test-first; the fetch (`leg_tier`) is a thin Kestrel wrapper tested with a mocked `call_kestrel_tool`.

**Patterns to follow:** the committed probes; `bridge_grounding/scoring.py` (pure-function + test style).

**Test scenarios:**
- Happy: an edge with `knowledge_assertion`/`manual_agent`/`biolink:causes` → `curated-causal`.
- Edge: `text_mining_agent`/`biolink:causes` → `text-mined` (NOT curated-causal).
- Edge: `manual_agent`/`biolink:associated_with` → `curated-associative`; `…/biolink:related_to` → `curated-neutral`.
- Edge: a leg with multiple edges → best tier wins (curated-causal over text-mined).
- Boundary: a leg with no edges → `none`.
- Happy (compose): (curated-causal, curated-causal) → "both legs curated-causal"; (curated-causal, none)
  → "one leg unsupported"; (none, none) → "no KG edge".
- Integration: `leg_tier` calls `one_hop_query` with `start_node_ids`/`mode:full` (mocked), then
  filters the returned dict edges client-side to those touching the target curie.

- [ ] **L2: BridgeGrounding model adaptation (provenance label fields)**

**Goal:** Repurpose the U1 model to carry the per-leg tier + chain label instead of co-occurrence tallies.

**Requirements:** R2, R4

**Dependencies:** L1; reuses U1 (`BridgeGrounding`, `LegSummary`, `grounded_bridges`, contracts).

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py` — `LegSummary` carries `from_curie`/`to_curie`/
  `predicate`/`evidence_tier`/`knowledge_level`/`agent_type`/`source`; `BridgeGrounding` carries
  `legs`/`label` (chain summary). **Not fully additive:** `BridgeGrounding.support_fraction` and
  `decision` are currently **required** — to carry only `legs`+`label`, make them optional/defaulted or
  remove them (the score is gone). The frozen-model + `model_copy` attach pattern is unchanged.
- Test: `backend/tests/test_bridge_grounding_models.py` (rewrite U1's tests — they construct
  `BridgeGrounding(support_fraction=…, decision=…)` and will break on the field change).

**Approach:** Reuse the no-reducer `grounded_bridges` key, the contracts, and the `entities`-tuple join
from U1 unchanged. **Runtime-safe:** no production reader and none of the v1 scorer modules import
`BridgeGrounding`/`LegSummary` (they operate on raw dicts) — the only breakage is
`test_bridge_grounding_models.py`, rewritten here.

**Test scenarios:**
- Happy: a provenance `BridgeGrounding` (legs + label) round-trips (frozen, additive-safe serialization).
- Edge: `model_copy` attach doesn't mutate the frozen original; join `grounded_bridges`→`bridges` by `entities`.

- [ ] **L3: `bridge_grounding` node + wiring + synthesis rendering**

**Goal:** A `bridge_grounding` node that labels ordered 3-node bridges and attaches `grounded_bridges`;
wired before synthesis; default-off; synthesis renders the per-bridge label.

**Requirements:** R3, R4, R5

**Dependencies:** L1, L2; reuses the v1 plan U5 wiring design.

**Files:**
- Create: `backend/src/kestrel_backend/graph/nodes/bridge_grounding.py` (`@validate_state`; ordered-3-node
  + exclude-subgraph filter; per-bridge `bridge_label` via L1; returns `grounded_bridges` +
  `bridge_grounding_errors`; best-effort, never throws).
- Modify: `backend/src/kestrel_backend/graph/builder.py` (reroute `temporal`/`integration →
  bridge_grounding → synthesis`, full graph only; leave post-synthesis `literature_grounding`).
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` (`BridgeGroundingConfig`,
  `enabled=False` default, per-leg fetch limit, exclude-subgraph flag — mirror `IntegrationConfig`).
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py` (read `grounded_bridges`, render the
  per-bridge provenance label).
- Modify: `backend/src/kestrel_backend/protocol.py` (add `NODE_STATUS_MESSAGES["bridge_grounding"]`;
  `total_nodes` 10→11).
- Test: `backend/tests/test_bridge_grounding_node.py` (sys.modules stub harness, per v1 plan U5).

**Approach:** Mirror the v1 plan U5 wiring decisions (substrate-independent); the node calls L1's
`bridge_label` per bridge instead of any scorer. Per-bridge isolation: a Kestrel failure on one bridge
degrades it (label `none`/error in `bridge_grounding_errors`), node still returns.

**Test scenarios:**
- Happy: a 3-node bridge with curated-causal legs → label attached, "both legs curated-causal".
- Edge: `enabled=False` short-circuits (no Kestrel calls); skips subgraph/2-node/4+-node bridges.
- Error: a `one_hop_query` failure on one bridge → that bridge labeled `none` + `bridge_grounding_errors`
  entry, node still returns.
- Integration: rewired graph routes through the node; `total_nodes`/`NODE_STATUS_MESSAGES` updated;
  synthesis renders the label; disabled = no-op.

## System-Wide Impact

- **Interaction graph:** `bridge_grounding` inserts before synthesis (reuse v1 wiring); new Kestrel load
  (2 `one_hop_query` full calls per scored bridge) — bounded by `max_scored_bridges`; node default-off.
- **Error propagation:** per-bridge isolation → label `none` + `bridge_grounding_errors`; node never throws.
- **State lifecycle:** `grounded_bridges` no-reducer key, single writer before synthesis reads it (U1).
- **Unchanged invariants:** `Bridge`/`grounded_bridges`/contracts stay backward-compatible (field swap on
  existing optional `BridgeGrounding`); `parse_kestrel_response` untouched (labeler reads one_hop dicts).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Label accuracy depends on correct CURIEs; resolution returns wrong-namespace CURIEs | Ship `enabled=False`; gate production enablement on the entity-resolution fix (BioMapper wiki report); a mis-resolved leg labels `none`, surfaced honestly |
| Most real bridges label `text-mined`/`none` (only ~23% legs curated-causal) | This is the honest, intended behavior — the label reports what evidence exists; it does not claim mechanism. Not a bug |
| Per-leg `one_hop_query` full fetches add Kestrel load | Bounded by `max_scored_bridges` + default-off; optimize to carry provenance on the `Bridge` (deferred) only if needed |
| Chain-summary label vocabulary churns | Settle the vocabulary in L1 against real bridge output before wiring L3 |

## Phased Delivery

The entity-resolution fix is a hard accuracy blocker with no committed timeline, and the node ships
default-off, so the in-pipeline node (L3) buys nothing observable until that fix lands. Sequence
accordingly so value ships now without premature graph surgery:

### Phase 1 (now) — L1 only: the deterministic labeler + eval
Build L1 (the pure `evidence_tier`/`predicate_class`/`bridge_label` classifiers + the per-leg
`leg_tier` fetch) and run it as **eval tooling** over real discovered bridges (reuse the
`kg_bridge_leg_probe` harness) to confirm the label distribution is sensible. This delivers the honest
provenance signal immediately, with zero changes to the production graph, the state model, or synthesis.

### Phase 2 (gated on the resolution fix being in a known release) — L2 + L3
Adapt the model (L2) and wire the node + synthesis rendering + enablement (L3) only once the resolution
fix is landing, since the in-pipeline label is both useless and inaccurate before then. **Trigger:** the
biomapper/kraken resolution fix (`docs/wiki/entity-resolution-namespace-fix.outline.md`) is merged and
verified; until then L2/L3 stay unbuilt rather than wired-but-disabled. Trim `BridgeGroundingConfig` to
what actually varies at runtime (`enabled`, per-leg fetch limit) — the exclude-subgraph behavior is a
fixed requirement, not a toggle.

**Superseded-apparatus cleanup:** the v1 (U0–U4 + harness) and v2 (superseded) scorer code remains on
the branch as the negative-result record. Decide its disposition when this work goes to a PR against
`dev` — either delete the scorer/LLM/co-occurrence modules (`bridge_grounding/{retrieval,prompts,
labeling,scoring,panel}.py` + their tests) or keep them explicitly as eval-only, so a reviewer is not
left navigating three scoring paths.

## Documentation / Operational Notes

- Node ships `enabled=False` (eval-only) until the resolution fix lands.
- An optional eval over real discovered bridges (reuse the `kg_bridge_leg_probe` harness) can sanity-check
  the label distribution — this is a *sanity check, not a calibration gate* (no confidence claim to validate).

## Sources & References

- **Origin document:** `backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md`
- Probes (the logic to productionize): `backend/assessment_data/kg_bridge_leg_probe.py`,
  `kg_provenance_probe.py`, `kg_bridge_graded_probe.py`
- v1 plan (reused U1 model + U5 wiring design): `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md`
- Resolution dependency: `docs/wiki/entity-resolution-namespace-fix.outline.md`
- Branch: `feat/bridge-grounding-scorer` (v1 U0–U4 + harness committed; v2 scorer superseded).
