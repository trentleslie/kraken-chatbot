---
title: "feat: Bridge Evidence-Provenance Labeler"
type: feat
status: active
date: 2026-06-17
deepened: 2026-06-18
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

**The label drives a differential read** (its reason to exist over no-label): `curated-causal` legs point
to a mechanistic link a curator asserted — trust it as a starting mechanism hypothesis; `text-mined` legs
are co-mention-derived — treat as hypothesis-generating, needs literature follow-up before relying on it;
`none` means the KG offers no direct edge — the bridge is speculative on that leg. Most discovered bridges
will be mostly `text-mined`/`none` (only ~23% of legs are curated-causal); that is the honest signal, and
the differential above is what makes surfacing it worthwhile rather than noise. L4 spot-checks this.

This supersedes the v1 co-occurrence scorer and the v2 KG-provenance scorer, both of which failed to
support a trustworthy per-bridge mechanism-confidence number (see origin findings). It reuses the
committed v1 substrate (U1 model/state/contracts, the before-synthesis node-wiring *design* from the v1
plan U5) and the classification logic validated by the committed probes. The `enabled=False` toggle and
synthesis rendering are **new** to this work (the v1 scorer had no config toggle and never wired into
synthesis) — see L3.

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
- **R5.** Ship `enabled=False` and flip to enabled only after L4's validation eval (against a pre-committed
  baseline, R5/L4) confirms the label distribution. The accuracy gate — the Tier 1 entity-resolution
  wrong-namespace fix — **has now landed** (PR #75, merged to `dev`: category-constrained `resolve_via_api`
  + disease intake hint), so discovered bridges resolve to correct-namespace CURIEs for the dominant cases.
  **Honest caveat:** the within-category canonical-ranking residual (Tier 2, biomapper2, still in flight)
  is **partially silent** — a leg resolving to a real-but-non-canonical same-category node (e.g. ICD
  instead of MONDO) fetches *that* node's edges and may yield a `none`/`text-mined` tier indistinguishable
  from a genuinely sparse leg. This is *why* the node ships default-off and the flip is gated on the
  baseline-band eval rather than on Tier 1 alone; a material distribution shift is the aggregate signal for
  the otherwise-silent residual. → L3, L4

## Scope Boundaries

- Single-bridge labeling only; ordered 3-node multi-hop bridges (exclude subgraph "connecting" bridges).
- **No** confidence score, ranking, LLM confirmation, co-occurrence retrieval, or Tier A calibration gate.

- **In scope, as a separate pre-flight cleanup PR (PR-A, unit C1):** delete the superseded v1/v2 scorer
  apparatus (`bridge_grounding/{retrieval,prompts,labeling,scoring,panel}.py` + their tests). No production
  reader; the negative result is preserved in the findings docs, the `kg_bridge_*` probes (NOT deleted),
  and git history. Ships **before** the feature PR (PR-B = L2 + L3) so the deletion has its own revert
  handle and the feature diff stays focused. Sequencing it first also means L2's model change no longer
  has the scorer modules as live constructors to worry about.

### Deferred to Separate Tasks

- Tier 2 within-category canonical / species ranking (MONDO-over-ICD, human/HGNC) — biomapper2, in
  flight in parallel; `../biomapper2/AGENT-TASK-canonical-namespace-preference.md`. Not a blocker (its
  residual is a surfaced label degradation, not a silent error).
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
  only as good as the CURIEs. Tier 1 (PR #75) cleared the cross-namespace blocker; ships `enabled=False`
  pending L4's validation eval (Tier 2 within-category residual still in flight).
- `[[persist-expensive-run-artifacts]]` — any eval run over real bridges auto-persists.

## Key Technical Decisions

- **Label, not score; no LLM, no calibration gate.** The signal can't support a confidence number; it can
  support an honest provenance label. This removes the entire scoring/LLM/Tier-A surface.
- **Fetch per-leg edges fresh via `one_hop_query` full mode; classify the dict edges directly.** Avoids
  extending `parse_kestrel_response` (which handles multi_hop *compact tuples*, not full-mode dicts) — no
  change to shared parsing code. The fetch returns ALL A-B edges, so "best tier over candidates" is natural.
  **Why re-fetch (not thread integration's edges) is correct, not just convenient:** the label answers
  *"what evidence exists between this resolved node pair?"* — a property of the pair (A,B), not of the one
  traversal edge integration happened to pick. Re-fetching all A-B edges and taking the best tier answers
  exactly that; same run, same CURIEs, so the view is consistent. Carrying provenance on the `Bridge` is
  therefore a pure **cost** optimization (deferred), not a correctness fix.
- **Static predicate-class set (the probe's), not `bmt`.** `bmt` is not a backend dependency; the static
  causal/associative/neutral set classified correctly in all three probes.
- **Curated = `knowledge_level ∈ {knowledge_assertion, logical_entailment}` AND `agent_type ∈
  {manual_agent, manual_validation_of_automated_agent}`.** Text-mined (`agent_type=text_mining_agent` /
  `knowledge_level=not_provided`) is its own lower tier — a text-mined `causes` is NOT curated-causal.
- **Ship default-off, flip after a validation eval (not "until resolution lands" — that gate is met).**
  The Tier 1 resolution fix has landed (PR #75), so the original blocker is cleared. The remaining
  prudence is operational: a `bridge_grounding` node is a new in-pipeline step that adds Kestrel load
  (two `one_hop_query` full calls per scored bridge), and Tier 2 is still in flight. So wire it
  `enabled=False`, run L4's validation eval over real discovered bridges to confirm the label
  distribution is sensible (not collapsed to `none`/`text-mined` from resolution residual), then flip
  the default in a one-line follow-up. The flip is config-only — no graph surgery.

## Open Questions

### Resolved During Planning

- *Where does provenance extraction live?* — In the labeler (reads one_hop-full dict edges), not
  `parse_kestrel_response`. Confirmed by the probes.
- *How to fetch a single leg's edges?* — `one_hop_query` full mode with **`start_node_ids` only**, then
  client-side filter the returned edges to those touching the target CURIE (server-side `end_node_ids` is
  UNVERIFIED — see Context). This is what L1 shipped. (multi_hop max_path_length=1 is rejected.)
- *L1 shipped API (verified 2026-06-18 in `bridge_grounding/provenance.py`):* `predicate_class(predicate)`,
  `is_curated(kl, agent)`, `evidence_tier(edge: dict)` → `"curated-{class}"` | `"text-mined"`,
  `leg_tier(curie_x, curie_y)` (async, start-only one_hop-full + client filter) → best tier | `"none"`,
  `bridge_label(leg1_tier, leg2_tier)` → `"no KG edge"` | `"one leg unsupported"` |
  `"both legs curated-causal"` | `"weakest leg {weaker}"`. **`leg_tier` returns only the tier string**
  (not the winning edge's metadata) — bears on L2's LegSummary shape (below).
- *L2 LegSummary shape (decision):* keep it **minimal** — `from_curie`, `to_curie`, `evidence_tier`.
  L3 populates all three from `leg_tier(x, y)` (it already has x/y) with **no L1 change**. Drop the
  richer `predicate`/`knowledge_level`/`agent_type`/`source` per-leg fields the original L2 sketched;
  surfacing the winning edge's metadata would require extending `leg_tier`, deferred as an enrichment
  only if synthesis rendering needs it.
- *Enablement (decision):* default-off, flip after L4's validation eval (see R5 / Key Decisions).
- *Superseded v1/v2 scorer modules (decision):* delete, as a separate pre-flight cleanup PR (PR-A / C1)
  that lands before the feature PR (PR-B / L2+L3) — see Scope Boundaries and Phased Delivery.
- *L3 wiring touch-points (verified 2026-06-18, unchanged by PR #74/#75):* `builder.py` still routes
  `temporal`/`integration → synthesis` (no `bridge_grounding` node yet); `synthesis.py` does not read
  `grounded_bridges`; no `BridgeGroundingConfig` exists; `protocol.total_nodes == 10`. The v1 U5 wiring
  design applies as written.

### Deferred to Implementation

- ~~The exact chain-summary label vocabulary~~ — **settled in L1** (`bridge_label` ships four labels:
  `no KG edge` / `one leg unsupported` / `both legs curated-causal` / `weakest leg {weaker}`). L4's eval
  may surface a refinement, but the vocabulary is fixed for L2/L3.
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

- [x] **L1: Provenance classification + per-leg labeler** — DONE (commit 3885f05; 30 unit tests + verified live)

**Goal:** Productionize the probe's logic: classify an edge's evidence tier, fetch a leg's edges and
take the best tier, and compose a bridge chain-summary label.

**Requirements:** R1, R2, R3

**Dependencies:** None (reuses `call_kestrel_tool`).

**Files:**
- Create: `backend/src/kestrel_backend/bridge_grounding/provenance.py` — `predicate_class`,
  `is_curated`, `evidence_tier(edge)`, `leg_tier(curie_a, curie_b)` (async, one_hop-full fetch),
  `bridge_label(leg1_tier, leg2_tier)` (two tier strings → chain summary).
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

- [x] **C1: Pre-flight cleanup — delete superseded v1/v2 scorer modules (PR-A)**

**Goal:** Remove the dead v1/v2 mechanism-scorer apparatus so the feature PR (and future readers) see one
scoring path, not three. Ships as its own PR before L2/L3.

**Requirements:** (supports R3 hygiene; no functional requirement)

**Dependencies:** None. Lands first (before L2).

**Files:**
- Delete: `backend/src/kestrel_backend/bridge_grounding/{retrieval,prompts,labeling,scoring,panel}.py`.
- Delete: `backend/tests/test_bridge_grounding_{retrieval,prompts,labeling,scoring}.py`,
  `backend/tests/test_bridge_grounding_panel.py`.
- Modify: `backend/src/kestrel_backend/bridge_grounding/__init__.py` — prune exports of the deleted modules.
- Keep: `provenance.py` (L1), `__init__.py`, and the `backend/assessment_data/kg_bridge_*` probes
  (the negative-result apparatus, incl. `kg_bridge_graded_probe.py` that produced AUC 0.61).

**Approach:** Pure removal, no runtime change. **Before deleting**, grep `src/` (outside `bridge_grounding/`)
and the whole test suite for imports of the five modules; resolve any dangling import (expected: none
outside the deleted tests and `__init__`).

**Execution note:** Mechanical deletion — no test-first; the verification is "the surviving suite imports
and passes."

**Test scenarios:**
- `Test expectation: none` (pure deletion). Verification is the regression below.
- Regression: full affected suite imports clean and passes after removal — no dangling import of a deleted
  module from `__init__.py`, another test, or any `src/` module.

**Verification:** The five modules + their tests are gone; `__init__.py` exports are pruned; the backend
test suite (run affected files individually) imports and passes; no production code references the removed
modules.

- [x] **L2: BridgeGrounding model adaptation (provenance label fields)**

**Goal:** Repurpose the U1 model to carry the per-leg tier + chain label instead of co-occurrence tallies.

**Requirements:** R2, R4

**Dependencies:** L1; **C1 lands first** (the scorer modules that construct `BridgeGrounding`/`LegSummary`
are gone, so the field change has no live non-test constructor to break); reuses U1
(`BridgeGrounding`, `LegSummary`, `grounded_bridges`, contracts).

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py` — repurpose `LegSummary` to the **minimal**
  provenance shape: `from_curie`, `to_curie`, `evidence_tier` (drop the v1 co-occurrence tally fields
  `support`/`refute`/`neither`/`off_topic`/`dropped_co_mention`). `BridgeGrounding` carries `legs`
  (`list[LegSummary]`, already present) + a new `label` (chain summary string). **Not fully additive:**
  `BridgeGrounding.support_fraction` and `decision` are currently **required** v1-score fields — remove
  them (the score is gone), along with the secondary `strong_leg_fraction`/`*_total` ranking fields. The
  frozen-model + `model_copy` attach pattern is unchanged.
- Test: `backend/tests/test_bridge_grounding_models.py` (rewrite U1's tests — they construct
  `BridgeGrounding(support_fraction=…, decision=…)` and `LegSummary(support=…)`, which break on the
  field change).

**Approach:** L3 populates each `LegSummary` from L1 — `from_curie=x`, `to_curie=y`, and
`evidence_tier = await leg_tier(x, y)` (**`leg_tier` is `async`** — the L3 node awaits it per leg and
passes the resolved *string* into `LegSummary`; do not store the coroutine). **No change to L1's
`leg_tier`** is needed. Reuse the no-reducer `grounded_bridges` key, the contracts, and the
`entities`-tuple join from U1 unchanged. **Runtime-safe (verified):** `state_contracts.py`
`BridgeGroundingOutput` references only `grounded_bridges`/`bridge_grounding_errors`/`model_usages` and
`BridgeGroundingInput` reads `bridges` — neither touches the removed `support_fraction`/`decision`, so
removing them does **not** break contract validation. No production reader consumes the removed fields,
and with the v1 scorer modules already removed in C1, the only remaining `BridgeGrounding`/`LegSummary`
constructor is `test_bridge_grounding_models.py` — rewritten here.

**Test scenarios:**
- Happy: a provenance `BridgeGrounding` (legs of `{from_curie,to_curie,evidence_tier}` + `label`) round-trips (frozen serialization).
- Edge: `model_copy` attach doesn't mutate the frozen original; join `grounded_bridges`→`bridges` by `entities`.
- Edge: constructing `BridgeGrounding` without the removed `support_fraction`/`decision` succeeds (fields gone, not just optional).

- [x] **L3: `bridge_grounding` node + wiring + synthesis rendering**

**Goal:** A `bridge_grounding` node that labels ordered 3-node bridges and attaches `grounded_bridges`;
wired before synthesis; default-off; synthesis renders the per-bridge label.

**Requirements:** R3, R4, R5

**Dependencies:** L1, L2 (and C1 already merged); reuses the v1 plan U5 wiring design.

**Files:**
- Create: `backend/src/kestrel_backend/graph/nodes/bridge_grounding.py` (`@validate_state`; ordered-3-node
  + exclude-subgraph filter via `predicate_directions != []`; per-bridge: `leg_tier` per leg → build
  `LegSummary`s → `bridge_label` → attach `BridgeGrounding` via `model_copy`; returns `grounded_bridges` +
  `bridge_grounding_errors`; best-effort, never throws).
- Modify: `backend/src/kestrel_backend/graph/builder.py` — synthesis has **two** inbound edges that must
  **both** be redirected through the new node, or one study type bypasses it:
  1. the conditional non-longitudinal path: in `route_after_integration`'s map, change the `"synthesis"`
     target to `"bridge_grounding"`;
  2. the longitudinal path: change `add_edge("temporal", "synthesis")` to `add_edge("temporal",
     "bridge_grounding")`;
  3. add `add_edge("bridge_grounding", "synthesis")`.
  Full graph only; leave post-synthesis `literature_grounding`. *(Verified 2026-06-18: current wiring is
  `integration → temporal|synthesis` via `route_after_integration` + `temporal → synthesis`; no
  bridge_grounding node present. Redirecting only one path would silently skip the node for the other
  study type.)*
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` (add `BridgeGroundingConfig` —
  `enabled=False` default + per-leg/`max_scored_bridges` fetch limit; exclude-subgraph is a fixed
  requirement, not a toggle — mirror `IntegrationConfig`; none exists today).
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py` (read `grounded_bridges`, render the
  per-bridge provenance label; verified it does not read `grounded_bridges` today).
- Modify: `backend/src/kestrel_backend/protocol.py` (add `NODE_STATUS_MESSAGES["bridge_grounding"]`;
  `total_nodes` 10→11; verified `total_nodes == 10`).
- Test: `backend/tests/test_bridge_grounding_node.py` (sys.modules stub harness, per v1 plan U5).

**Approach:** Mirror the v1 plan U5 wiring decisions (substrate-independent); the node calls L1's
`leg_tier` + `bridge_label` per bridge instead of any scorer. Per-bridge isolation: a Kestrel failure on
one bridge degrades it (label `none`/error in `bridge_grounding_errors`), node still returns. Ship
`enabled=False` (flip is L4, config-only). The dead-scorer deletion is **not** here — it shipped in C1.

**Test scenarios:**
- Happy: a 3-node bridge with curated-causal legs → `BridgeGrounding` attached, label "both legs curated-causal".
- Edge: `enabled=False` short-circuits (no Kestrel calls); skips subgraph/2-node/4+-node bridges (`predicate_directions == []`).
- Error: a `one_hop_query` failure on one bridge → that bridge labeled `none` + `bridge_grounding_errors`
  entry, node still returns.
- Integration (both study types): the rewired graph routes through `bridge_grounding` on **both** the
  longitudinal path (via `temporal`) and the non-longitudinal path (via `route_after_integration`) — assert
  the node runs in each, not just one; `total_nodes`/`NODE_STATUS_MESSAGES` updated; synthesis renders the
  label; disabled = no-op.

- [ ] **L4: Validation eval over real bridges + enablement flip (follow-up)**

**Goal:** Confirm the labeler produces a sensible label distribution on real, now-correctly-resolved
bridges, then flip the node default to `enabled=True`.

**Requirements:** R5

**Dependencies:** L3 (the wired node must exist). May run after L3 merges; the flip is a separate small change.

**Files:**
- Create/extend: an eval runner over real discovered bridges (reuse the `kg_bridge_leg_probe` harness in
  `backend/assessment_data/`) — auto-persists results (per `[[persist-expensive-run-artifacts]]`).
- Modify (the flip, follow-up): `backend/src/kestrel_backend/graph/pipeline_config.py`
  (`BridgeGroundingConfig.enabled` default `False → True`).

**Approach:** Run the labeler over a representative set of real discovery-pipeline bridges (post-PR #75
resolution). The honest expected distribution is **already** skewed `text-mined`/`none` (only ~23% of
legs are curated-causal), so "not all-`none`" is **not** a usable gate — Tier-2 residual would void *some*
legs and shift the distribution toward `none` without collapsing it, landing in the indistinguishable
middle. So the go/no-go must be **pre-committed against a reference baseline, not eyeballed:**
- **Reference:** record the curated-causal / curated-* / text-mined / none **leg-fraction** from the
  committed `kg_bridge_leg_probe` baseline (the ~23% curated-causal figure) as the expected healthy shape.
- **Go criterion:** the live curated-causal leg-fraction over the sampled bridges falls within a
  pre-committed tolerance band of that baseline; a material drop below the band signals resolution residual
  voiding legs → **hold the flip / wait for Tier 2** (the exact band % is set when L4 runs, against the
  recorded baseline).
- **Usefulness spot-check (not just frequency):** on a small hand-picked sample, confirm the label changes
  interpretation — a `curated-causal` leg points to a different researcher action than a `text-mined`/`none`
  one (see Overview). A label that only ever restates "the KG is sparse here" earns no place in the report.
This is a **sanity check, not a calibration gate** (no confidence claim to validate) — but the go decision
is decidable against the pre-committed baseline, not a hindsight read.

**Execution note:** This is eval tooling + a one-line config change, not a test-bearing feature unit.

**Test scenarios:**
- `Test expectation: none` — eval tooling (runnable harness) + a config-default change; covered by L3's
  node tests for behavior. The eval's output is a persisted artifact, not an assertion.

**Verification:** A persisted label-distribution artifact over real bridges, and (on a go decision) the
`enabled` default flipped with the node observably running in a discovery run.

## System-Wide Impact

- **Interaction graph:** `bridge_grounding` inserts before synthesis (reuse v1 wiring); new Kestrel load
  (2 `one_hop_query` full calls per scored bridge) — bounded by `max_scored_bridges`; node default-off.
- **Error propagation:** per-bridge isolation → label `none` + `bridge_grounding_errors`; node never throws.
- **State lifecycle:** `grounded_bridges` no-reducer key, single writer before synthesis reads it (U1).
- **Unchanged invariants:** `Bridge`, the `grounded_bridges`/`bridge_grounding_errors` state keys, and the
  contracts registry stay intact; `parse_kestrel_response` untouched (labeler reads one_hop dicts). The
  `BridgeGrounding`/`LegSummary` **field set changes** (remove v1 score/tally fields, add `label`/minimal
  legs) — safe because the only constructors are the v1 scorer modules (deleted first in C1/PR-A) and the
  rewritten model tests; no production reader consumes the removed fields, and `state_contracts` validators
  don't reference them (verified).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Label accuracy depends on correct CURIEs | Tier 1 fix landed (PR #75) — the dominant wrong-namespace cases (disease→pathway → zero edges) are fixed. Residual Tier 2 mis-ranking (non-canonical same-category node) is **partially silent**: a mis-resolved leg looks identical to a genuinely sparse one. Ship `enabled=False`; L4's **baseline-band** eval (live curated-causal leg-fraction vs the committed `kg_bridge_leg_probe` baseline) is the aggregate detector that gates the flip — not "not all-`none`", which the expected sparse distribution would always pass |
| Most real bridges label `text-mined`/`none` (only ~23% legs curated-causal) | This is the honest, intended behavior — the label reports what evidence exists; it does not claim mechanism. Not a bug |
| Per-leg `one_hop_query` full fetches add Kestrel load (2 full-mode calls × N scored bridges per run) | Bounded by a `max_scored_bridges` cap + default-off; **set a concrete cap and a per-call edge `limit` in `BridgeGroundingConfig`** (mirror `IntegrationConfig`'s bounds) so the load ceiling is explicit, not implied; optimize to carry provenance on the `Bridge` (deferred) only if the cap proves too tight |
| Deleting the v1/v2 scorer modules breaks an unnoticed import | C1 (PR-A) greps `src/` (outside `bridge_grounding/`) and the test suite for imports of the five modules before removal, and verifies the surviving suite imports clean; isolating it as its own PR gives the deletion an independent revert handle |

## Phased Delivery

The original accuracy blocker (the entity-resolution wrong-namespace fix) **has landed** — PR #75
(Tier 1: category-constrained `resolve_via_api` + disease intake hint), merged to `dev`. Phase 1 (L1)
shipped in PR #74. So this pass delivers the in-pipeline node, gated by a validation eval rather than by
the resolution fix.

### Phase 1 (DONE) — L1: the deterministic labeler
Shipped in PR #74 (commit 3885f05): the pure `evidence_tier`/`predicate_class`/`bridge_label` classifiers
+ the per-leg `leg_tier` fetch, 30 unit tests, eval-only.

### Phase 2 (now) — two PRs off fresh branches from `dev`
- **PR-A (C1): pre-flight cleanup** — delete the superseded v1/v2 scorer modules + tests. Pure removal, no
  runtime change, trivial to review/revert; lands first.
- **PR-B (L2 + L3): the feature** — minimal model adaptation (L2) + the `bridge_grounding` node, graph
  rewiring (both inbound synthesis edges), synthesis rendering, and `enabled=False` (L3). Built on top of
  PR-A. (The old `feat/bridge-grounding-scorer` branch merged in PR #74; both PRs start fresh from `dev`.)

### Phase 3 (follow-up) — L4: validation eval + enablement flip
Run the labeler over real, now-correctly-resolved bridges; confirm the label distribution is sensible
(skewed `text-mined`/`none` is expected and honest; an all-`none` collapse is the signal to hold the flip
/ wait on Tier 2). On a go decision, flip `BridgeGroundingConfig.enabled` default to `True` in a one-line
follow-up. Tier 2 (biomapper2 within-category canonical ranking) is in flight in parallel and tightens
label fidelity further, but is not required for the flip.

## Documentation / Operational Notes

- Node ships `enabled=False`; the flip to `True` (L4) is a one-line `pipeline_config` change gated on the
  validation eval, not on further graph work.
- L4's eval over real discovered bridges (reuse the `kg_bridge_leg_probe` harness) sanity-checks the label
  distribution — a *sanity check, not a calibration gate* (no confidence claim to validate).

## Sources & References

- **Origin document:** `backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md`
- Probes (the logic productionized in L1): `backend/assessment_data/kg_bridge_leg_probe.py`,
  `kg_provenance_probe.py`, `kg_bridge_graded_probe.py`
- v1 plan (reused U1 model + U5 wiring design): `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md`
- Accuracy dependency (LANDED): `docs/wiki/entity-resolution-namespace-fix.outline.md`; Tier 1 = PR #75
  (`docs/plans/2026-06-17-003-feat-entity-resolution-category-filter-plan.md`), merged to `dev`.
- Tier 2 (parallel, biomapper2): `../biomapper2/AGENT-TASK-canonical-namespace-preference.md`.
- L1 shipped in PR #74 (branch `feat/bridge-grounding-scorer`, since merged to `dev`). **L2/L3/L4 start
  from a fresh branch off `dev`** — the old scorer branch is merged and gone.
