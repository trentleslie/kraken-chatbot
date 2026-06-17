---
title: "feat: Bridge-Grounding v2 — KG-Provenance Gate + LLM Confirmation"
type: feat
status: superseded
date: 2026-06-17
origin: docs/brainstorms/2026-06-17-bridge-grounding-v2-kg-provenance-requirements.md
deepened: 2026-06-17
superseded_by: backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md
---

> **⛔ SUPERSEDED (2026-06-17).** Document-review + three Kestrel probes showed this scorer's signal is
> **weak and sparse**: only 23% of real bridge legs are curated-causal, the graded chain score
> separates real-vs-spurious at only **AUC 0.61**, and combining with literature backing did not help
> (popularity trap). The initial "6/6 validation" over-claimed — it tested direct curated drug→disease
> edges, not the legs of discovered bridges. **Reframed from a mechanism-confidence SCORE to a
> deterministic evidence-provenance LABEL** (no LLM, no calibration gate). See
> `backend/assessment_data/BRIDGE_GROUNDING_V2_FINDINGS.md`. The KG-provenance *extraction* (U0/U1 reuse,
> leg edge knowledge_level/agent_type/predicate) carries forward to the reframe; the scoring/gate/Phase-2
> units below do not. Retained for the provenance-extraction design and the negative-result record.

# feat: Bridge-Grounding v2 — KG-Provenance Gate + LLM Confirmation

## Overview

Rebuild the per-`Bridge` grounding signal on a new substrate. v1 (co-occurrence abstracts + LLM
labeling) **failed Tier A calibration** (margin −0.333; the hard negative coffee→caffeine→pancreatic
scored highest) because co-occurrence tracks "studied association", not mechanism. A live Kestrel
probe (2026-06-17) found the bridge's **own leg edges already encode** mechanism-quality signal: a
typed Biolink predicate, **and — more decisively — a `knowledge_level` and `agent_type`** that
distinguish a curated mechanism assertion from text-mined co-occurrence, plus `primary_knowledge_source`,
`qualifiers` (incl. `negated`), and (on ~18% of edges) `publications`.

v2 grounds each bridge from that edge provenance, with the gate ordered by **curation first**
(`knowledge_level` + `agent_type`), predicate causal/associative class as a *modulator within a
curation tier*, and the LLM used only to confirm strength / detect contestation on the edge's own
publications (Phase 2, conditional). The work reuses v1's substrate-independent units (U0 predicate/
orientation extraction, U1 model+state, the qualitative band, default-off, before-synthesis topology)
and the U8 calibration harness, and retires the co-occurrence retrieval (U2) + from-scratch labeling
(U3) as the primary path (kept eval-only as the baseline v2 is measured against).

## Problem Frame

See origin requirements and the failure analysis (`backend/assessment_data/BRIDGE_GROUNDING_TIER_A_FINDINGS.md`).
The deepest risk the whole exercise tested — "does the score track mechanism at all?" — was answered
NO for v1. **The deepening pass (2026-06-17) sharpened the v2 risk: the discriminator is not predicate
type but *curation*.** In a Translator aggregate (~⅓ of `causes` edges are text-mined; SemRep
`causes`-extraction precision ≈ 0.55, e.g. the spurious "TP53 CAUSES neuroblastoma"), a naive
"causal predicate = mechanism" rule rewards co-occurrence wearing a predicate costume — the **same v1
failure in a new substrate**. v2 must gate on curation (`knowledge_level`/`agent_type`) to avoid it.

## Probe Validation (2026-06-17 — the bet holds on real edges)

A near-zero-cost Kestrel-only probe (`backend/assessment_data/kg_provenance_probe.py` + a CURIE-based
variant) tested the **joint gate** — a leg has a curated (`knowledge_assertion`/`manual_agent`) AND
causal-predicate edge — on labeled pairs with hand-verified MONDO/CHEBI/NCBITaxon CURIEs:

- **Positives 6/6** had a curated-causal edge (H.pylori→ulcer `causes`/DrugMechDB; metformin→T2D 5×
  curated-causal; levodopa→Parkinson; imatinib→CML; donepezil→Alzheimer; warfarin→thrombosis).
- **Negatives 1/6** — only aspirin→ovarian cancer (a *defensible* curated chemoprevention signal);
  caffeine→pancreatic, beta-carotene→lung, vitamin E→CVD all had only text-mined-causal or
  curated-**associative** edges → correctly rejected.

This is the **opposite of v1** (which ranked the hard negative above the positives, margin −0.333).
Two confirmations: (1) the **joint** gate is load-bearing — caffeine→pancreatic has a *curated*
`has_adverse_event` (associative → curation-alone fails) AND a *text-mined* `contributes_to`
(causal → predicate-alone fails); only curated-AND-causal passes. (2) **Resolution is the binding
dependency** — a wider name-resolved probe failed on wrong-namespace CURIEs (disease names →
KEGG/PANTHER *pathways*; `VO:` is not a valid Kestrel prefix; wrong-species genes). v2's accuracy
rides on correct MONDO/HGNC/CHEBI CURIEs, hence on the biomapper resolution fix
(`[[biomapper2-wraps-kestrel-hgnc-marker]]`).

## Requirements Trace

- **R1** — Per-leg grounding from the leg edge's `knowledge_level`, `agent_type`, `predicate` (+ class),
  `primary_knowledge_source`, `negated`/qualifiers, and `publications` (KG-sourced, not co-occurrence). → V1, V2
- **R2** — Gate **curation-first** (`knowledge_level`+`agent_type`), predicate causal/associative class
  modulates *within* a tier; spurious-but-studied chains (text-mined / associative / statistical) must
  score low. Negated edges count as refuting, never support. → V2
- **R3** — Compose two legs into a chain verdict via weaker-leg gating (factored into a shared,
  substrate-neutral helper); keep the stronger leg only if a band tie-break needs it. → V2
- **R4** — LLM confirmation runs over the edge's own publications, scoped to confirm strength / detect
  contestation — not re-derive the relation; conditional on Phase-1 calibration. → V6 (Phase 2)
- **R5** — **The central experiment:** read the panel chains' provenance from *live* Kestrel edges
  (discard hand-authored fixtures); reuse the U8 evaluator + pre-registered thresholds; v2 must PASS
  where v1 failed (coffee→caffeine→pancreatic LOW, positives clear floor, margin ≥ +0.30). → V5
- **R6** — Keep v1's substrate-independent plumbing (U1 model/state, band, default-off,
  before-synthesis); extend U0 (it currently drops the provenance columns — new work); retire U2/U3 as
  primary; keep v1 scorer eval-only as baseline. → V3, V4, V7

## Scope Boundaries

- Single-bridge grounding only — no pairwise competing-bridge ranking.
- Ordered 3-node multi-hop bridges only (exclude subgraph "connecting" bridges — v1 O5 carries over).

### Deferred to Separate Tasks

- Beta-Binomial CI / resampling — deferred (v1 U9 seam carries over).
- Tier B time-sliced calibration — deferred behind the temporal-eval port.
- LLM confirmation layer (V6) is **Phase 2**, conditional — see Phased Delivery and the V5 pre-registration.

## Context & Research

### Relevant Code and Patterns (verified firsthand + by the deepening pass, 2026-06-17)

- `backend/src/kestrel_backend/kestrel_client.py` — `parse_kestrel_response`. U0 extended it to emit
  per-hop `{predicate, forward}` from `edges`/`edge_schema`. **CORRECTION (deepening):** the parse
  currently reads **only** `subject`/`predicate`/`object` (`_edge_triple` + the `edge_schema` position
  loop); it **discards** `primary_knowledge_source` (idx 4), `knowledge_level` (idx 7), `agent_type`
  (idx 8), and `id` (idx 9). V1 is genuinely *new extraction work*, not "make available." Also
  `_hop_predicates` collapses a multi-edge hop to the **alphabetically-first** predicate
  (`_triple_map` sorts by predicate, `_hop_predicates` takes `[0]`) — V1 must stop discarding the
  other candidate edges.
- `backend/src/kestrel_backend/graph/nodes/integration.py` — multi-hop bridge builder consumes only
  `path["predicates"]`; it does **not** see raw edge tuples, so V1 must enrich the **parser** output
  (then the builder), not the builder alone.
- `backend/src/kestrel_backend/graph/state.py` — `Bridge` (frozen; `predicates`,
  `predicate_directions`), `BridgeGrounding`/`LegSummary` (frozen), `grounded_bridges` (no-reducer),
  `bridge_grounding_errors`. Adding fields is additive/backward-compatible (verified).
- `backend/src/kestrel_backend/graph/state_contracts.py` — `BridgeGroundingInput/Output` + registry
  entry already present (verified).
- `backend/src/kestrel_backend/bridge_grounding/scoring.py` — U4 `score_chain`/`ChainScore`. **CORRECTION:**
  not directly reusable — its input is `list[tuple[int,int,int]]` co-occurrence counts and it computes
  a ratio internally; `weak_leg_index` is a dataclass **field**, not a helper. Factor the weaker-leg /
  floor / secondary-key *composition shape* into a shared helper; use that.
- `backend/src/kestrel_backend/bridge_grounding/{retrieval,prompts,labeling}.py` — v1 co-occurrence
  path → eval-only baseline.
- `backend/src/kestrel_backend/bridge_grounding/panel.py` + `assessment_data/bridge_grounding_tier_a.py`
  — U8 harness. **CORRECTION:** `build_bridge` constructs SYNTHETIC bridges with hand-set predicates;
  no resolution, no real edges. V5 must replace this with live resolution + edge fetch (see V5).
- `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` — `resolve_via_api(entity)` wraps
  `call_kestrel_tool("hybrid_search", {"search_text": …, "limit": 1})` → real CURIE. Reuse in V5.
- `get_edges` — call directly via `call_kestrel_tool("get_edges", {"edge_ids": <id|list>, "slim": False})`
  (the `kestrel_tools.py` `@tool` wrapper declares only `{"edge_ids": str}`, but `call_kestrel_tool`
  passes args through, mirroring `resolve_via_api`). `slim:false` surfaces `publications` (on ~18% of
  edges) + nested `attributes`; whether it surfaces `negated` is UNVERIFIED (see Negation guard).
- `backend/src/kestrel_backend/protocol.py` — `total_nodes: int = 10` (V4: 10→11); `NODE_STATUS_MESSAGES`
  has no `bridge_grounding` entry (V4 must add one).
- `backend/src/kestrel_backend/graph/builder.py` — reroute `integration`/`temporal → synthesis` through
  `bridge_grounding` (full graph only; do NOT touch the separate post-synthesis `literature_grounding`).
- Kestrel exposes `knowledge_level`/`agent_type`/`primary_knowledge_source`/`degree_percentile` as
  **query-time constrainable** fields (17-field constraint API) + a 6-preset ranking ladder — so
  text-mining exclusion and hub-avoidance can happen at *retrieval*, not only post-hoc.

### Institutional Learnings

- `[[bridge-grounding-v1-fails-calibration]]` — the negative result driving this pivot.
- `[[persist-expensive-run-artifacts]]` — calibration runs auto-persist (U8 harness already does).
- `[[biomapper2-wraps-kestrel-hgnc-marker]]` — entity resolution can return wrong-namespace/species
  CURIEs; V5's name→CURIE resolution must be sanity-checked or a chain silently mis-resolves.
- The Claude Agent SDK can pin neither `temperature` nor `model` — Phase 2 reproducibility via frozen
  snapshots only. (Phase 1 is fully deterministic — a calibration-trust win over v1.)

### External References (Biolink / Translator — deepening research)

- **Predicate `is_a` tree encodes causal vs associational** (no boolean flag; classify by walking
  ancestry, ideally via `bmt`): CAUSAL = `causes`/`contributes_to`/`affects`/`regulates`/
  `directly_physically_interacts_with`/`treats`(KL-caveated); ASSOCIATIVE = `associated_with`/
  `correlated_with`/`coexpressed_with`/`genetically_associated_with`/`gene_associated_with_condition`;
  NEUTRAL = `related_to`/`interacts_with`/`physically_interacts_with`.
- **`knowledge_level` — Kestrel exposes 5 values** (`knowledge_assertion | prediction |
  logical_entailment | statistical_association | not_provided`, per `docs/kestrel-api-reference.md`).
  `statistical_association` is the epidemiology flag; `not_provided` is "most often text-mined."
  **NOTE:** the Biolink spec has a 7th value `text_co_occurrence`, but it does **not** appear in
  Kestrel — the text-mined bucket maps to `agent_type=text_mining_agent` AND/OR
  `knowledge_level in {not_provided, statistical_association}`, NOT to a `text_co_occurrence` KL.
  **`knowledge_level` + `agent_type` together are a cleaner mechanism-quality signal than predicate type.**
- **`AgentTypeEnum`**: `manual_agent` ≈ curated; `text_mining_agent` ≈ down-weight; `computational_model`
  ≈ prediction/inflation. No official `infores:` reliability table exists — prefer `agent_type`.
- **`negated` boolean** must never be ignored (a negated `causes` = "does NOT cause"). `object_direction_qualifier`,
  `causal_mechanism_qualifier` carry direction/mechanism signal.
- Sources: Biolink Model v4.4.2 schema; Unni et al. 2022; ARAX ranking (PMC10027432, the SemMedDB
  publication-count confidence curve = popularity trap); RTX-KG2 (~⅓ SemMedDB, BMC Bioinformatics 2022);
  SemRep precision PMC7222583; NLP-KG pitfalls arXiv:2310.15572; degree-bias mitigation PMC4269436.

## Key Technical Decisions

- **Gate on curation FIRST, predicate class as a modulator (priority corrected by deepening).** Signal
  ordering strongest→weakest: `knowledge_level` → `agent_type` → (optional) curated-source allowlist →
  predicate causal/associative class. A text-mined `causes` (`agent_type=text_mining_agent` /
  `knowledge_level in {not_provided, statistical_association}`) must score **below**
  a curated `associated_with` (`knowledge_assertion`/`manual_agent`). This is the variable that
  separates v2 from a relabeled v1 — predicate-class alone reproduces the v1 failure on real edges.
- **Negation/direction guard (⚠️ data source UNVERIFIED — probe before building).** Biolink edges can
  carry `negated == true` ("X does NOT cause Y"), which must count as refuting, never support. **But
  `negated` is NOT a documented Kestrel field** — it is not in the slim edge tuple, and the API
  reference's list of what `get_edges slim:false` adds (`attributes` + `publications`) does **not**
  include it, nor is it among the 17 query-constrainable fields. So whether Kestrel exposes `negated`
  at all (and if so, whether it's buried in `qualifiers` (idx 3, already in slim) or the nested
  `attributes` blob) is unknown. **Resolve with a single live `get_edges slim:false` probe on a
  known-negated edge BEFORE building the guard.** If `negated` is absent, the guard is a deferred
  concern (note the gap; don't silently no-op it). `object_direction_qualifier`, if present in
  `qualifiers`, feeds direction logic.
- **Publication/edge counts are a FLOOR, not a mechanism score.** ~18% of edges have publications;
  curated assertions often lack PMIDs. Pub-count is a tie-breaker / confidence annotation, **never a
  multiplicative gate** (else curated no-PMID edges get penalized for the KG's sparse coverage).
  Reconcile with the existing >1000-edge hub detection — popularity ≠ mechanism.
- **Multi-edge legs aggregate source-gated, not best-causal.** A leg has many edges; capture ALL
  candidates (changing U0's lossy single-predicate collapse). The discriminator is "is there a causal
  predicate from a *curated/asserted* edge", with text-mined causal edges treated as associative-tier.
  "best-causal regardless of source" is a **registered failure mode** to check in V5, not a default.
- **Use Kestrel query-time constraints where possible.** Exclude `text_mining_agent` / high-degree hubs
  at retrieval (`agent_type ne …`, `degree_percentile lt …`) rather than only post-hoc — cheaper and
  cleaner. (Pin the exact constraints in V5 from the observed distribution.)
- **Extend `Bridge` with per-hop provenance (one pass), fetch publications lazily.** Phase-1 provenance
  is in the multi_hop tuple already; re-querying would duplicate. `get_edges` only for Phase-2
  publications/`negated`.
- **Phase the LLM out of the critical path, but DON'T assume Phase 1 suffices.** Phase 1 is the cheap
  deterministic gate; V5 must produce the real-edge KL×agent_type×predicate×source distribution per leg
  and **pre-register the specific separation that licenses skipping Phase 2.** Framing Phase 2 as
  "only if Phase 1 fails" risks motivated reasoning; treat it as data-gated.
- **Keep v1 as the baseline comparator, not deleted** — V7 scores both on the panel.

## Open Questions

### Resolved During Planning (by the deepening pass)

- *Do Kestrel predicates distinguish mechanism from association?* — The predicate `is_a` tree does, but
  predicate type **alone is insufficient** (text-mining over-asserts `causes`). Curation
  (`knowledge_level`/`agent_type`) is the load-bearing discriminator.
- *Is SemMedDB needed?* — No; but note Kestrel's aggregate *contains* SemMedDB/text-mined edges, which
  is exactly why curation-gating (not predicate type) is required.
- *Where does provenance come from?* — multi_hop compact edges carry predicate + source + KL +
  agent_type (extend U0 to read them, all currently dropped); `publications` + `negated` via `get_edges`.

### Deferred to Implementation

- The exact KL/agent_type/predicate-class → weight table — **pre-register in V5** from the observed
  real-edge distribution (data-driven, auditable).
- Multi-edge aggregation policy (source-gated) — choose from the V5 per-leg distribution.
- Whether Phase-1 (KG-only) passes Tier A — determines if V6 (LLM) is built at all; pre-registered in V5.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
bridge (ordered A→B→C; legs are KG edges)
  for each leg, over ALL candidate edges for the (curieA, curieB) pair:
     drop edges with negated == true (or mark refuting)
     curation_tier = tier(knowledge_level, agent_type)   # assertion/curated ≫ prediction ≫ text/statistical
     pred_class    = classify(predicate)                  # causal | associative | neutral (Biolink is_a)
     edge_weight   = f(curation_tier, pred_class, qualifiers)   # curation dominates; class modulates
  leg_score = source-gated aggregate of candidate edge_weights   # NOT "best-causal anywhere"
  pub_count = floor/annotation only (NOT a multiplier)
  chain_score = weaker_leg(leg_scores)                    # shared composition helper (factored from U4)
  decision/band from chain_score                          # reuse U1 model + qualitative band
  # Phase 2 only, where the gate is ambiguous AND the edge has publications:
  if ambiguous(leg) and leg.publications: LLM over get_edges(edge_id).publications → confirm | contest
→ synthesis renders the band (reuse v1 before-synthesis wiring)
```

The coffee→caffeine→pancreatic chain: even if a `causes`/`affects` edge exists, it should be
`text_mining_agent` / `not_provided` KL → gated to associative-tier → low score. (V5 verifies this on
real edges; it is the central pre-registered claim.)

## Implementation Units

### Phase 1 — Deterministic KG-Provenance Gate (build + calibrate before any LLM)

- [ ] **V1: Per-leg edge provenance extraction (new parser work)**

**Goal:** Capture each leg's full provenance — `knowledge_level`, `agent_type`, `primary_knowledge_source`,
`predicate`, `negated`/direction qualifier, `edge_id` — across ALL candidate edges for the hop.

**Requirements:** R1, R2

**Dependencies:** Extends U0's edge parsing (which currently drops these columns and collapses multi-edge hops).

**Files:**
- Modify: `backend/src/kestrel_backend/kestrel_client.py` — extend the `edge_schema` position loop +
  `_edge_triple`/`_triple_map`/`_hop_predicates` to read `primary_knowledge_source`/`knowledge_level`/
  `agent_type`/`id` (positions from schema) and to **return all candidate edges per hop**, not just the
  alphabetically-first predicate.
- Modify: `backend/src/kestrel_backend/graph/state.py` — `Bridge`: additive per-hop provenance fields
  (parallel to `predicates`); a small `LegEdgeProvenance` value type.
- Modify: `backend/src/kestrel_backend/graph/nodes/integration.py` — populate the new fields from the
  enriched parse output.
- Create: `backend/src/kestrel_backend/bridge_grounding/provenance.py` — `predicate_class(predicate)`
  (Biolink is_a; prefer `bmt` ancestry over a hardcoded set), `curation_tier(knowledge_level, agent_type)`,
  and a lazy `fetch_edge_publications_and_negation(edge_id)` via `get_edges {"edge_ids":…, "slim":False}`.
- Test: `backend/tests/test_bridge_grounding_provenance.py`

**Approach:** Read positions from `edge_schema` (never hardcode), mirroring U0. Keep existing
`predicates`/`forward` output **and** add the new fields/candidates additively. `negated` is not in the
slim tuple — surface it via the lazy full-mode fetch (and use it in V2's negation guard); for Phase 1,
if `negated` requires the extra call, decide in V5 whether to constrain it out at query time instead.

**Execution note:** Characterization test first — pin the current `parse_kestrel_response` output
(incl. U0's `predicates`) before adding fields; then assert the additions don't change existing keys.

**Patterns to follow:** U0's edge_schema-driven extraction; `tests/test_kestrel_parse.py` additive pattern.

**Test scenarios:**
- Happy: a 2-hop path whose edges carry KL/agent_type/source → per-hop provenance populated, hop-aligned.
- Edge: a hop with multiple edges of different predicates/sources → ALL candidates captured (not collapsed).
- Edge: missing KL/agent_type/source on an edge → field None at that hop, no positional shift.
- Edge: `predicate_class` classifies causes/contributes_to/affects→causal, associated_with/correlated_with→associative, related_to/interacts_with→neutral.
- Edge: `curation_tier` ranks knowledge_assertion/manual_agent above text_co_occurrence/text_mining_agent/not_provided.
- Characterization: existing `curies/names/predicates/forward` output unchanged (additive only).
- Integration: a real `multi_hop_query` (mode=full) response yields populated multi-candidate leg provenance.

- [ ] **V2: Curation-first mechanism gate + composition helper**

**Goal:** Map each leg's candidate-edge provenance to a mechanism score (curation-first, class-modulated,
negation-guarded, pub-count-as-floor), then compose into a chain score that ranks curated mechanisms
above text-mined/associative/spurious chains.

**Requirements:** R1, R2, R3

**Dependencies:** V1.

**Files:**
- Create: `backend/src/kestrel_backend/bridge_grounding/provenance_scoring.py` — `leg_mechanism_score`
  (curation_tier dominant; predicate_class modulates within tier; drop/penalize `negated`; source-gated
  multi-edge aggregate), `score_chain_from_provenance`.
- Modify: `backend/src/kestrel_backend/bridge_grounding/scoring.py` — **factor the weaker-leg / floor /
  secondary-key composition into a shared, substrate-neutral helper** (do NOT reuse `score_chain`'s
  count-tuple signature). Reuse the helper in both v1 and v2.
- Test: `backend/tests/test_bridge_grounding_provenance_scoring.py`

**Approach:** Weight table (pre-registered in V5): curation tiers (assertion > prediction >
statistical/text/not_provided) × predicate class (causal > associative > neutral). `negated` → refuting.
Source-gated aggregation: a leg's mechanism score reflects the best **curated** causal edge, with
text-mined causal edges demoted to associative-tier. Pub-count enters only as a floor/insufficiency
signal (no edge / no usable provenance → `insufficient_literature`). Never raises.

**Test scenarios:**
- Happy: curated `causes` (knowledge_assertion/manual_agent) legs → high score, `grounded`.
- Edge (THE v1 failure): a text-mined `causes` leg (text_mining_agent/not_provided) → scored as
  associative-tier (LOW), NOT high — the coffee→caffeine→pancreatic shape.
- Edge: curated `associated_with` outranks text-mined `causes` (curation dominates predicate class).
- Edge: `negated==true` causal edge → refuting, not support.
- Edge: a leg with only `related_to`/neutral edges → neutral/low.
- Edge: multi-edge leg with one curated-causal + many text-mined → curated-causal sets the score (source-gated), but many text-mined causal + no curated → associative-tier.
- Boundary: no usable edge → `insufficient_literature`; never raises.
- Anti-popularity: two legs identical except pub-count → pub-count does NOT raise the mechanism score.

- [ ] **V3: BridgeGrounding model adaptation (reuse/extend U1)**

**Goal:** Carry the provenance fields (per-leg curation_tier, predicate_class, knowledge_level,
agent_type, source, negated, pub_count; chain score; contestation flag) on the existing model.

**Requirements:** R3, R6

**Dependencies:** V2; reuses U1 (`BridgeGrounding`, `LegSummary`, `grounded_bridges`, contracts).

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py` — extend `LegSummary`/`BridgeGrounding` with the
  provenance fields (additive; keep v1 co-occurrence fields optional for the baseline; `ci_*` stay null).
- Test: `backend/tests/test_bridge_grounding_models.py` (extend U1's tests).

**Approach:** Reuse the frozen-model + `model_copy` attach + no-reducer `grounded_bridges` pattern
unchanged. The strong-leg secondary key from v1 was for *ranking*; v2 is single-bridge — keep it only
if a band tie-break uses it, else mark it dormant (avoid carrying v1's ranking YAGNI).

**Test scenarios:**
- Happy: a provenance `BridgeGrounding` round-trips (frozen, additive-safe serialization).
- Edge: `model_copy` attach doesn't mutate the frozen original (reuse U1 assertions).
- Edge: join `grounded_bridges` to `bridges` by `entities` tuple (reuse U1).

- [ ] **V4: `bridge_grounding` node + wiring (the node U5 deferred in v1, now v2)**

**Goal:** A `bridge_grounding` node that scores ordered 3-node bridges from KG provenance and attaches
`grounded_bridges`; wired before synthesis; default-off; synthesis renders the band.

**Requirements:** R3, R6

**Dependencies:** V1–V3; reuses the v1 plan's U5 wiring design.

**Files:**
- Create: `backend/src/kestrel_backend/graph/nodes/bridge_grounding.py` (`@validate_state`; ordered-3-node
  + exclude-subgraph filter per v1 O5; per-leg provenance → V2 score → attach; returns `grounded_bridges`
  + `bridge_grounding_errors`).
- Modify: `backend/src/kestrel_backend/graph/builder.py` (reroute `temporal`/`integration → bridge_grounding
  → synthesis`, full graph only; leave the post-synthesis `literature_grounding` untouched).
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py` (`BridgeGroundingConfig`, `enabled=False`
  default, band thresholds, the KL/agent_type/predicate-class weight table, optional query-time
  constraints — mirror `IntegrationConfig`).
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py` (read `grounded_bridges`, render the band).
- Modify: `backend/src/kestrel_backend/protocol.py` (add `NODE_STATUS_MESSAGES["bridge_grounding"]`;
  bump `total_nodes` 10→11).
- Test: `backend/tests/test_bridge_grounding_node.py` (sys.modules stub harness, per v1 plan U5).

**Approach:** Mirror the v1 plan U5 wiring decisions (substrate-independent); the only change is the node
calls the V2 provenance scorer. Best-effort per bridge (never throws; partial `grounded_bridges` on error).

**Test scenarios:**
- Happy: a 3-node bridge with curated-causal legs → grounding attached, decision grounded.
- Edge: `enabled=False` short-circuits; skips subgraph/2-node/4+-node bridges.
- Error: a Kestrel/provenance failure on one bridge → that bridge degraded, node still returns.
- Integration: rewired graph routes through the node; `NODE_STATUS_MESSAGES`/`total_nodes` updated;
  synthesis renders the band; disabled = no-op.

- [ ] **V5: Calibration on REAL edges — the central experiment (GATE)**

**Goal:** Resolve the panel chains to live Kestrel CURIEs, fetch their real leg-edge provenance
(discarding the hand-authored fixtures), score with V2, and apply the pre-registered Tier A gate. v2
must PASS where v1 failed — and this is the first test that the bet holds on *real* data, not fixtures.

**Requirements:** R5

**Dependencies:** V1, V2 (+ the V4 node path optional); reuses `panel.py` evaluator + thresholds.

**Files:**
- Modify: `backend/src/kestrel_backend/bridge_grounding/panel.py` — for each chain, **keep the
  hand-picked CURIEs** (validate them against Kestrel; optionally cross-check via
  `entity_resolution.resolve_via_api`, but do NOT depend on re-resolution — a single mis-resolution
  of one of 18 names would make a leg `insufficient` and `evaluate_tier_a` returns `not_evaluable` for
  the *whole* panel). **Discard ONLY the hand-authored `predicates`/`predicate_directions`** — those
  are the circular part (a human pre-labeled the mechanism). Fetch each leg's real edges via
  `multi_hop_query(start, end, max_hops=1, mode=full)` between the (validated) CURIEs → V1 provenance.
  The CURIEs are entity identifiers, not the thing under test; the predicates/KL/agent_type read from
  live edges are.
- Modify: `backend/assessment_data/bridge_grounding_tier_a.py` — score via V2; **emit the full per-leg
  predicate×knowledge_level×agent_type×source distribution** (not just the headline) so the aggregation
  policy and the skip-Phase-2 decision are made from data.
- Test: `backend/tests/test_bridge_grounding_panel.py` (extend the pure `evaluate_tier_a` tests for the
  v2 result shape; the live run stays a manual spike).

**Approach:** This is the load-bearing experiment the deepening pass identified — the v1 panel's
hand-authored predicates made the prior design's calibration circular. **Pre-register before running:**
(a) the KL/agent_type/predicate weight table; (b) the source-gated aggregation policy; (c) the
separation that licenses skipping Phase 2 (e.g. margin ≥ +0.30 on real edges with the hard negative's
only causal edges being text-mined). Per-chain assumptions V5 must report: do the negatives'/hard-negative's
real edges carry curated-causal predicates (must NOT), and do the positives' binding legs carry a
curated causal edge (must, or weaker-leg gating sinks them)? Persist artifacts by default (SOP).

**Execution note:** STOP and surface the verdict + the distribution before Phase 2. Pre-register, then run.

**Test scenarios:**
- Happy (pure): `evaluate_tier_a` over synthetic v2 ChainResults → pass when margin ≥ 0.30 (reuse).
- Edge (pure): resolution returns no/ wrong-namespace CURIE for a chain → that chain `insufficient`
  (gate handles it), surfaced distinctly so a resolution gap isn't read as a scorer failure.
- The live run is the calibration spike (manual); its verdict + distribution gate Phase 2.

### Phase 2 — LLM Confirmation Layer (ONLY if the V5 pre-registration is not met)

- [ ] **V6: LLM confirmation / contestation over edge publications**

**Goal:** Where the KG gate is ambiguous **and** the leg edge has publications, read those abstracts and
confirm strength / flag contestation — refining (not replacing) the provenance score.

**Requirements:** R4

**Dependencies:** V1 (`fetch_edge_publications…`), V2; reuses `pubmed_client.fetch_abstracts` + the SDK
labeling adapter (`bridge_grounding/labeling.py`, with the no-model/no-temp findings).

**Files:**
- Create: `backend/src/kestrel_backend/bridge_grounding/confirmation.py`
- Modify: `backend/src/kestrel_backend/bridge_grounding/prompts.py` (a "confirm/contest a KG-asserted
  directed relation" prompt — narrower than v1's from-scratch labeling).
- Modify: `backend/src/kestrel_backend/graph/nodes/bridge_grounding.py` (invoke where ambiguous;
  record contestation on `BridgeGrounding`).
- Test: `backend/tests/test_bridge_grounding_confirmation.py`

**Approach:** Scope tightly — the LLM sees the relation type + the edge's own PMIDs' abstracts and
answers "do these confirm or contest the asserted relation?", not "is there a mechanism here?". **Caveat
(deepening F2):** ambiguous legs (associative/thin provenance) are the *least* likely to carry the ~18%
publications — V5 must first measure pub coverage on the ambiguous legs; if near-zero, the Phase-2
rescue must be source-tier refinement (or query-time constraints), not LLM-over-PMIDs. Bounded LLM spend.

**Test scenarios:**
- Happy: confirming abstracts → affirm; contesting abstracts → contestation flag.
- Edge: no edge publications → confirmation skipped, gate score stands (no LLM call).
- Error: SDK failure → leg keeps the gate score, node never throws (reuse v1 resilience).
- Integration: re-run Tier A shows the layer improves (or doesn't change) the verdict vs Phase 1.

- [ ] **V7: Baseline comparator (v1 co-occurrence vs v2 provenance)**

**Goal:** Score the panel with both scorers and report the contrast, so the improvement is measured.

**Requirements:** R6

**Dependencies:** V5; v1 scorer kept eval-only.

**Files:**
- Modify: `backend/assessment_data/bridge_grounding_tier_a.py` (dual-score `--baseline` mode + comparison).
- Modify: `backend/assessment_data/BRIDGE_GROUNDING_TIER_A_FINDINGS.md` (append the v1-vs-v2 result).

**Approach:** Keep v1 modules; add a mode that runs both over the same (now real-edge) panel and emits
both margins side by side.

**Test scenarios:** Test expectation: none — eval/reporting harness over already-tested scorers; the
artifact is the comparison output.

## System-Wide Impact

- **Interaction graph:** `bridge_grounding` inserts before synthesis (reuse v1 wiring); the
  `parse_kestrel_response` change is additive (3 consumers: integration/synthesis/pathway — V1
  characterization test guards them). New retrieval shape: `multi_hop` mode=full + `get_edges`.
- **Error propagation:** per-bridge/per-leg isolation → `insufficient_literature` + `bridge_grounding_errors`;
  node never throws; Kestrel `get_edges` failure (Phase 2) degrades to the gate score.
- **State lifecycle:** `grounded_bridges` no-reducer key, single writer before synthesis reads it (reuse U1).
- **API surface:** more Kestrel load (full-mode edges, optional `get_edges`); consider query-time
  constraints to bound it. Reuses the multi_hop edges already fetched for Phase-1 provenance.
- **Integration coverage:** the live V5 run is the real cross-layer proof — resolution → KG provenance
  → score → gate verdict end to end, on real edges.
- **Unchanged invariants:** `Bridge`/`grounded_bridges`/contracts stay backward-compatible (additive only);
  v1 scorer modules remain importable as the baseline.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| **Circular calibration** — scoring v2 over hand-authored panel predicates proves only the partition isn't inverted, not that real edges separate | V5 reads predicate/KL/agent_type from **live** Kestrel edges, fixtures discarded; the real-edge distribution is the deliverable (deepening F1) |
| **Predicate type alone doesn't separate** — text-mining over-asserts `causes` (SemRep ≈0.55 P), real mechanisms also carry associative edges | Curation-first gate (`knowledge_level`/`agent_type` primary); text-mined causal demoted to associative-tier; V5 verifies on real edges (deepening F4, F6) |
| **Multi-edge "best-causal" reintroduces "studied=high"** | Source-gated aggregation; capture all candidates (U0 change); "best-causal-anywhere" is a registered failure mode checked in V5 (deepening F3) |
| **~18% publication coverage** breaks pub-count weighting AND Phase-2 rescue | Pub-count is a floor/annotation, not a multiplier; V5 measures pub coverage on ambiguous legs before committing Phase 2 (deepening F2) |
| **Popularity/hub bias** — well-studied middle node B inflates the score | Counts are a floor not a score; reconcile with existing >1000-edge hub detection; optional `degree_percentile` query-time constraint (deepening F4) |
| **Resolution is the binding dependency (confirmed by probe)** — wrong-namespace CURIEs (disease→KEGG/PANTHER pathway, wrong-species gene, `VO:` invalid) void legs in BOTH the panel and production. The gate works only on correct MONDO/HGNC/CHEBI CURIEs | Panel uses hand-validated CURIEs (no re-resolution). **Production v2 accuracy is gated on the biomapper resolution fix** (`[[biomapper2-wraps-kestrel-hgnc-marker]]`, `[[biomapper-pre-resolver-state-and-prod-ablation]]`) — sequence v2 enablement behind it; until then v2 is eval-only (already default-off) |
| **`negated` not in slim tuple** → silently counts "X does not cause Y" as support | V1 surfaces `negated` (full mode / `get_edges`); V2 negation guard treats it as refuting (deepening Q3) |

## Phased Delivery

### Phase 1 (V1–V5) — prove the curation-gated deterministic scorer on REAL edges
Build provenance extraction, the curation-first gate, model/node/wiring, and **re-run Tier A on live
Kestrel edges** with a pre-registered weight table + aggregation policy + skip-Phase-2 criterion. The
verdict + the per-leg KL×agent_type×predicate×source distribution decide whether Phase 2 is needed.

### Phase 2 (V6–V7) — add LLM confirmation only where the V5 pre-registration is not met
Bounded LLM confirmation over edge publications (where they exist) + the v1-vs-v2 baseline comparison.

## Documentation / Operational Notes

- Node ships `enabled=False` (eval-only) until Tier B calibration lands (carry v1 decision).
- Update `BRIDGE_GROUNDING_TIER_A_FINDINGS.md` with the v2 verdict + the real-edge distribution + v1/v2 comparison.
- Phase 1 is fully deterministic (KG metadata) — a calibration-trust win; only Phase 2 inherits the
  SDK no-temp/no-model non-determinism (frozen snapshots remain the basis).

## Sources & References

- **Origin document:** docs/brainstorms/2026-06-17-bridge-grounding-v2-kg-provenance-requirements.md
- v1 failure analysis: `backend/assessment_data/BRIDGE_GROUNDING_TIER_A_FINDINGS.md`
- v1 plan (reused units): `docs/plans/2026-06-08-002-feat-bridge-grounding-scorer-plan.md`
- skimgpt mining: `docs/references/km-gpt-dch-competing-hypotheses.md`
- Kestrel API: `.claude/skills/kestrel-api`, `docs/kestrel-api-reference.md`; live edge-shape probe 2026-06-17.
- Biolink Model v4.4.2 (predicate is_a tree, KnowledgeLevelEnum, AgentTypeEnum, negated/qualifiers);
  ARAX PMC10027432; RTX-KG2 BMC Bioinformatics 2022; SemRep PMC7222583; NLP-KG pitfalls arXiv:2310.15572.
- Branch: `feat/bridge-grounding-scorer` (v1 U0–U4 + U8 harness already committed).
