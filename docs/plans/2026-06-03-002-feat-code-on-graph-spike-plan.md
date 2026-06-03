---
title: "feat: Code-on-Graph go/no-go spike harness"
type: feat
status: active
date: 2026-06-03
origin: docs/brainstorms/code-on-graph-spike-requirements.md
deepened: 2026-06-03
---

# feat: Code-on-Graph Go/No-Go Spike Harness

## Overview

Decide whether an LLM-driven **iterative** Kestrel query-refinement loop ("code-on-graph") beats kraken's **static** query plan at endpoint-to-endpoint bridge discovery — **staged** so the idea can be killed cheaply before any heavy methodology is built:

- **Phase 0 — Lean kill-test (MVP):** a hand-curated ~30-item gold set (direct CHEBI/MONDO CURIEs, *no* crosswalk ETL), both arms, and a recall gate (paired McNemar + recall-lift + hallucinated-CURIE=0). A recall NO-GO or any hallucinated CURIE **kills the idea here** — stop, no further build.
- **Phase 1 — Full methodology (runs only on a passing Phase-0 recall signal, toward a defensible PROCEED):** DrugMechDB-scale gold set + crosswalks, the EITL expert precision arm, the treatment-fairness guard, and the reproducibility manifest.

Nothing in the production pipeline changes — `integration.py` is touched only *after* a PROCEED verdict.

## Problem Frame

Kraken's discovery nodes all "fetch-then-reason": fixed query (or fixed fan-out), then reason with `allowed_tools=[]`. No node lets the LLM steer querying based on what it finds; code-on-graph is that missing loop. It's a speculative bet — CoG's published wins come from keeping the KG out of the prompt, which kraken already does, so an iterate loop *adds* turns/calls against an already-lean baseline (origin: `docs/brainstorms/code-on-graph-spike-requirements.md`). The staging exists because document review showed a fully defensible gate re-implements the L+ build's riskiest component as throwaway and needs a live human EITL campaign — disproportionate to a kill-test. Phase 0 buys the kill outcome cheaply; Phase 1 buys defensibility only when warranted. Serves Lance's discovery themes #2 (mechanism of action) and #4 (drug repurposing) — `docs/discovery-pipeline-requirements.md`.

## Requirements Trace

- R1. Standalone throwaway harness; no production node change until PROCEED (origin R1).
- R2. Compare static baseline vs minimal LLM iterate-loop on identical inputs (origin R2).
- R3. Endpoint-to-endpoint: start CURIE + gold-target CURIE → bridge node(s) (origin R3).
- R4. Iterate-loop emits a typed JSON query spec (structured-spec executor); self-corrects to a hard turn cap (origin R4).
- R5–R7. Recall reference = labeled gold bridges (Phase 0: manual set; Phase 1: DrugMechDB-scale), Kestrel-reachable hop-agnostically, N ≥ powered floor (origin R5–R7).
- R8. Precision reference = EITL expert votes on off-gold bridges, blinded, ≥2 reviewers/pair + κ — **Phase 1 only** (origin R8).
- R9. Grounding contract: every emitted CURIE must reference a node returned by an executed Kestrel call; hallucinated CURIE = hard fail (origin R9).
- R10. Report recall, false-confirmation (Phase 1), hallucinated-CURIE rate, cost (LLM + Kestrel calls separately) (origin R10).
- P1–P5. McNemar significance, R0 anchor, cost rule, treatment-fairness guard (Phase 1), frozen definitions (origin Pre-Registration).

## Scope Boundaries

- **No production node changes** (`integration.py:detect_bridges_via_api`, `direct_kg.py`, `pathway_enrichment.py`, a future `graph_reasoning` node).
- **No arbitrary-Python sandbox** — structured-spec executor only.
- **No local subgraph materialization / relation-ranking** — Kestrel stays remote.
- **No category-pair framing in the gate** — endpoint-to-endpoint only.
- **Not run at `synthesis.py:validate_bridge_hypotheses`** (confirmation-seeking site).

### Deferred to Separate Tasks
- Category-pair confirmation arm + L+ `graph_reasoning` build: separate plan, only on PROCEED. **Kill-propagation rule:** a Phase-0 recall NO-GO also kills the category-pair arm (it fails the strictly-easier task) — do not run it without an explicit new decision.
- Repurposing prerequisites (analyte-anchored entry; bridges terminating on drug/intervention nodes): separate discovery-pipeline work.

## Context & Research

### Relevant Code and Patterns
- `backend/tests/recall_gate.py` — **the structural template**: standalone `__main__`, `load_dotenv()`, committed JSON fixture + `_threshold`, per-item table, exit 0=PASS/1=FAIL. `_same_entity`/`_ambiguous` via `equivalent_ids` — the precedent for "did we recover the gold bridge?" and the 1→many identity problem (port the `_ambiguous` distinction into the grounding check).
- `backend/src/kestrel_backend/assessment/{runner,cassette,scorer,capture}.py` — argparse runner that continues past failures, sets `LANGFUSE_ENABLED=false`; respx record/replay keyed on `method:url:md5(body)` — **replay raises on any unmatched body** (governs the cassette decision below); scipy stats; Pydantic→JSON.
- `backend/src/kestrel_backend/kestrel_client.py` — `multi_hop_query(...max_hops≤5, limit=100...)`; `call_kestrel_tool`; **must check `result.get("isError")`** (auth-as-content trap, `:259-275`); lazy connect via `KESTREL_API_KEY`.
- `backend/src/kestrel_backend/graph/nodes/integration.py` — `parse_multi_hop_result` **hardcodes `paths[:10]`** (`:323`), no disable param → the spike needs its own **cap-free shared parser** used identically by both arms.
- `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` — `_canonical_curie` (`:290`) + the R1a membership gate: the grounding template.
- `backend/src/kestrel_backend/graph/sdk_utils.py` — `query_with_usage` → `(text, ModelUsageRecord)`; `ClaudeAgentOptions(allowed_tools=[], max_turns, permission_mode="bypassPermissions")` — **no `temperature` field exists**; `ModelUsageRecord` tracks tokens, not Kestrel-call/turn counts (harness adds its own).
- `backend/src/kestrel_backend/kestrel_tools.py` — dormant; its **12 tool names + input schemas are the enumerated verb whitelist** for the structured-spec executor (extract them programmatically; do not run it as a live MCP server).
- `expert-in-the-loop` `shared/schema.ts` (`pairs` 7 required cols; `votes` supersession chain → κ uses **active votes only**); `client/src/components/ColumnMapper.tsx` (auto-maps `llm_confidence/llm_model/llm_reasoning/resolution_layer` into first-class fields); `client/src/pages/review.tsx` (**renders** `llmReasoning/confidence/model` and renders metadata when `showMetadataPanel` is on).

### Institutional Learnings
- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md` — stdio MCP fabricates KG facts; loop must fetch via HTTP `call_kestrel_tool` + reason with `allowed_tools=[]`; never count a degraded/fabricated bridge as a recall hit.
- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` — `cd backend && uv run python -m pytest`; assert `sys.executable` is the venv.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — pre-commit thresholds; **cassette replay does NOT remove LLM nondeterminism** → variance bands, not fixed-percentage determinism.
- `docs/solutions/best-practices/verify-temporal-provenance-before-kg-holdout-eval-2026-05-29.md` — score recall under OWA; no date-frozen Kestrel holdout.
- `docs/solutions/runtime-errors/triage-tier2-undefined-variable-2026-05-06.md` — SDK/loop paths ship crashes; force-execute the loop in tests; `ruff check --select F821`.

### External References
- **DrugMechDB** (`github.com/SuLab/DrugMechDB`, **CC0-1.0**): single `indication_paths.yaml`; drug=`links[0].source`, disease=`links[-1].target`, ordered bridges = interior nodes. **Phase 1 only** (Phase 0 uses a manual set).
- **Crosswalks (Phase 1):** DrugBank→CHEBI via UniChem `wholeSourceMapping` (src 2→7); MeSH→MONDO via MONDO `mondo_exactmatch_mesh.sssom.tsv`; UniProt→NCBIGene via UniProt `idmapping.dat.gz`.
- **Stats** (`statsmodels`): `contingency_tables.mcnemar(table, exact=True)` primary, `exact=False, correction=False` sensitivity (avoid deprecated `sandbox.stats.runs.mcnemar`); power depends on discordance π_D not N (Monte-Carlo the exact test); `inter_rater.cohens_kappa`/`fleiss_kappa`; TREC pooling → "pooled recall" (relative robust, absolute biased up).

## Key Technical Decisions

- **Staged build.** Phase 0 lean kill-test (manual gold set, both arms, recall gate) can issue a NO-GO without crosswalk ETL, EITL, fairness guard, or manifest. Phase 1 adds those only on a passing recall signal / toward PROCEED.
- **No determinism claim — the gate is a variance-band measurement.** The Agent SDK exposes **no temperature control**, so the iterate-loop is non-deterministic. Run each item **K≥3 times**; score per-item recall as a stable majority (a item is a "hit" iff ≥⌈K/2⌉ runs find the gold bridge), and report the run-to-run variance band. No point-estimate "deterministic verdict."
- **Cassette policy (resolves the replay break).** Only the **static baseline** (one fixed query/trial) replays from a cassette. The **iterate arm runs live** against Kestrel with **record-as-you-go**; a cassette miss mid-loop **fails the trial loudly** (recorded `terminal_state=cassette-miss`), never a silent live fallback that would corrupt pairing. McNemar pairing is per-item (same gold endpoints), not per-KG-snapshot.
- **Cap-free shared parser, total-evidence budget held equal.** Both arms use one harness-owned parser without the production `paths[:10]` cap. The quantity held equal is the **aggregate distinct-path budget** (the loop's cumulative paths across turns are capped at the same total the baseline may return), not a per-call cap — so a multi-call loop can't out-evidence a single-call baseline by accident.
- **Recall and precision are disjoint, and recall is the kill axis.** Recall (Phase 0/1) = gold-path hits vs the labeled bridge, scored under OWA; EITL-validated off-gold bridges **do not** feed back into recall. Precision/false-confirmation (Phase 1) is a **PROCEED-confirmation** criterion, never a kill criterion. The plan states explicitly that on single-path gold the exploratory iterate arm may be *penalized* on recall and *rewarded* on precision — they can pull opposite directions, which is why recall+McNemar (not precision) gates the kill.
- **Grounding** reuses `_canonical_curie` + `equivalent_ids` (port `recall_gate.py:_ambiguous` for the 1→many case); a per-trial boolean violation rolls up to a harness-wide count; any > 0 → hard NO-GO.
- **Pilot is baseline-only.** R0 from a baseline-only run on ~10 items + a **conservative π_D prior (0.25)** sets the powered-N via Monte-Carlo exact-McNemar. The iterate arm is built only after N is confirmed achievable. No circular dependency.
- **Frozen config, plain committed file (Phase 0); hashed manifest (Phase 1 only).** Phase 0 pre-registration = a committed `config.py` (git timestamp is the freeze proof). The hash-enforcing manifest is Phase-1 reproducibility infra, not needed to kill the idea.
- **Per-turn Kestrel-call cap frozen from principle, not pilot** (baseline's 1 call × an a-priori multiplier), so the cost gate isn't tuned on treatment behavior.

## Open Questions

### Resolved During Planning
- Determinism? → not available; variance-band measurement with K≥3 reruns + majority hit.
- Cassette for the loop? → loop runs live/record-as-you-go; baseline replays; miss = fail-loud.
- Pilot circularity? → baseline-only R0 + conservative π_D prior.
- DrugMechDB license / extraction? → CC0; walk `links`.
- McNemar variant? → exact primary, asymptotic uncorrected sensitivity.

### Deferred to Implementation
- Kestrel `multi_hop_query` response shape (`{"paths":[...]}` vs bare list) — verify on the first live call.
- UniChem `src_id`s (DrugBank=2, ChEBI=7) — verify before hardcoding (Phase 1).
- Whether the manual Phase-0 gold set reaches N≥30 by hand in <4h, else trigger Phase-1 crosswalk early.

## Output Structure

    backend/tests/code_on_graph_spike/
      __init__.py
      config.py                 # frozen Pydantic pre-registration (Phase 0 plain file)
      kestrel_paths.py          # cap-free shared parser + grounding (_canonical_curie/equivalent_ids)
      baseline.py               # static multi_hop_query arm (+ cassette replay)
      iterate_loop.py           # structured-spec executor, live record-as-you-go, K reruns
      pilot.py                  # baseline-only R0 + conservative π_D → powered-N
      recall_scorer.py          # majority-of-K recall vs gold (OWA), McNemar table
      gate_recall.py            # Phase-0 recall gate → NO-GO or proceed-to-Phase-1
      # ---- Phase 1 (built only on a passing recall signal) ----
      drugmechdb.py / crosswalk.py / gold_set.py
      eitl_export.py            # blinded precision arm
      fairness_guard.py         # under-specified seed + negative control
      gate_full.py / manifest.py / run_spike.py
      test_*.py                 # companion mocked-Kestrel unit tests
    backend/tests/fixtures/code_on_graph_spike/
      gold_set_manual.json      # Phase 0 hand-curated ~30 items, direct CURIEs
      gold_set.json             # Phase 1 DrugMechDB-derived
      config_frozen.json

## High-Level Technical Design

> *Directional guidance for review, not implementation specification.*

```
PHASE 0 (lean kill-test)                                     PHASE 1 (only if recall passes)
 manual gold_set (≈30, direct CURIEs)                          DrugMechDB → crosswalk → reachability
   └ baseline-only pilot → R0, π_D=0.25 prior → powered-N        → scaled gold set
   └ baseline arm (cassette replay)  ─┐                          EITL blinded precision arm → κ, false-confirm
   └ iterate arm (live, K≥3 reruns)  ─┴→ majority hit/miss       fairness guard (under-spec seed + neg control)
   └ recall gate: McNemar(exact) + lift + hallucinated=0         full gate + verdict lattice + hashed manifest
        ├ recall NO-GO or hallucinated>0 → KILL (stop)
        └ recall passes → PHASE 1
```

Iterate-loop control (per item, K reruns; no temperature control → variance band):
```
LLM emits JSON spec {verb∈whitelist, start, end, max_hops≤5, predicate_filter, degree_constraint}
  → validate vs the 12-verb whitelist  → dispatch call_kestrel_tool (live, record-as-you-go)
  → transport isError: retry transparently, max 2, OUTSIDE turn budget; then terminal_state=transport-failed
  → valid-empty / malformed-spec: feed back, consumes 1 turn (cap 5)
  → grounding: emitted CURIE ∈ canonical(returned ∪ equivalent_ids), else terminal_state=grounding-violation (hard-fail count++)
  → stop: bridge found | turn cap | empty-exhausted | aggregate-path-budget hit
```

## Implementation Units

### Phase 0 — Lean kill-test (issues a cheap NO-GO; gates Phase 1)

- [ ] **Unit 0.1: Manual gold set + frozen pre-registration config**

**Goal:** A committed ~30-item gold set (direct CHEBI/MONDO CURIEs, hand-verified bridge) and the frozen threshold config — no crosswalk ETL.

**Requirements:** R3, R5, R7, P1, P3, P5

**Dependencies:** None

**Files:**
- Create: `backend/tests/code_on_graph_spike/config.py`
- Create: `backend/tests/fixtures/code_on_graph_spike/gold_set_manual.json`
- Test: `backend/tests/code_on_graph_spike/test_config.py`

**Approach:**
- Hand-curate ≥30 drug→bridge→disease items where the drug, disease, and ≥1 interior bridge node are entered as **direct Kestrel CURIEs** (CHEBI/MONDO/NCBIGene/GO), each `trial_id`-keyed with `{start_curie, gold_target_curie, gold_bridge_curies[], hop_length}`. Include ≥2 known-easy 2-hop items as the loop smoke test (replaces the heavyweight fairness guard for Phase 0).
- Frozen Pydantic config (plain committed file, git timestamp = freeze proof): recall-lift 15% (+relative form if R0>85%), α=0.05, McNemar exact primary, turn cap 5, per-turn Kestrel-call cap (principled: baseline-calls × a-priori multiplier), aggregate-path budget, K reruns (≥3), majority-hit rule, N floor.
- **Bridge unit (frozen):** a recall hit = a returned path containing **all** gold interior node(s) (identity via `_canonical_curie`+`equivalent_ids`); the "any-one-interior" variant is reported for sensitivity only and does **not** affect the gate.

**Patterns to follow:** `recall_gate.py` fixture + threshold; `graph/pipeline_config.py` frozen config.

**Test scenarios:**
- Happy path: config loads; all thresholds present and typed; model asserts immutable.
- Edge case: gold_set has <30 items → loader flags "trigger Phase-1 crosswalk".
- Edge case: a gold item's CURIE has bad prefix → validation error, not silent skip.
- Test expectation: gold-set schema round-trips (Pydantic) with `trial_id` unique.

- [ ] **Unit 0.2: Cap-free parser + static baseline arm + baseline-only pilot**

**Goal:** The static arm, the shared parser, and R0/π_D/powered-N.

**Requirements:** R2, R3, R5, R7, P1, P2, R10

**Dependencies:** Unit 0.1

**Files:**
- Create: `backend/tests/code_on_graph_spike/kestrel_paths.py`, `baseline.py`, `pilot.py`
- Test: `test_kestrel_paths.py`, `test_baseline.py`, `test_pilot.py`

**Approach:**
- `kestrel_paths.py`: a **cap-free** path parser (no `paths[:10]`), node-identity helpers reusing `_canonical_curie`/`equivalent_ids`/`_same_entity` semantics, and the grounding check (port `_ambiguous`). Both arms import this — identical evidence handling.
- `baseline.py`: `multi_hop_query(start, end, max_hops=5, limit=L)`; check `isError`; cassette-replay supported (one fixed body/trial). Record per-trial `{trial_id, method, bridges, hop_counts, kestrel_calls=1, terminal_state}`.
- `pilot.py`: baseline-only on ~10 items → R0; combine with conservative π_D=0.25 prior; Monte-Carlo exact-McNemar to set N=`max(30, powered-N)`. Verify the live `multi_hop_query` response shape first. If reachable N<floor → emit INCONCLUSIVE before building the iterate arm.

**Execution note:** Confirm the live response shape before trusting the parser.

**Patterns to follow:** `recall_gate.py`; `assessment/scorer.py` scipy.

**Test scenarios:**
- Happy path: known 2-hop item → baseline returns gold bridge → hit (cap-free, full path set).
- Edge case: empty result → miss, `terminal_state=empty`.
- Error path: `isError` transport → retried (max 2) → `terminal_state=transport-failed`, not a method miss.
- Happy path: pilot computes R0 + powered-N; Monte-Carlo matches closed-form sanity check.
- Edge case: R0=86% → gate switches to relative form (frozen at pilot); R0=84% → absolute.
- Edge case: reachable N=floor−1 → INCONCLUSIVE, halt before iterate arm.

- [ ] **Unit 0.3: Iterate-loop arm (structured-spec executor, K reruns, live)**

**Goal:** The treatment — JSON-spec query loop with grounding, bounded cost, variance band.

**Requirements:** R2, R4, R9, R10, P3

**Dependencies:** Unit 0.1, Unit 0.2 (shared parser/grounding; powered-N confirmed)

**Files:**
- Create: `backend/tests/code_on_graph_spike/iterate_loop.py`
- Test: `test_iterate_loop.py`

**Approach:**
- LLM (`query_with_usage`, `allowed_tools=[]`) emits a typed JSON spec validated against the **enumerated 12-verb whitelist** (extracted from `kestrel_tools.py`); harness dispatches `call_kestrel_tool` **live with record-as-you-go**; results fed back as prompt text.
- **No temperature control** → run each item **K≥3 times**; per-item hit = majority of K; record the variance band. Cassette miss → `terminal_state=cassette-miss` (fail loud).
- **Transport vs empty:** transport `isError`/SSE/auth-as-content retry transparently (max 2) *outside* the turn budget; only valid-empty/malformed-spec consumes a turn (cap 5). **Grounding:** emitted CURIE ∉ canonical(returned ∪ equivalent_ids) → `grounding-violation`, increments the hard-fail count. **Aggregate-path budget** capped equal to baseline.
- Persist transcript + raw Kestrel responses; LLM-call and Kestrel-call counters separate (includes `get_nodes`/`equivalent_ids` lookups consumed by grounding).

**Execution note:** Test-first against the failure paths (loop/fallback is the crash-prone path); run `ruff check --select F821`.

**Test scenarios:**
- Happy path: loop finds gold bridge within turn cap → hit; majority-of-K stable.
- Edge case: turn cap hit → `terminal_state=turn-cap-hit`, miss.
- Error path: malformed/invalid-verb spec → consumes a turn, re-prompted.
- Error path (R9): emit a CURIE Kestrel never returned → grounding-violation (assert `chebi:6801` returned vs `CHEBI:6801` emitted is **grounded**, not a violation).
- Error path: transport error mid-loop → retried outside turn budget, no turn consumed; cassette miss → fail-loud terminal state.
- Integration: LLM-call vs Kestrel-call counters (incl. grounding lookups) recorded separately.

- [ ] **Unit 0.4: Recall gate (Phase-0 kill decision)**

**Goal:** Apply the recall-axis gate and decide KILL vs proceed-to-Phase-1.

**Requirements:** R5, R9, R10, P1, P2, P3

**Dependencies:** Unit 0.2, Unit 0.3

**Files:**
- Create: `backend/tests/code_on_graph_spike/recall_scorer.py`, `gate_recall.py`
- Test: `test_recall_scorer.py`, `test_gate_recall.py`

**Approach:**
- `recall_scorer.py`: majority-of-K per-item hit/miss for both arms vs gold bridge (OWA); build the paired `[[a,b],[c,d]]` table; report pooled recall + the concordant-miss count separately.
- `gate_recall.py`: recall lift (absolute/relative per frozen P1) **and** McNemar significance (`mcnemar(table, exact=True)`, α=0.05; asymptotic sensitivity); hallucinated-CURIE roll-up; cost worst-case (turn-cap × principled per-turn cap, LLM + Kestrel separately) ≤3× baseline. **Phase-0 lattice:** `hallucinated>0` → NO-GO (override) → recall lift absent or McNemar n.s. → NO-GO (idea killed) → N<powered or variance band flips the McNemar verdict across reruns → INCONCLUSIVE → else **proceed to Phase 1**.

**Test scenarios:**
- Happy path: lift clears bar + McNemar significant + hallucinated=0 → proceed-to-Phase-1.
- Edge case: lift ≥15% but McNemar p>0.05 → NO-GO (point estimate alone fails).
- Edge case: hallucinated=1 → NO-GO override regardless of recall.
- Edge case: McNemar verdict flips across the K-rerun variance band → INCONCLUSIVE.
- Integration: scorer table feeds the gate; verdict reproduced on the committed fixture.

### Phase 1 — Full methodology (built only on a passing Phase-0 recall signal)

- [ ] **Unit 1.1: DrugMechDB ingestion + crosswalk + reachability (scale the gold set)**

**Goal:** Replace the manual set with a DrugMechDB-derived, crosswalk-mapped, Kestrel-reachable gold set at the powered N.

**Requirements:** R5, R6, R7

**Dependencies:** Phase-0 PROCEED-signal

**Files:** Create `drugmechdb.py`, `crosswalk.py`, `gold_set.py`, `backend/tests/fixtures/code_on_graph_spike/gold_set.json`; Test `test_drugmechdb.py`, `test_crosswalk.py`, `test_gold_set.py`

**Approach:** Parse `indication_paths.yaml` (CC0); drug=`links[0].source`, disease=`links[-1].target`, ordered interior bridges by walking links (DFS fallback on branched records). Normalize prefixes. Map DrugBank→CHEBI (UniChem), MeSH→MONDO (MONDO SSSOM), UniProt passthrough/→NCBIGene; frozen 1→many tie-break (include only if exactly one Kestrel-reachable CHEBI; reuse `_same_entity`). **Hop-agnostic reachability** (≤5 hops = executor cap, NOT the 2-hop baseline); exclude >5-hop gold paths and node-types absent from Kestrel (report strata). Coverage report of mapped/unmapped by namespace.

**Test scenarios:** Happy: 2-link record → correct extraction. Edge: multi-intermediate → ordered list; branched → DFS/flagged. Edge: 1→2 CHEBI → excluded, logged. Edge: >5-hop gold → excluded stratum. Error: unmapped id → coverage report, not dropped.

- [ ] **Unit 1.2: EITL blinded precision arm**

**Goal:** Expert false-confirmation judgment on off-gold bridges (the PROCEED-confirmation criterion).

**Requirements:** R8, R10, P5

**Dependencies:** Unit 1.1 (or Phase-0 pool); **Precondition (do first):** read `client/src/pages/review.tsx` + confirm the campaign sets `display.showMetadataPanel=false`, and choose CSV headers that do **not** match the `ColumnMapper` auto-aliases (`llm_*`, `resolution_layer`). A runtime assertion in `eitl_export.py` fails the export if any method-correlated field maps to a rendered column.

**Files:** Create `eitl_export.py`; Test `test_eitl_export.py`

**Approach:** Pool both arms' off-gold bridges, dedup by unique bridge (`method` set-valued in metadata, one pair per unique bridge), shuffle. Emit the 7 required EITL columns; **all method-correlated signal (`llm_*`, `method`, `on_drugmechdb_path`, `trial_id`) in metadata under non-auto-mapped header names, never rendered**. Ingest **active votes only** (filter supersession chain); `no_match`=spurious, `unsure` excluded; pair-level aggregation frozen (≥2 reviewers; 1-1 → 3rd adjudicator). Cohen's/Fleiss' κ on raw labels + report raw agreement & prevalence (kappa paradox); κ<0.6 → precision INCONCLUSIVE.

**Test scenarios:** Happy: pool → 7-column CSV ingests. Edge (blinding): assert **no** rendered column carries `llm_*`/method; export fails if it would. Edge: same bridge both arms → one pair, method={static,iterate}. Edge: superseded votes excluded. Edge: 1-1 split → adjudication. Edge: κ<floor → precision INCONCLUSIVE.

- [ ] **Unit 1.3: Treatment-fairness guard (discriminating)**

**Goal:** Separate "iterate genuinely loses" from "weak impl" without being theatre.

**Requirements:** P4

**Dependencies:** Unit 0.3

**Files:** Create `fairness_guard.py`; Test `test_fairness_guard.py`

**Approach:** Disjoint held-out known-reachable set (sized so gate N still clears floor; skip with a logged note if N-budget can't afford it). Seed spec **under-specified by ≥1 degree the loop must recover** (correct endpoints, wrong/absent predicate or max_hops) so passing requires ≥1 genuine self-correction. **Negative control:** a steering-disabled loop must FAIL the guard on the same set (proves discrimination). Below pass-fraction → eventual kill reported INCONCLUSIVE. Runs before the full gate, one-way info barrier (must not retune the loop).

**Test scenarios:** Happy: real loop recovers ≥ pass-fraction → guard passes. Edge: steering-disabled negative control fails the guard. Edge: held-out overlaps gate set → disjointness assertion. Edge: N-budget too small → guard skipped with logged caveat.

- [ ] **Unit 1.4: Full gate + manifest + orchestrator**

**Goal:** The defensible PROCEED/NO-GO/INCONCLUSIVE verdict with reproducibility.

**Requirements:** Gate criteria, P1–P4, R10

**Dependencies:** Units 1.1–1.3, Unit 0.4

**Files:** Create `manifest.py`, `gate_full.py`, `run_spike.py`; Test `test_gate_full.py`

**Approach:** Hashed artifact manifest (DrugMechDB SHA, crosswalk versions, Kestrel API version, model id, loop seed spec, thresholds, **EITL UI commit SHA + blinding-verified flag**); gate refuses on hash mismatch. **Verdict lattice:** `hallucinated>0`→NO-GO → any criterion fail→NO-GO → any inconclusive (N<powered / κ<floor / fairness-guard fail / McNemar n.s. / variance-band flip)→INCONCLUSIVE → else PROCEED. `run_spike.py` chains Phase 0→1, per-criterion table + headline verdict, exit-code by verdict.

**Test scenarios:** Happy: all pass + McNemar significant → PROCEED. Edge: κ<floor → INCONCLUSIVE. Edge: manifest hash mismatch → refuses. Edge: fairness-guard fail → INCONCLUSIVE not NO-GO. Integration: end-to-end tiny fixture → reproducible verdict.

## System-Wide Impact

- **Interaction graph:** Standalone — no LangGraph node/contract modified. Reads live Kestrel; Phase 1 creates an EITL campaign.
- **Error propagation:** transport errors retry (max 2) then `transport-failed`, never a method miss or hallucination; per-trial failures captured, not raised; cassette miss fails loud.
- **State lifecycle:** `trial_id` minted in the gold set, threaded through blinded EITL votes back to gold (else recall uncomputable); raw responses + transcripts persisted for adjudication.
- **API surface parity:** none.
- **Integration coverage:** DrugMechDB→Kestrel mapping, blinded-export→vote-rejoin, live iterate vs cassette-baseline pairing — exercise on a small live slice.
- **Unchanged invariants:** `integration.py`, `synthesis.py`, the DAG, all state contracts.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| No temperature control → loop nondeterministic | K≥3 reruns, majority hit, variance band; verdict INCONCLUSIVE if McNemar flips across the band |
| Cassette can't replay loop's novel queries | Loop runs live/record-as-you-go; baseline replays; cassette miss = fail-loud |
| EITL UI renders `llm_*`/metadata → un-blinds | Phase-1 precondition reads `review.tsx`; non-auto-mapped headers; `showMetadataPanel=false`; export assertion |
| Pilot circularity | Baseline-only R0 + conservative π_D prior; iterate built after N confirmed |
| Aggregate-evidence asymmetry (multi-call loop > single-call baseline) | Hold the aggregate distinct-path budget equal, not per-call |
| Per-turn cap tuned on treatment = forking path | Cap frozen from principle (baseline-calls × a-priori multiplier) |
| Recall (single-path gold) penalizes exploratory iterate | Recall is the kill axis; precision is PROCEED-confirmation; opposite-pull acknowledged, not conflated |
| Crosswalk attrition / spike-weight | Phase 0 needs no crosswalk; Phase 1 only on a passing recall signal |
| stdio-MCP fabrication | HTTP `call_kestrel_tool` + `allowed_tools=[]`; degraded outputs never count as hits |

## Documentation / Operational Notes
- Throwaway, not in CI; run `cd backend && uv run python tests/code_on_graph_spike/run_spike.py`; unit tests `cd backend && uv run python -m pytest tests/code_on_graph_spike/`.
- Phase 1 creates one EITL `custom` campaign — coordinate around its production launch.
- After verdict, archive `backend/tests/code_on_graph_spike/` to `docs/experiments/` (NO-GO/INCONCLUSIVE) or promote to `assessment/` (PROCEED); don't leave it in `tests/` indefinitely.

## Sources & References
- **Origin:** [docs/brainstorms/code-on-graph-spike-requirements.md](docs/brainstorms/code-on-graph-spike-requirements.md)
- Feasibility: `docs/code-on-graph-feasibility.md`; node map: `docs/pipeline-node-tool-map.md`; pipeline wants: `docs/discovery-pipeline-requirements.md`
- Code: `backend/tests/recall_gate.py`, `backend/src/kestrel_backend/assessment/{runner,cassette,scorer,capture}.py`, `backend/src/kestrel_backend/kestrel_client.py`, `backend/src/kestrel_backend/graph/nodes/{integration,entity_resolution}.py`, `backend/src/kestrel_backend/graph/sdk_utils.py`, `backend/src/kestrel_backend/kestrel_tools.py`, `expert-in-the-loop/{shared/schema.ts,client/src/components/ColumnMapper.tsx,client/src/pages/review.tsx}`
- Learnings: the five `docs/solutions/` files cited in Context
- External: DrugMechDB (CC0); UniChem; MONDO SSSOM; statsmodels `contingency_tables.mcnemar`, `inter_rater`; TREC pooling (Buckley & Voorhees, Zobel)
