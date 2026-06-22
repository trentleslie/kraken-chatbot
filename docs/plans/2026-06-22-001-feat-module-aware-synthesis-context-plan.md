---
title: "feat: Module-aware synthesis context (fix synthesis-overflow at module scale)"
type: feat
status: active
date: 2026-06-22
origin: docs/brainstorms/2026-06-18-run-brown-module-through-pipeline-requirements.md
---

# feat: Module-aware synthesis context (fix synthesis-overflow at module scale)

## Overview

The discovery pipeline's `synthesis` node assembles its LLM context from accumulated state via
**uncapped per-entity formatters**. At module scale (the 48-analyte Brown C1 run, 2026-06-22, git
`09b0f61`: 44 well-characterized entities, 4,099 findings) the assembled context **exceeds the model's
input-token window** (~882K chars ≈ **230K tokens** vs sonnet-4's ~200K-token limit — see Diagnostic
Evidence) and the Claude Agent SDK synthesis call fails (observed on console as `Fatal error in
message reader: Command failed with exit code 1`). The `except` at
`backend/src/kestrel_backend/graph/nodes/synthesis.py:1010` **silently** falls back to
`fallback_report()`, producing an 882KB raw per-entity dump with **no module-level narrative** — while
the run reports `status=complete, errors=0` (synthesis took 3.5s vs the 24-analyte pilot's real 150s
LLM call). The actual deliverable was not produced, and nothing surfaced it.

This plan **redesigns what synthesis sees**: replace the per-entity raw dumps with (a) cross-entity
aggregation (diseases/pathways recurring across ≥N module members) and (b) a compact per-member
prioritization table, cap the findings section, make every cap configurable against a **token-derived
budget**, and make the SDK-failure degradation **visible and recorded** instead of silent. The result
is a bounded context that produces a genuine module-level report and — once the cap value is proven by
the R7 run — never crashes; until then it never *silently* collapses to a raw dump.

## Problem Frame

The pipeline was built for a handful of entities per query; synthesis context assembly was never
bounded for module-scale input. The 24-analyte C1 pilot (12 well-char entities) fit and produced a
strong module narrative ("Brown WGCNA Module — Inflammatory-Metabolic Stress Axis"); doubling to 48
(44 well-char) broke it. This is the synthesis half of Phase B of the Brown-module effort
(see origin: `docs/brainstorms/2026-06-18-run-brown-module-through-pipeline-requirements.md`). It
removes the **synthesis-overflow** blocker, which is the first thing that breaks at module scale; it
is **not** sufficient on its own to make a full-217 run practical — that remains gated on
`pathway_enrichment` latency (~307s now; origin C1 measured ~47s/analyte ⇒ a single full-217 run is
~1–2h) and the `cold_start` cap-of-8 silent drop (origin: ~97/105 sparse metabolites dropped at full
scale). bridge_grounding, the prior dominant cost, is already fixed (~400s → ~35s, PR #83). Two design
questions were resolved during brainstorming with the user:
- **Scope:** module-aware redesign (not just minimal caps).
- **Per-member data:** aggregate **and** keep a cheap per-member prioritization table (the "hybrid"
  the origin doc flagged as the highest-leverage open question — answered here in favor of capturing
  both the module-theme and per-member axes).

### Diagnostic Evidence (measured 2026-06-22 from the failure artifact)

Offline analysis of the failing run's deterministic dump
(`brown_diagnostic_runs/brown_c1_pilot_20260622T070237Z.json`, the 881,749-char `fallback_report`,
which shares synthesis's section formatters and underlying data) establishes the root cause and sizes
the fix — **no paid re-run required**:

- **Root cause = input-token-window overflow.** 881,749 chars ≈ **230K tokens** (regex BPE-proxy;
  **3.80 chars/token** for this CURIE/predicate-dense content — *below* the 4.0 rule of thumb, so a
  char cap chosen via 4.0 would under-protect). Every estimate (220K–294K tokens across 4.0–3.0
  chars/token) **exceeds sonnet-4's ~200K-token input window**. The failure is a deterministic
  over-window rejection, not transient flakiness. *(Caveat: the literal SDK error string is
  console-only and absent from the artifact; the size math is dispositive independently.)*
- **The plan caps the right sections.** Per-section char contribution of the dump:
  **Analysis Findings 514,537 (58%)**, **Disease Associations 184,067 (21%)**, **Pathway/Process
  Memberships 151,043 (17%)** — together **96%**. Everything else combined is ~4%. Capping findings +
  aggregating disease/pathway therefore targets essentially all of the bloat.
- **Finding count: 4,099** (not 37). Confirms the harness ~110× under-count (R6) and that "~4,000" was
  a correct estimate, not an artifact of the broken counter. ≈125 chars/finding.
- **Derived budget (pre-committed, not "tune later").** Target assembled context ≤ ~100K tokens
  (≈ **350,000 chars** at 3.5 chars/token), leaving ~100K-token headroom under the 200K window for the
  system prompt + model output. That is ~2.5× below the 882K/230K that crashed — conservative by
  construction. This becomes the initial `max_context_chars` (R7 confirms; tune only downward if
  needed).

## Requirements Trace

- **R1.** Synthesis LLM call must **succeed** at module scale (≥44 well-characterized entities) —
  i.e., the assembled context stays within a configurable budget **derived from the model's ~200K-token
  input window** (the real ceiling; `max_context_chars` is a char *proxy* for it, see Diagnostic
  Evidence) so the SDK call does not exceed the window.
- **R2.** The synthesis report at module scale is a coherent **module-level narrative**, not a raw
  per-entity dump (cross-entity theme + per-member prioritization).
- **R3.** When the synthesis LLM call fails for any reason, the degradation is **visible**: logged at
  WARNING with the exception and recorded in `state["errors"]` (so coverage/monitoring sees it). No
  silent fallback.
- **R4.** All new bounds are **configurable** via the existing `pipeline_config` pattern, with
  `Field(description=...)` documenting why each default exists.
- **R5.** **Single-entity** queries are **unaffected** (the per-entity report shape is preserved
  verbatim). Module-aware assembly engages only at `module_mode_min_entities` resolved entities
  (default >2, so genuine *pair* queries also keep the per-entity shape — see Key Technical Decisions);
  above that threshold the report shape changes by design.
- **R6.** The assessment harness `coverage()` must report **accurate** finding/association counts
  (it currently under-reports ~110× — verified: 4,099 actual findings vs 37 reported) so the fix can
  be honestly verified. *(Sizing for this plan no longer depends on it — the Diagnostic Evidence below
  measured the failing context directly — so Unit 6 is a harness-honesty cleanup that can ship
  independently, not a gate for Units 1–5.)*
- **R7.** End-to-end: re-running the 48-analyte Brown pass produces an LLM narrative (not the
  deterministic fallback), under the token-derived budget, with a human-sized report, **verified by
  machine-checkable assertions** (not only a human read — see Unit 7).

## Scope Boundaries

- Not changing `pathway_enrichment` / `integration` node logic; aggregation reuses their state output.
- Not addressing the new `pathway_enrichment` ~307s latency (separate perf task).
- Not addressing resolution data-quality quirks (GH1→somatropin, alpha-ketoglutarate→cyanohydrin,
  proline category=Protein) — those belong to biomapper/resolution.
- Not implementing the Semantic Scholar circuit-breaker (separate, tracked).
- Not changing reducer semantics in `state.py` (the additive reducers are correct; only the harness
  that *reads* streamed deltas is wrong).

### Deferred to Separate Tasks

- **Bounded SDK retry-then-raise** for transient synthesis failures: the learnings
  (`docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md`) recommend
  bounded retry for the intermittently-flaky SDK subprocess. This plan's caps address the
  *deterministic* context-size failure; a general retry wrapper is a broader SDK-reliability change —
  future iteration.
- **Tuning cap values for the full-217 scale**: ship with the token-derived defaults above (pinned
  from the 200K window, validated at 48 analytes by R7); the headline 217-analyte scale (~79 well-char
  per origin C0) is never exercised here, so re-confirm/tune the budget before a full-217 run. This is
  re-validation at a larger scale, **not** shipping unvalidated values now.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/nodes/synthesis.py` — the node. Key functions:
  `assemble_synthesis_context` (line 711), `fallback_report` (line 810), `run` (line 960),
  `SYNTHESIS_PROMPT` (line 41), and formatters `format_disease_associations` (199),
  `format_pathway_memberships` (248), `format_findings_summary` (376), `format_entity_summary` (119).
  Existing caps are module constants (`MAX_LIT_PAPERS_PER_HYPOTHESIS=4`, `MAX_LIT_ABSTRACT_CHARS=1500`,
  lines 634–635) + inline slices. **synthesis.py does not currently import `get_pipeline_config`** —
  wiring it is net-new.
- `backend/src/kestrel_backend/graph/pipeline_config.py` — config pattern to mirror:
  `class XConfig(BaseModel)` with `Field(default=, description=)`; register on `PipelineConfig`
  (lines 298–314) via `name: XConfig = Field(default_factory=XConfig)`; access via
  `@lru_cache get_pipeline_config()` (lines 317–325). Closest example: `BridgeGroundingConfig`
  (lines 270–295). Nested-config precedent: `EntityResolutionConfig` embeds `BiomapperConfig`.
- `backend/src/kestrel_backend/graph/state.py` — `DiscoveryState` (TypedDict, line 378). Additive
  reducer fields (accumulate, may contain duplicates across branches): `resolved_entities` (410,
  `Annotated[list[EntityResolution], operator.add]`), `novelty_scores` (414), `direct_findings` (423),
  `cold_start_findings` (424), `disease_associations` (427), `pathway_memberships` (428). Additional
  additive list keys (also accumulate; relevant to Unit 6's count fix): `inferred_associations` (429),
  `analogues_found` (430), `hub_flags` (431), `shared_neighbors` (434), `gap_entities` (447),
  `temporal_classifications` (450), `model_usages` (471), `literature_errors` (461),
  `bridge_grounding_errors` (467). Plain last-write-wins: `bridges` (446), `hypotheses` (458),
  `biological_themes` (435). `errors` (475) is additive and read by coverage. Models: `DiseaseAssociation` (entity_curie, disease_curie, disease_name,
  evidence_type), `PathwayMembership` (entity_curie, pathway_curie, pathway_name), `NoveltyScore`
  (curie, edge_count, classification), `Finding` (entity, claim, tier, confidence).
- `backend/tests/test_bridge_grounding_node.py` — **template** for the new synthesis tests: tests
  synthesis `format_*` helpers directly, config gating via
  `monkeypatch.setattr(<node>, "get_pipeline_config", lambda: cfg)`, and `run()`.
- `backend/tests/test_pathway_enrichment.py` — `HAS_SDK` monkeypatch + lightweight
  dict/`SimpleNamespace` state construction.
- `backend/assessment_data/brown_c1_pilot_e2e.py` — the E2E harness; `coverage()` (line ~93) does
  `merged.update(out)` over streamed deltas, clobbering additive keys (the R6 bug).

### Institutional Learnings

- `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`
  — "Drop with a visible `degraded` flag, never silently"; stage non-SDK results before the `try`;
  emit a result on every exit path. Directly shapes R3.
- `docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md` — SDK subprocess
  fails intermittently; never swallow failures (anti-pattern = silent shrink / fabricated result).
- `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md` —
  a blanket merge over LangGraph state mishandles `operator.add` keys (the R6 root cause); mirror the
  reducer map when merging streamed deltas.
- `docs/solutions/best-practices/langgraph-pipeline-production-formalization.md` — ship new behavior
  behind config with `Field(description=...)` recording why each default exists; cassette replay does
  not remove LLM non-determinism. Notes issue #47 as the same silent-truncation family as R6.
- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` —
  run tests as `cd backend && uv run python -m pytest ...`.

### External References

- None used — internal pipeline code with strong local patterns.

## Key Technical Decisions

- **Aggregate, don't dump (module queries only).** For queries with ≥ `module_mode_min_entities`
  resolved entities, replace the per-entity `format_disease_associations` /
  `format_pathway_memberships` output in the synthesis context with cross-entity aggregation + a
  per-member table. Below that threshold (single-entity **and** small pair/triple queries) keep the
  existing per-entity sections verbatim (R5). Gate on resolved-entity count, not a config flag, so
  behavior matches intent automatically.
- **Two distinct thresholds, two config fields (do not conflate).** `module_mode_min_entities`
  (default >2, e.g. 5) gates *whether* assembly switches to module-aware mode; `min_members_for_recurrence`
  (default 2) gates *which* diseases/pathways qualify for the shared-recurrence lists. These pull in
  opposite directions (inclusive recurrence wants a low value; not-reshaping-small-queries wants a
  higher mode threshold), so a single field would couple two independent decisions — a latent bug.
- **Dedupe before counting members; keep strongest evidence.** Because
  `disease_associations`/`pathway_memberships` use additive reducers and may carry duplicates from
  multiple branches, aggregation must dedupe by `(entity_curie, disease_curie)` /
  `(entity_curie, pathway_curie)` before counting *distinct* member entities. When duplicate
  `(entity, key)` rows differ only in `evidence_type`, **retain the strongest** (don't silently drop
  evidence strength that the member table's "top disease" selection relies on). Otherwise recurrence
  counts inflate.
- **Caps default-ON with token-derived values (this fixes a live crash).** The bounds fix a
  deterministic over-window crash, so they ship enabled. `max_context_chars` defaults to a value
  **derived from the ~200K-token window** (≈350K chars ≈ ≤100K tokens at 3.5 chars/token, ~2.5× under
  the failing 882K — see Diagnostic Evidence), not a guessed-then-tuned number. R7 confirms; tune only
  downward, and re-validate before the full-217 scale.
- **`max_context_chars` is a char proxy for a token ceiling.** The real limit is tokens, not chars
  (3.80 chars/token measured for this CURIE-dense content). The cap is expressed in chars for cheap
  in-process checking, sized conservatively against the token budget; Unit 7 asserts the *token*
  estimate stays under budget, closing the proxy gap.
- **Visible degradation, not retry.** The primary fix (caps) prevents the deterministic overflow; the
  `except` is fixed to log + record in `state["errors"]` + set a degraded marker. This makes a future
  over-budget case **observable**, not impossible — "always produces a narrative" holds only once the
  R7 run proves the cap value. Bounded retry for transient SDK flakiness is deferred (see Scope).
- **Config over constants.** New caps live in a `SynthesisConfig` Pydantic sub-model (mirroring
  `BridgeGroundingConfig`), read via `get_pipeline_config()` newly imported into synthesis.py.
  (Module constants like `MAX_LIT_PAPERS_PER_HYPOTHESIS` would also satisfy R4; the sub-model is chosen
  for consistency with the established config pattern and runtime overridability.)
- **Harness reads reducers correctly.** Fix `coverage()` to mirror `state.py` reducer semantics
  (concatenate additive list keys across streamed deltas; overwrite plain keys) rather than a blanket
  `merged.update()`.

## Open Questions

### Resolved During Planning

- Scope of fix → module-aware redesign (user decision).
- Per-member data treatment → aggregate **and** keep a compact per-member table (user decision).
- Config vs constants → config sub-model, mirroring existing pattern (research-confirmed).
- Are source fields additive? → yes, all six (research-confirmed) → aggregation must dedupe.
- Root cause → **input-token-window overflow** (~230K tokens vs ~200K window), measured offline from
  the failure artifact (Diagnostic Evidence, 2026-06-22). Not transient flakiness.
- Which sections dominate → findings 58%, disease 21%, pathway 17% (96% combined) → the planned cuts
  are correctly targeted (measured, not assumed).
- Module-mode threshold vs recurrence threshold → **two separate config fields** (decoupled) —
  `module_mode_min_entities` (>2) and `min_members_for_recurrence` (2).
- Initial `max_context_chars` → **pre-committed at ≈350K chars** (token-derived), not deferred.
- Degraded handling → log + record in `state["errors"]` (+ marker); retry deferred (learnings).
- Caps enablement → **default-ON** (user sign-off 2026-06-22): fixes a live crash, not an
  experiment; only cap *values* tuned later post-measurement. No enable flag.
- SDK-failure handling → **visible-degraded only** (user sign-off 2026-06-22): log WARNING +
  record in `state["errors"]`; bounded SDK retry stays a separate reliability task (see Deferred).

### Deferred to Implementation

- `max_context_chars` initial value is **pre-committed at ≈350,000 chars** (token-derived; see
  Diagnostic Evidence). The remaining cap values (`max_findings_per_tier`, `max_aggregated_diseases`,
  `max_aggregated_pathways`, `module_mode_min_entities`, `min_members_for_recurrence`,
  `max_member_table_rows`) get conservative starting values; R7 confirms context stays under the token
  budget and the report is coherent. Re-validate at the full-217 scale.
- Exact helper/function names for the new aggregation + member-table functions.
- Whether to add a dedicated `synthesis_degraded: bool` state key vs recording only in `errors` —
  decide during implementation based on whether the frontend/report needs to display it (recording in
  `errors` satisfies R3 with no schema change; a bool is additive polish). The *baseline* return shape
  is already fixed by Unit 5's Approach — a `synthesis_degraded: <type>: <msg>` string appended to the
  node's returned `errors` list (merged into state via the additive reducer); the deferred decision is
  only whether to *additionally* surface a typed bool.
- The precise "top disease per member" selection rule for the member table (highest evidence tier,
  then by source) — settle when writing `format_member_table` against real data.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation
> specification. The implementing agent should treat it as context, not code to reproduce.*

Synthesis context assembly, before vs after, for a multi-entity (module) query:

```
BEFORE (unbounded):                      AFTER (bounded, module-aware):
  Entity Resolution Summary                Entity Resolution Summary
  Disease Associations   ── per entity     Module-Level Disease Recurrence  ── diseases shared by ≥N members
    ### CURIE: [all diseases] × 44         Module-Level Pathway Recurrence  ── pathways shared by ≥N members
  Pathway Memberships    ── per entity     Member Prioritization Table      ── 1 line/member: bucket|edges|top disease
    ### CURIE: [all pathways]  × 44        Shared Neighbors / Themes        (unchanged, already capped)
  Shared Neighbors / Themes                Cross-Type Bridges               (unchanged)
  Cross-Type Bridges                       Gap Analysis                     (unchanged)
  Gap Analysis                             Analysis Findings (top-K / tier) ── capped, "… and N more"
  Analysis Findings  ── ALL ~4,000 ✗       Literature Evidence              (unchanged)
  → 882KB → SDK crash → silent fallback    → bounded → SDK succeeds → module narrative

                                         [context-char backstop: if assembled > max_context_chars,
                                          log WARNING (caps should prevent reaching here)]
```

`run()` failure path, before vs after:

```
BEFORE: except Exception: report = fallback_report(state)   # silent, errors stays 0
AFTER:  except Exception as e:
            logger.warning("synthesis LLM failed, using fallback: %s", e, exc_info=True)
            report = fallback_report(state)
            return {..., "errors": [f"synthesis_degraded: {type(e).__name__}: {e}"]}  # additive → visible
```

## Implementation Units

- [ ] **Unit 1: `SynthesisConfig` + wire config into synthesis.py**

**Goal:** Introduce a configurable cap surface for synthesis and make the node read it.

**Requirements:** R4 (supports R1, R2).

**Dependencies:** None.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/pipeline_config.py`
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`
- Test: `backend/tests/test_pipeline_config.py` (extend), `backend/tests/test_synthesis_node.py` (new)

**Approach:**
- Add `class SynthesisConfig(BaseModel)` mirroring `BridgeGroundingConfig`: fields
  `max_findings_per_tier`, `max_aggregated_diseases`, `max_aggregated_pathways`,
  `module_mode_min_entities` (default >2, e.g. 5), `min_members_for_recurrence` (default 2),
  `max_member_table_rows`, `max_context_chars` (default **350_000**, token-derived — see Diagnostic
  Evidence), each `Field(default=…, ge=…, description="why")`. The two threshold fields are
  intentionally separate (see Key Technical Decisions).
- Register `synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)` on `PipelineConfig`.
- Import `get_pipeline_config` into synthesis.py; read `get_pipeline_config().synthesis` where caps
  are applied (Units 2–4). Do **not** add to `get_semaphore()`'s map (synthesis has no per-node
  semaphore).

**Patterns to follow:** `BridgeGroundingConfig` + its registration; config override idiom
`monkeypatch.setattr(synthesis, "get_pipeline_config", lambda: cfg)`.

**Test scenarios:**
- Happy path: `PipelineConfig().synthesis` exposes all cap fields with documented defaults.
- Edge case: field bounds reject invalid values (e.g. negative `max_findings_per_tier` raises
  `ValidationError` via `ge=`).
- Happy path: `get_pipeline_config().synthesis` is reachable and overridable in a node test via
  monkeypatch.

**Verification:** `PipelineConfig` carries a `synthesis` sub-config; synthesis.py imports and reads it;
config tests green.

- [ ] **Unit 2: Cross-entity aggregation helpers**

**Goal:** Compute module-level disease/pathway recurrence across members.

**Requirements:** R2 (supports R1).

**Dependencies:** Unit 1 (caps).

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`
- Test: `backend/tests/test_synthesis_node.py`

**Approach:**
- Add `aggregate_shared_diseases(disease_associations, min_members, max_items)` → diseases grouped by
  `disease_curie`, counting **distinct** `entity_curie` after deduping `(entity_curie, disease_curie)`
  (on duplicate `(entity, disease)` rows differing only in `evidence_type`, keep the strongest);
  keep those with ≥ `min_members` members (caller passes `min_members_for_recurrence`); rank by member
  count then evidence strength; cap to `max_items`; render markdown (disease name, member count, member
  list, strongest evidence type).
- Add `aggregate_shared_pathways(pathway_memberships, min_members, max_items)` — same over
  `pathway_curie`.
- Both return `""` when nothing qualifies (so single-entity queries emit nothing — R5).

**Execution note:** Implement test-first — pure functions over Pydantic models, ideal for TDD.

**Technical design:** *(directional)* group → dedupe `(entity, key)` → `Counter` over distinct
entities → filter `≥ min_members` → sort desc → slice `max_items` → format.

**Patterns to follow:** existing `format_*` helpers in synthesis.py (markdown section + `### ` shape);
`format_bridges` test style in `test_bridge_grounding_node.py`.

**Test scenarios:**
- Happy path: 3 entities sharing disease D → D listed with member count 3 and all three members.
- Edge case (dedupe): same `(entity_curie, disease_curie)` appearing twice (additive-reducer
  duplicate) counts the member **once**, not twice.
- Edge case (dedupe keeps strongest evidence): same `(entity_curie, disease_curie)` with two different
  `evidence_type` values → member counted once **and** the rendered "strongest evidence type" reflects
  the stronger of the two (evidence not silently dropped).
- Edge case (threshold): a disease in only 1 member with `min_members=2` is excluded.
- Edge case (cap): with `max_aggregated_diseases=N` and N+5 qualifying diseases, exactly N rendered,
  highest member-count retained.
- Edge case (empty): empty input → `""`; single-entity input → `""` (nothing shared by ≥2).
- Happy path (pathways): analogous distinct-member counting for `aggregate_shared_pathways`.

**Verification:** helpers produce correct distinct-member counts with dedupe, respect threshold + cap,
and return empty for single-entity input.

- [ ] **Unit 3: Per-member prioritization table**

**Goal:** A compact one-line-per-member table capturing the per-member axis.

**Requirements:** R2 (supports R5).

**Dependencies:** Unit 1.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`
- Test: `backend/tests/test_synthesis_node.py`

**Approach:**
- Add `format_member_table(resolved_entities, novelty_scores, disease_associations, max_rows)` →
  one row per resolved entity: `name (curie) | category | bucket | edge_count | top disease`. Join
  `novelty_scores` (edge_count + classification) with `resolved_entities` (name/category) by curie;
  "top disease" = the entity's highest-evidence-tier `DiseaseAssociation` (or "—"). Sort by
  `edge_count` desc. Each row ≈50 chars, but at the 217-analyte target a full table is ~217 rows (~11KB)
  and itself becomes a dump — so cap to `max_member_table_rows` (top-N by edge_count) with a
  "… and N more members" elision line. At 48 analytes this is a no-op; it protects the headline scale.

**Execution note:** Test-first (pure function).

**Patterns to follow:** synthesis `format_*` helpers; markdown table rendering.

**Test scenarios:**
- Happy path: 3 members with differing edge counts → 3 rows sorted by edge_count desc, each showing
  bucket + edge_count.
- Edge case: a member with no disease associations → "top disease" shows "—".
- Edge case: a `novelty_score` curie with no matching `resolved_entity` (and vice versa) → row still
  renders without KeyError (graceful join).
- Edge case: empty resolved set → `""`.
- Edge case (cap): with `max_member_table_rows=N` and N+5 members → exactly N rows (highest edge_count)
  + a "… and N more members" line.

**Verification:** table renders one bounded row per member with correct bucket/edges/top-disease and
graceful joins, and never exceeds `max_member_table_rows`.

- [ ] **Unit 4: Cap findings + rewire module-aware assembly + prompt framing**

**Goal:** Bound the findings section and assemble the module-aware context (and fallback report),
plus orient the prompt.

**Requirements:** R1, R2, R5 (supports R7).

**Dependencies:** Units 1–3.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`
- Test: `backend/tests/test_synthesis_node.py`

**Approach:**
- Cap `format_findings_summary` to `max_findings_per_tier` per tier, appending "… and N more (tier T)"
  elision lines. **Note:** the current helper groups by `Finding.tier` (1/2/3) and sorts by tier only;
  it does **not** rank within a tier. This unit adds within-tier ranking by `Finding.confidence`
  (map `high`→0, `moderate`→1, `low`→2 as the sort key) so the cap keeps the strongest findings, not
  an arbitrary slice. (Findings are the 58% dominant section per Diagnostic Evidence, so this is the
  load-bearing cut.)
- In `assemble_synthesis_context` **and** `fallback_report`: when resolved-entity count
  ≥ `module_mode_min_entities` (the module-mode switch — NOT the recurrence threshold), emit
  `aggregate_shared_diseases` + `aggregate_shared_pathways` (each passed `min_members_for_recurrence`)
  + `format_member_table` **in place of** the per-entity `format_disease_associations` /
  `format_pathway_memberships`; otherwise keep the existing per-entity sections (R5). Always apply the
  capped findings section. Keep shared-neighbors, themes, bridges, gaps, literature sections unchanged.
- Add a **context-char backstop**: after assembling, if `len(context) > max_context_chars`, log a
  WARNING (the caps should prevent reaching this — it is a tripwire, not a truncator). Because the real
  limit is tokens, the backstop should also log an **estimated token count** (`len(context) / 3.5`) so
  the warning is interpretable against the ~200K window.
- `SYNTHESIS_PROMPT`: add a conditional instruction — *if the input is a module (≥ module_mode_min_entities
  entities), treat them as a group/module; lead with the unifying theme; use the Module-Level Recurrence
  + Member Prioritization sections.* Keep single/small-query behavior intact.

**Patterns to follow:** existing section-assembly order in `assemble_synthesis_context` (711) and
`fallback_report` (810); the `[:N]` slice + "… and N more" elision idiom already in the file.

**Test scenarios:**
- Happy path (cap): 200 findings in tier 1 with `max_findings_per_tier=20` → 20 rendered + an
  elision line stating the remainder.
- Happy path (module assembly): a state with 44 well-char entities and shared diseases →
  `assemble_synthesis_context` contains the aggregation + member-table sections and **omits** the
  per-entity `### CURIE` disease/pathway dumps.
- Edge case (single entity): 1 resolved entity → per-entity sections retained, no aggregation
  sections, prompt-agnostic (R5).
- Edge case (2-entity boundary, R5): 2 resolved entities with `module_mode_min_entities=5` → per-entity
  sections **retained** (small pair query unaffected); separately, 2 entities with
  `module_mode_min_entities=2` → module-aware sections engage. Pins the decoupled-threshold behavior.
- Bound test: `assemble_synthesis_context` and `fallback_report` with 50 well-char entities ×
  inflated disease/pathway/findings stay **under `max_context_chars`** AND under the token budget
  (`len/3.5 < ~100K`) — the core R1 guard (note the unit char bound is necessary but only R7 proves
  real SDK success).
- Edge case (backstop): force an over-budget context (tiny `max_context_chars`) → WARNING logged with
  the estimated token count.

**Verification:** module-scale assembled context is bounded and module-shaped; small queries unchanged.

- [ ] **Unit 5: Make synthesis fallback visible (no silent degradation)**

**Goal:** Fix the silent `except`; surface SDK-synthesis failures.

**Requirements:** R3.

**Dependencies:** None (independent of 1–4, but verified together).

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`
- Test: `backend/tests/test_synthesis_node.py`

**Approach:**
- In `run()` (line ~1010): on SDK exception, `logger.warning(... exc_info=True)`, build the fallback
  report, and **append a `synthesis_degraded: <type>: <msg>` string to the returned `errors`** (the
  `errors` reducer is additive, so it surfaces in final state and coverage). Same for the
  `not HAS_SDK` and empty-text paths (the latter is expected in tests — log at INFO/DEBUG, not an
  error, to avoid false positives). Optionally set a `synthesis_degraded` marker (deferred decision).

**Execution note:** Test-first — assert on the returned dict and captured logs.

**Patterns to follow:** other nodes' error recording into `state["errors"]`; `caplog` usage in
existing tests.

**Test scenarios:**
- Error path: `HAS_SDK=True`, `query_with_usage` raises → returns the fallback report **and**
  `errors` contains a `synthesis_degraded:` entry; WARNING logged.
- Edge case: `HAS_SDK=False` → fallback report returned; this is the test/no-SDK path and must **not**
  be recorded as a degradation error (avoid false positives in the suite).
- Edge case: SDK returns empty text → fallback used, logged at non-error level.

**Verification:** an SDK synthesis crash is logged and recorded in `errors`; the no-SDK path stays
clean.

- [ ] **Unit 6: Fix assessment harness `coverage()` under-count (reducer-aware merge)**

**Goal:** Accurate finding/association counts so the fix is verifiable.

**Requirements:** R6 (supports R7).

**Dependencies:** None.

**Files:**
- Modify: `backend/assessment_data/brown_c1_pilot_e2e.py`
- Test: none (scratch assessment harness) — see Test expectation note.

**Approach:**
- In the stream-consume loop, stop using a blanket `merged.update(out)` for additive list keys.
  Mirror `state.py` reducer semantics: for **every** `operator.add` list key — `resolved_entities`,
  `novelty_scores`, `direct_findings`, `cold_start_findings`, `disease_associations`,
  `pathway_memberships`, `inferred_associations`, `analogues_found`, `hub_flags`, `shared_neighbors`,
  `gap_entities`, `temporal_classifications`, `model_usages`, `literature_errors`,
  `bridge_grounding_errors`, `errors` — **concatenate** across deltas; for plain last-write-wins keys
  (`bridges`, `hypotheses`, `biological_themes`, `synthesis_report`, etc.), overwrite. (Prefer
  deriving the additive-key set programmatically from `DiscoveryState`'s annotations so a future
  reducer field cannot silently reintroduce the under-count.) Then `coverage()` counts reflect reality
  (the run that reported 37 findings under-counted findings/associations — see the "measure actual
  per-section contribution" item below for the real magnitude).

**Test expectation:** none — this is scratch assessment tooling, not shipped library code; correctness
is validated by the R7 run reporting plausible counts. (If kept long-term, a small unit test over a
synthetic delta stream would be worthwhile — noted, not required.)

**Verification:** re-running the harness reports finding/association counts consistent with the
rendered report (no ~100× under-count).

- [ ] **Unit 7: End-to-end verification — re-run the 48-analyte Brown pass**

**Goal:** Prove the fix at the scale that broke it.

**Requirements:** R7 (closes R1, R2).

**Dependencies:** Units 1–6.

**Files:**
- Run: `backend/assessment_data/brown_c1_pilot_e2e.py` (`--n-proteins 24 --n-metabolites 24
  --ceiling-min 45`); artifact auto-saved to `backend/assessment_data/brown_diagnostic_runs/`.

**Approach:**
- Re-run, then assert **machine-checkable** acceptance (not only a human read) — add these checks to
  the harness so pass/fail is reproducible despite LLM non-determinism:
  1. `synthesis_degraded` **absent** from `errors` (no fallback path taken).
  2. `synthesis_duration > 30s` (a real LLM call happened, not the ~3.5s deterministic fallback).
  3. assembled-context **estimated tokens** (`chars/3.5`) **< budget** (~100K) and chars < `max_context_chars`.
  4. `synthesis_report` is human-sized (regex: **< 10** `### CURIE`-style raw per-entity sections;
     total size ~tens of KB, not ~882KB).
  5. `coverage()` counts are now plausible (findings ≫ the old 37; consistent with the rendered report).
- **Signal-preservation check (guards information loss):** diff the 48-analyte narrative against the
  known-good 24-analyte pilot narrative — confirm no high-value finding present at 24 disappeared at
  48 purely due to aggregation. This is a **human/domain** gate on top of the machine assertions; name
  who signs off (the researcher/owner). A bounded, crash-free-but-degraded report must not pass
  silently.

**Test expectation:** the machine assertions above ARE the automated acceptance; the signal-preservation
diff is a human gate. (Paid ~12-min biomapper-ON SDK run; auto-saves its artifact.)

**Verification:** R1, R2, R7 satisfied on the real module-scale input — proven by the machine assertions
*and* the signal-preservation sign-off, not "looks like a narrative" alone.

## System-Wide Impact

- **Interaction graph:** synthesis is the terminal node; changes affect only the final report and the
  `errors` channel. No upstream node is touched. The `errors` additions are read by coverage and any
  monitoring of `state["errors"]`.
- **Error propagation:** SDK synthesis failure now propagates as a recorded `errors` entry + log,
  instead of vanishing. Downstream consumers that treat `errors == []` as "fully clean" will now
  correctly see synthesis degradations.
- **State lifecycle risks:** none new — no schema changes required (recording in existing additive
  `errors`); a `synthesis_degraded` bool, if added, is additive and backward-compatible.
- **API surface parity:** the WebSocket/report payload (`synthesis_report`) shape is unchanged; only
  its *content* at module scale changes (narrative vs raw dump). Frontend rendering already handles
  arbitrary markdown.
- **Integration coverage:** the bound test (Unit 4) and the E2E run (Unit 7) together prove what unit
  mocks cannot — that the real assembled context stays under the SDK's effective limit.
- **Unchanged invariants:** single-entity / small-query report shape (R5); reducer semantics in
  `state.py`; all non-synthesis nodes; the literature/bridges/themes/gaps sections.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Caps too tight → narrative loses signal; **or** aggregation drops a high-value single-member finding | Aggregation preserves cross-entity signal; member table preserves per-member coverage; the Unit 7 signal-preservation diff vs the 24-analyte pilot is the explicit guard, with a named human sign-off. A 1-member strong link still surfaces via the member table's "top disease". |
| `max_context_chars` set above the real **token** ceiling → still over-window | Initial value is **token-derived** (≈350K chars ≈ ≤100K tokens, ~2.5× under the failing 882K/230K); Unit 4 asserts both char and estimated-token bounds; R7 asserts est-tokens < budget on real data. The true gate is R7, not the unit char bound. |
| Caps slightly loose → first module run re-crashes | Conservative token-derived default + the visible-degraded path makes it observable (recorded in `errors`), not silent; R7 is engineered to pass, not hoped to. |
| Aggregation double-counts, or drops evidence strength, due to additive-reducer duplicates | Explicit dedupe by `(entity, key)` before counting distinct members, retaining the strongest `evidence_type` (Unit 2 tests cover both). |
| Recording every empty-text fallback as an error → false positives in test suite | Only the genuine SDK-exception path records `synthesis_degraded`; no-SDK / empty-text paths log at non-error level (Unit 5 tests cover both). |
| Harness `coverage()` fix incomplete (misses a reducer key) | Mirror the exact additive-key list from `state.py` (enumerated in this plan); R7 sanity-checks counts against the rendered report. |

## Documentation / Operational Notes

- After R7 passes, document the module-scale synthesis pattern via `ce:compound` (the learnings
  researcher flagged this as net-new territory worth capturing).
- Update the origin requirements doc's Phase B section with the synthesis-overflow resolution and the
  chosen cap defaults.
- No deployment/rollout concerns: caps are default-on bug fixes; no migration; no env vars.

## Sources & References

- **Origin document:** [docs/brainstorms/2026-06-18-run-brown-module-through-pipeline-requirements.md](docs/brainstorms/2026-06-18-run-brown-module-through-pipeline-requirements.md)
- Run artifact (the failure): `backend/assessment_data/brown_diagnostic_runs/brown_c1_pilot_20260622T070237Z.json`
- Related code: `backend/src/kestrel_backend/graph/nodes/synthesis.py`,
  `backend/src/kestrel_backend/graph/pipeline_config.py`, `backend/src/kestrel_backend/graph/state.py`
- Related learnings: `docs/solutions/best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`,
  `docs/solutions/best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md`,
  `docs/solutions/best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md`
- Memory: `brown-c1-48-run-synthesis-overflow-wall`
