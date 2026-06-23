---
title: "feat: context-management insight section in the performance report"
type: feat
status: active
date: 2026-06-22
deepened: 2026-06-22
---

# feat: context-management insight section in the performance report

## Overview

Add a `## Context management` section to the discovery pipeline's per-node performance report that shows **how the synthesis node compresses massive multi-analyte results into a bounded LLM context**. The pipeline integrates thousands of findings plus many diseases, pathways, and members into a single capped context (PR #85's module-aware aggregation); today that compression is invisible — the only trace is "… and N more" text inside the report the model sees, and a `WARNING` log if the budget tripwire fires. This feature surfaces the compression as structured telemetry in every run's performance report.

It is also the **empirical instrument for the 217-analyte scaling question**: at module scale the report will show, e.g., "member table: 50 / 217 (167 elided), findings: 150 / 20,400, context: 35% of budget" — turning "are the PR #85 caps tuned right for 217?" from a guess into a measured observation on each run.

**Disk-only v1**: the stats land in the performance report `.json` and `.md` artifacts. No WebSocket event or frontend rendering (consistent with the perf report's deferred-WS posture, PR #84).

## Problem Frame

The synthesis context is bounded near-constant regardless of analyte count by fixed `SynthesisConfig` caps (PR #85): `max_findings_per_tier` (≤50 per tier, ≤150 total across tiers), `max_aggregated_diseases`/`max_aggregated_pathways` (≤30 each, cross-member), `max_member_table_rows` (≤50), and a `max_context_chars` backstop. This keeps the 200K-token window safe (the 24-analyte run used only ~27% of the char budget). The trade is **signal compression**, and that compression is opaque to operators: a 24-analyte run already elides ~2,283 of 2,433 findings (2,433 − 150), and a 217-analyte run would elide ~167 of 217 members from the prioritization table. Operators tuning the caps (the config comments explicitly target the "217-analyte" scale and say "tune against the R7 run") have no per-run readout of how much was shown vs. dropped, or how close the context came to its budget.

## Requirements Trace

- **R1.** The synthesis node emits a `synthesis_context_stats` structure into pipeline state capturing, per run: assembled-context size (chars + estimated tokens) against `max_context_chars` and the ~200K-token window (utilization %); module-aware mode on/off and the `module_mode_min_entities` threshold + distinct-entity count; shown/total/elided counts for the **capped** sections that are active (findings always; aggregated diseases, aggregated pathways, and the member table when module mode is on); and a literature-grounding readout (hypotheses with literature attached / total hypotheses) as a **separate descriptive metric**, not a compression row.
- **R2.** Counts that depend on an internal filter (the disease/pathway "qualifying" count after `min_members_for_recurrence`) are derived from a **single shared helper** used by both the aggregator and the stats path, so the reported total cannot drift from the actual cut. Counts that are pure functions of inputs already in scope at the assembly site (findings, member table, literature) are computed in place from those same inputs and cap constants.
- **R3.** `performance_report.build_report` reads `synthesis_context_stats` from state and `render_markdown` renders a `## Context management` section: a register-voice prose caveat plus a compression table (only the active capped sections) and a budget line. The section adapts to mode (per-entity runs show findings + a note that disease/pathway sections were uncapped) and degrades gracefully (omitted / "not available") when the stats are absent.
- **R4.** The stats survive the WebSocket `stream_mode="updates"` accumulator path as a single-writer last-write-wins **dict** field (no reducer; it falls through `main.py`'s `else` branch, not the list-concat path), and the terminal `reporting` node reads it from true graph state.
- **R5.** The change adds no new failure surface: stats computation is best-effort (wrapped) and a failure leaves the field absent without failing synthesis; report building is already fail-safe. No public formatter signature changes, so no existing caller (tests, `_disease_pathway_sections`, `fallback_report`, the Brown harness) breaks.

## Scope Boundaries

- No WebSocket event and no frontend rendering of the stats (disk artifact only) — deferred with the rest of perf-report v2.
- Not changing any `SynthesisConfig` cap value or the aggregation/compression behavior — this feature only *measures* it.
- Not changing the public signatures of the synthesis formatters (`format_findings_summary`, `aggregate_shared_diseases`, `aggregate_shared_pathways`, `format_member_table`, `format_literature_evidence`) — avoids breaking their direct test/harness callers.
- Not instrumenting per-hypothesis paper truncation (`MAX_LIT_PAPERS_PER_HYPOTHESIS`) in v1 — the literature readout is hypothesis-grounding rate only (noted as a future metric).
- Not instrumenting the classic chatbot (`agent.py`) — it has no assembled-context compression step.

### Deferred to Separate Tasks

- Live WebSocket surfacing of the compression stats in the chat UI: future perf-report v2 (touches `protocol.py` + frontend).
- Acting on what the stats reveal at 217 (re-tuning caps): a separate task once a 217-scale run produces the numbers.
- Per-hypothesis paper-truncation metric for the literature section: future enhancement.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/nodes/synthesis.py`:
  - `assemble_synthesis_context(state)` (line ~933) builds the context, holds the raw lists (`direct_findings`, `cold_start_findings`, `disease_associations`, `pathway_memberships`, `resolved`, `hypotheses`) and `cfg = get_pipeline_config().synthesis`, and logs the `max_context_chars` tripwire WARNING at the end (line ~1030). It is called **unconditionally at line 1217** in `run()`, before the `if HAS_SDK:` branch — so stats are available on every path, including the `fallback_report` paths (lines 1246/1258/1266) and the no-SDK path.
  - Capped sections: `format_findings_summary` (line 547; per-tier slice + `elided`), `aggregate_shared_diseases`/`aggregate_shared_pathways` (lines 298/346; build a distinct-key dict, filter by `min_members_for_recurrence` to `qualifying`, then slice to `max_aggregated_*`), `format_member_table` (line 383; slice to `max_member_table_rows`, `hidden` count), all reached via `_disease_pathway_sections` (line ~893) in **module mode only**.
  - Per-entity mode (< `module_mode_min_entities`): `_disease_pathway_sections` instead calls **uncapped** `format_disease_associations` (line 219) / `format_pathway_memberships` (line 268), and emits no member table.
  - `format_literature_evidence` (line 828): splits `hypotheses` into `grounded` (have `literature_support`) vs `ungrounded`; no hypothesis-level elision (per-hypothesis paper cap is the real truncation).
  - `run()` (line 1189, `@validate_state(SynthesisInput, SynthesisOutput)`) returns `{"synthesis_report": report, "model_usages": [...], ...}` at line ~1289 — the emit point for the new field.
- `backend/src/kestrel_backend/graph/state_contracts.py`: `validate_state` **returns the original `result` dict** (line 394) and only *checks* the output model; `_ContractBase` uses `extra="ignore"` (line 48). **Verified: adding `synthesis_context_stats` to the return dict passes through untouched — no contract change required.** (Optionally declare it first-class on `SynthesisOutput` for visibility, mirroring `literature_errors`; not required for correctness.)
- `backend/src/kestrel_backend/graph/performance_report.py` — `build_report(state, meta)` receives full state; `render_markdown(report)` emits GFM with a register-voice caveat (the section to extend). Build is fail-safe (wrapped in `reporting.py`).
- `backend/src/kestrel_backend/graph/state.py` — `DiscoveryState`; `report_path: str` is an existing plain (no-reducer) field to mirror for `synthesis_context_stats: dict`.
- `backend/src/kestrel_backend/main.py` — the `stream_mode="updates"` accumulator: `_get_concat_fields()` matches only `Annotated[..., operator.add]`; a plain `dict` field falls through to the `else: accumulated_state[key] = value` branch (last-write-wins). **Verified: no `main.py` change needed.**
- `backend/src/kestrel_backend/graph/nodes/reporting.py` — terminal node; reads true merged state, calls `build_report`. No change beyond inheriting the field.
- `backend/src/kestrel_backend/writing_style.py` — `RESEARCH_REGISTER`; the new prose caveat follows it (renderer author matches it, as for the existing caveats).

### Institutional Learnings

- The LangGraph `stream_mode="updates"` reducer-replication learning (single-writer plain fields are correctly last-write-wins in `main.py`'s accumulator; only `operator.add`/custom-reducer fields need replication) — applies directly: `synthesis_context_stats` is single-writer, so no reducer and no `main.py` change. **Note: that solution doc currently lives on the unmerged `docs/langgraph-updates-reducer-learning` branch, not on this branch — the principle is applied here and verified against `main.py` directly.**
- `docs/solutions/performance-issues/synthesis-context-window-overflow-silent-fallback-2026-06-22.md` (present on this branch) — the overflow incident this feature makes visible; basis for the char/token proxy (~3.5–3.8 chars/token) the size readout uses.
- PR #84 perf-report learning (auto memory) — the perf report is an extraction job over existing state; `build_report` is fail-safe, disk-only v1. The "U7 gate matched a marker string that never occurs" / "wrong-count gate" lesson motivates R2 (single-source the filter count; assert against a hand-computable fixture).

### External References

- None. Entirely in-repo, pattern-following (perf-report build/render, register prose, single-writer state fields).

## Key Technical Decisions

- **Out-parameter, not signature changes (R5).** `assemble_synthesis_context(state, stats_out: dict | None = None)` gains an optional mutable out-dict (default `None` → fully backward compatible: the Brown harness's `len(assemble_synthesis_context(merged))` and all current callers/tests are untouched). When `run()` passes a dict, assembly fills it in place. The five public formatters keep their `str` returns. This replaces the rejected "thread `(text, shown, total)` out of every formatter" approach, which would have broken `_disease_pathway_sections`, `fallback_report`, ~6 test call sites, and the harness.
- **Single-source the one internal count (R2).** The disease/pathway "qualifying" count (entries surviving `min_members_for_recurrence`) is computed inside `aggregate_shared_diseases`/`aggregate_shared_pathways`. Extract that filter into a tiny shared helper (e.g. `recurrence_qualifying(associations, min_members)`), call it from both the aggregators (internal change, no public signature change) and the stats path. This guarantees the reported "total" equals the actual cut. Findings, member-table, and literature counts are pure functions of inputs already in `assemble_synthesis_context`'s scope and the cap constants, so they are computed in place (same inputs, adjacent to the call → no drift).
- **Stats available on all paths.** Because `assemble_synthesis_context` runs unconditionally (line 1217), populate `stats_out` there; stats exist even on degraded/fallback runs. The field is absent only if the best-effort computation itself raises.
- **Best-effort, fail-safe (R5).** Wrap the stats population so any error leaves `synthesis_context_stats` absent and never fails synthesis; `build_report`/`render_markdown` treat absence as "section omitted / not available."
- **Mode-adaptive shape.** `module_mode` boolean drives which compression rows exist: module mode → findings + diseases + pathways + member table; per-entity mode → findings only (disease/pathway are uncapped in that path, so they are reported as "uncapped (per-entity)" rather than a misleading `shown==total, elided==0` row). Literature readout always present.
- **Literature is a grounding readout, not a compression row.** Report "hypotheses with literature attached / total hypotheses" as a labeled line, not in the shown/total/elided table (there is no hypothesis-level elision; the real truncation is per-hypothesis papers, deferred). Avoids the category error of mixing grounding-rate with capping-compression.
- **Drop a redundant `tripwire_fired` bool** — "% of `max_context_chars`" already shows it (>100% means it fired). Keep chars, est tokens, and the two utilization percentages.
- **No state-contract change** — verified `validate_state` returns the original dict and `extra="ignore"` accepts the new key.

## Open Questions

### Resolved During Planning

- Re-derive counts or capture at the cap? Out-param + one shared helper for the only genuinely-internal count (disease/pathway qualifying); compute the rest in place (R2). No formatter signature changes.
- Reducer / `main.py` change? No — single-writer plain dict, last-write-wins; verified against `main.py`. Contract change? No — verified `validate_state` passes the dict through.
- Stats on fallback runs? Yes — `assemble_synthesis_context` runs unconditionally, so stats exist on every path; absent only if computation raises.
- Literature metric? Grounding readout (attached/total hypotheses), separate from the compression table.
- WS/live vs disk? Disk-only v1 (user decision). Behavior change? None — measurement only.

### Deferred to Implementation

- Exact key layout of the stats dict (flat vs. nested `sections` map) — pick what renders cleanly; keep JSON-serializable primitives only.
- Whether findings are reported per-tier or aggregated — `format_findings_summary` computes per-tier `elided`; aggregate is the default, per-tier a stretch.
- Token-estimate divisor surfaced — reuse the exact value the `assemble_synthesis_context` tripwire log already uses (≈3.5), for consistency.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
synthesis.run(state)                                  # line 1189; @validate_state passes extra keys through
  └─ stats = {}                                       # best-effort wrapper around the next call
     context = assemble_synthesis_context(state, stats_out=stats)   # line 1217, ALWAYS runs
          ├─ findings:   shown/total/elided from direct+cold lists & cfg.max_findings_per_tier (in place)
          ├─ module mode? (distinct_entities >= cfg.module_mode_min_entities)
          │     ├─ diseases/pathways: total = recurrence_qualifying(assoc, min_members)  ← shared helper
          │     │                     shown = min(total, cfg.max_aggregated_*)
          │     └─ member table: shown=min(N, cfg.max_member_table_rows), total=N
          │   else (per-entity): diseases/pathways uncapped → mark "uncapped", no member table
          ├─ literature: attached = #hypotheses with literature_support; total = #hypotheses
          └─ size: chars=len(context), est_tokens≈chars/3.5, %max_context_chars, %200K window
  └─ return { "synthesis_report": report, ..., "synthesis_context_stats": stats }   # plain dict, last-write-wins

reporting.run(state) → build_report(state, meta)      # reads state["synthesis_context_stats"] from true state
                       render_markdown(report)          # adds "## Context management" (register caveat + table + literature line)
```

## Implementation Units

- [ ] **Unit 1: Synthesis emits `synthesis_context_stats` (out-param + shared count helper)**

**Goal:** The synthesis node computes compression/budget telemetry without changing any public formatter signature, and emits it as a single-writer state field.

**Requirements:** R1, R2, R4, R5

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py` (add `synthesis_context_stats: dict` plain field, mirroring `report_path`)
- Modify: `backend/src/kestrel_backend/graph/nodes/synthesis.py`:
  - extract `recurrence_qualifying(associations, min_members)` helper; call it inside `aggregate_shared_diseases`/`aggregate_shared_pathways` (internal — no signature change) and from the stats path
  - add optional `stats_out: dict | None = None` to `assemble_synthesis_context`; populate it in place (findings/member-table/literature counts from in-scope vars + caps; disease/pathway totals via the helper; size/budget from `len(context)` and `cfg`; `module_mode` flag)
  - in `run()`, wrap the `assemble_synthesis_context(..., stats_out=stats)` call best-effort and add `synthesis_context_stats` to the returned dict
- Test: `backend/tests/test_synthesis_context_stats.py`

**Approach:**
- The only genuinely-internal count is the disease/pathway qualifying total; the shared helper guarantees the stats total equals the aggregator's cut (R2). Everything else is computed from variables already in `assemble_synthesis_context` and the `cfg` cap constants.
- Default `stats_out=None` keeps every existing caller (tests, `_disease_pathway_sections`, `fallback_report`, the Brown harness `len(assemble_synthesis_context(merged))`) working unchanged.
- Best-effort: a failure populating `stats_out` leaves `synthesis_context_stats` absent; synthesis still returns its report (R5).

**Execution note:** Test-first for the helper and the stats math — assert exact counts against a hand-computable fixture before wiring into `run()`. No characterization snapshot of the formatters is needed because their signatures and output are unchanged.

**Patterns to follow:** `get_pipeline_config().synthesis` access; the per-tier `elided` computation already in `format_findings_summary`; the plain `report_path` field in `state.py`; the unconditional `assemble_synthesis_context` call at line 1217.

**Test scenarios:**
- Happy path (R2), module mode: fixture with 200 findings (cap 150), 40 diseases qualifying after `min_members_for_recurrence` (cap 30), 217 members (cap 50) → findings shown=150/total=200/elided=50; diseases shown=30/total=40/elided=10; member_table shown=50/total=217/elided=167. (Synthetic cap-boundary fixture — distinct from the Brown-harness validation, which has fewer members than the caps and shows no elision.)
- Edge case (R2): the disease/pathway `total` is the post-`min_members_for_recurrence` qualifying count, not raw `len(disease_associations)` and not the pre-filter distinct count — a fixture with single-member diseases confirms they are excluded from `total`; the count comes from the shared helper (assert the helper and the aggregator agree).
- Edge case (mode): below `module_mode_min_entities` → `module_mode=False`, findings stats present, disease/pathway marked uncapped, member-table absent; no crash.
- Edge case: empty/degenerate state (no findings/diseases/hypotheses) → zeros / empty, no crash.
- Error path (R5): force the stats population to raise → synthesis still returns its report and `synthesis_context_stats` is absent.
- Integration (R4): drive the returned dict through the schema-derived concat-field detection / accumulator path `main.py` uses and assert `synthesis_context_stats` is preserved last-write-wins (falls through the `else` branch, not list-concat).
- Backward-compat: `assemble_synthesis_context(state)` (no `stats_out`) still returns the identical context string (call it once with and once without the out-param on the same state; assert string equality).

**Verification:** a fixture/real synthesis run populates `synthesis_context_stats` with counts that match the actual capped output; existing synthesis tests (`test_synthesis_node.py`, `test_synthesis_register.py`, `test_synthesis_fallback_marker.py`, `test_langgraph_prototype.py` synthesis cases) pass unchanged; `assemble_synthesis_context`'s string output is byte-identical with and without `stats_out`.

- [ ] **Unit 2: Performance report renders `## Context management`**

**Goal:** The performance report surfaces the telemetry as a readable, mode-adaptive section, degrading gracefully when absent.

**Requirements:** R3, R5

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/src/kestrel_backend/graph/performance_report.py` (`build_report` copies `state.get("synthesis_context_stats")` into the report dict; `render_markdown` adds the section)
- Test: `backend/tests/test_performance_report.py` (extend existing)

**Approach:**
- `render_markdown` adds `## Context management`: a one-line register-voice caveat (elision is expected at module scale and is the compression that keeps the assembled context within the token window); a compression table (Section | Shown | Total | Elided) listing only the active capped sections; a budget line (context chars / est tokens / % of `max_context_chars` / % of 200K window); and a literature line ("hypotheses with literature attached: A / T"). In per-entity mode, omit the disease/pathway/member rows and note they were uncapped.
- Absent `synthesis_context_stats` → omit the section or render "Context management: not available for this run"; never crash (R5).
- Tables/labels literal; only the caveat is register prose; the JSON report retains the full stats structure.

**Patterns to follow:** the existing `render_markdown` structure and its register-voice `wall_clock_source` caveat; the disk-only JSON+MD dual output.

**Test scenarios:**
- Happy path (module mode): a report dict with module-mode stats renders the `## Context management` heading, the compression table with findings/diseases/pathways/member rows, the budget line, and the literature line; JSON retains the full structure.
- Edge case (per-entity mode): stats with `module_mode=False` render findings + the "uncapped (per-entity)" note, no disease/pathway/member rows.
- Edge case (R3/R5): `synthesis_context_stats` absent → no table (or explicit "not available"); render does not crash; the rest of the report is intact.
- Edge case (register): the caveat prose contains no em-dash and the heading contains no dash; affirmative assertion that the caveat wording is present (not just dash-absence).
- Happy path (regression): existing perf-report assertions still hold — per-node table intact, `run_id` shown, raw query never emitted.

**Verification:** rendering a report with module-mode stats shows an accurate register-voice section; per-entity and absent cases degrade correctly; all existing `test_performance_report.py` tests pass; a real validation run's `.md`/`.json` contain the section.

## System-Wide Impact

- **Interaction graph:** synthesis (fills `stats_out`) → state → terminal `reporting` node → `build_report`/`render_markdown`. No graph topology/routing change; `reporting.py` inherits the field.
- **Error propagation:** stats population is best-effort in synthesis (R5); report build is fail-safe; absence degrades to an omitted section.
- **State lifecycle risks:** `synthesis_context_stats` is single-writer → last-write-wins is correct under the LangGraph updates accumulator (`else` branch); no reducer, no merge risk. Locked by the R4 test.
- **API surface parity:** WS-streamed and terminal-node report paths both read from state; the terminal node reads true merged state, so it always sees the field. Classic chatbot has no compression step (out of scope).
- **Integration coverage:** the R4 accumulator test and a real validation run (section present in the on-disk artifact) cover the cross-layer path that a `build_report` unit test alone would not.
- **Unchanged invariants:** no `SynthesisConfig` cap changes; the assembled context *text* is unchanged (string output byte-identical with/without `stats_out`); no public formatter signatures change; no state-contract change; the perf-report schema gains one field but existing fields, the per-node table, `run_id`/query handling, and timing/cost metrics are untouched.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Disease/pathway "total" drifts from the actual cut | Single shared `recurrence_qualifying` helper used by both the aggregator and the stats path (R2); test asserts helper and aggregator agree |
| Out-param accidentally changes `assemble_synthesis_context`'s string output | Default `stats_out=None`; a test asserts the context string is byte-identical with and without the out-param |
| Stats computation throws and takes synthesis down | Best-effort wrapper (R5): failure leaves the field absent, synthesis still returns its report |
| Field silently dropped on the WS accumulation path | R4 test drives the dict through the accumulator and asserts last-write-wins (`else` branch, not list-concat) |
| Per-entity mode reports misleading `shown==total` for uncapped sections | Mode-adaptive shape: per-entity omits disease/pathway/member rows and notes they were uncapped |
| Literature row misread as compression | Reported as a separate grounding readout (attached/total hypotheses), not in the shown/total/elided table |
| Section noise on small queries | Tabular + one caveat line; small module-mode queries simply show `shown==total` (no elision), which is itself informative |

## Documentation / Operational Notes

- No env, deploy, schema-migration, or config changes. The `.json` gains a `synthesis_context_stats` object; the `.md` gains a `## Context management` section.
- Validate on the existing 24-analyte Brown harness run (`backend/assessment_data/brown_c1_pilot_e2e.py`): confirm the section reports numbers consistent with that run — findings 150 / 2,433; member table 24 / 24 (no elision, since 24 < the 50-row cap); context ~94,225 chars / ~27% of `max_context_chars`. Same artifact-on-disk validation used for the register feature. (A true 217-scale run is the eventual gate for the cap-fidelity question, separate from landing this feature.)
- Riding in the same PR as the `writing_style.py` em-dash-polish commit already on branch `feat/perf-report-context-insight`.
- Testing: `cd backend && uv run python -m pytest tests/test_synthesis_context_stats.py tests/test_performance_report.py tests/test_synthesis_node.py -v` (module invocation; validate individually given the ~16–18 pre-existing full-suite failures).

## Sources & References

- Related code: `backend/src/kestrel_backend/graph/nodes/synthesis.py` (`assemble_synthesis_context` line 933, `aggregate_shared_diseases` 298 / `aggregate_shared_pathways` 346, `format_member_table` 383, `format_findings_summary` 547, `format_literature_evidence` 828, `_disease_pathway_sections` 893, `run` 1189 / assemble call 1217), `backend/src/kestrel_backend/graph/performance_report.py`, `backend/src/kestrel_backend/graph/state.py` (`report_path` precedent), `backend/src/kestrel_backend/graph/state_contracts.py` (`validate_state` returns original dict; `extra="ignore"`), `backend/src/kestrel_backend/main.py` (accumulator `else` branch), `backend/src/kestrel_backend/graph/nodes/reporting.py`, `backend/src/kestrel_backend/graph/pipeline_config.py` (`SynthesisConfig`).
- Builds on: PR #84 (per-node performance report), PR #85 (module-aware synthesis caps), PR #86 (research register + `writing_style.py`, merged to dev as `d00cc35`; this branch carries its em-dash-polish follow-up).
- Learnings: `docs/solutions/performance-issues/synthesis-context-window-overflow-silent-fallback-2026-06-22.md` (present); the LangGraph updates-mode reducer-replication learning (principle applied + verified against `main.py`; its solution doc is on the pending `docs/langgraph-updates-reducer-learning` branch).
- Baseline: the 24-analyte Brown validation run (git_sha `d00cc35`, completed 2026-06-23 UTC / evening of 2026-06-22 local): context 94,225 chars / ~26.9K est tokens / 100K-token target; direct_findings 2,433 (150 shown); hypotheses 66; literature_support 33.
