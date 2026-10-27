---
title: "feat: Signed-weight data spine (module_spine) — ingest signed kME/kIM + module→outcome direction"
type: feat
status: active
date: 2026-07-16
deepened: 2026-07-16
origin: docs/brainstorms/2026-07-16-signed-weight-data-spine-requirements.md
---

# feat: Signed-weight data spine (module_spine)

## Overview

Thread per-member **signed kME**, per-member **kIM** (raw intramodular connectivity), and an
optional per-module **eigengene→outcome direction** from the analyte-upload path into pipeline
state, exposed as a module-centric `module_spine: dict[group → ModuleSpine]` field on
`DiscoveryState`. This is Axis A of the discovery-pipeline hypothesis-generation improvements
project and it OWNS the `module-weight-schema` seam — the shape of this signed data in state,
which axes B (triage), D (entity-semantics), and E (synthesis) all consume.

This axis touches **only** the upload path: frontend column-mapping, the shared R19 ingest gate,
`state.py`, and `intake.py`. It does **not** change triage / integration / synthesis behavior.

## Problem Frame

The discovery pipeline's failure mode is **direction (sign), not hit rate** — a measured **30.8%
sign-inversion rate**. Today the pipeline is handed a bare member list (`{name, group?, type?}`)
and has no signal for which way each member moves with its module, nor which way the module points
toward the outcome, so it explains co-variation it has never seen. Finding #1 is the root fix:
give state the signed within-module structure (see origin:
docs/brainstorms/2026-07-16-signed-weight-data-spine-requirements.md).

**Do not overclaim** (must survive into implementation): D1 rank concordances are n=13–15
(suggestive; rules out *strong* not *moderate* concordance); the solid result is the D2 null (956
draws); Arivale is a wellness cohort so non-coherence with Brown ≠ Brown being wrong. The signed
spine makes the direction signal *available* — it does not by itself prove strong concordance. The
**low-n kME caveat** (kME can crown a weakly-connected member on eigengene correlation alone) is
why we also carry **kIM** as a downstream veto input (L5).

## Requirements Trace

- R1. Per-member signed **kME** ∈ [-1, 1] ingested from the upload and present in state (L1, L4).
- R2. Per-member **kIM** (optional) carried alongside kME for the downstream low-n veto (L2, L5).
  **kIM is raw intramodular connectivity (kWithin) — unbounded and non-negative — NOT a correlation;
  it is validated numeric + finite + `>= 0`, NOT bounded to [-1, 1].** (See "Ledger clarification"
  under Open Questions — this refines L4's range clause, which is correct for kME but wrong for raw kIM.)
- R3. Optional per-module **eigengene→outcome direction** (signed ME-trait correlation + trait
  label). A panel without it still runs (L3), but the direction is **load-bearing for the
  sign-inversion metric** (R8): the metric is member-vs-*outcome* = sign(kME) × sign(direction), so
  without it the metric is uncomputable — the coverage summary must flag that, not silently pass.
- R4. Schema is a **module-centric `ModuleSpine`** keyed by group, holding `members{name→kME(+kIM)}`,
  optional `direction`, and a **reserved** correlation slot for v2 (L1, L2).
- R5. Ingest vehicle = **numeric kME/kIM columns** in the existing PR#92 per-row column-mapping;
  values validated numeric ∈ [-1, 1]; **missing allowed** (partial panels degrade) (L4).
- R6. `module_spine` is **cleanly optional/absent** for classic and no-kME runs; nothing downstream
  may require it (L7).
- R7. Preserve per-(name, group) keying with signed **magnitude** (never sign-only) so the axis-D
  sign-coherence guard and axis-E direction label can read it (L15, L16, L17/L18). `members` is a
  **`dict[name → MemberWeight]`** (matches the `entity_type_hints`/`entity_groups` precedent so
  downstream axes do not re-index a list). `MemberWeight.name` MUST equal the verbatim first-seen
  canonical run-set name (`key_to_canonical`), byte-identical to `entity_groups` keys, or the
  downstream name→CURIE join misses silently.
- R8. A measurement substrate: per-run `module_spine` coverage (members with kME, with kIM, whether
  a direction was supplied) so the sign-inversion-rate metric is interpretable, with pinned inputs.

## Scope Boundaries

- No change to triage, integration, bridge, or synthesis **logic** — those axes consume
  `module_spine` but are out of this plan.
- No connectivity-floor / p-value thresholding (the kIM veto logic is axis B).

### Deferred to Separate Tasks

- Full within-module **N×N correlation matrix** ingestion and validation: **v2** (L2). This plan
  reserves the schema slot only.
- Consumption of `module_spine` by triage / entity-semantics / synthesis: sibling axes B/D/E.

## Context & Research

### Relevant Code and Patterns

- `backend/src/kestrel_backend/graph/state.py` — Pydantic models + `DiscoveryState` TypedDict.
  Precedent per-member maps to mirror: `entity_type_hints: dict[str,str]`, `entity_groups:
  dict[str, list[str]]` (both plain single-writer fields, **no `operator.add` reducer** — set once
  at intake before the direct_kg|cold_start fork; the state.py comment for `structured_analytes`
  documents that a reducer here would duplicate-concat).
- `backend/src/kestrel_backend/analyte_ingest.py` — `validate_and_normalize(...) -> NormalizedPanel`,
  the single R19 source of truth: row cap, per-field length + control-char checks (`_has_control_chars`,
  `_clean_field`), case-insensitive dedup with cross-group accretion under a first-seen canonical
  name, group-selection filter, run ceiling. Returns errors (never raises).
- `backend/src/kestrel_backend/graph/nodes/intake.py` (structured branch, ~lines 643–719) —
  re-runs `validate_and_normalize` (gate #2) and emits `entity_type_hints` / `entity_groups` in the
  IntakeOutput dict; the `upload_rejected` short-circuit pattern.
- `backend/src/kestrel_backend/main.py` (~lines 915–1030) — WS gate #1: type-checks
  `structured_analytes`/`selected_groups`, calls `validate_and_normalize`, rejects via `ErrorMessage`,
  passes the panel to `handle_pipeline_mode`.
- `client/src/lib/analyteParse.ts` — `MappingTarget` / `ColumnMapping` / `StructuredAnalyte`,
  `suggestMapping` (unambiguous-header auto-map, R6), `normalizeType`, `buildAnalytes`,
  `isFormulaInjection`. `client/src/components/ColumnMappingPanel.tsx`, `AnalyteUpload.tsx` (UI glue);
  `client/src/lib/analyteParse.test.ts` (vitest).

### Institutional Learnings

- `docs/solutions/best-practices/untrusted-panel-upload-entry-point-validation-2026-07-09.md` —
  validate at **every** entry point behind one shared helper (WS gate #1 + intake gate #2); reject by
  routing (`upload_rejected` → END), never optimistically clear client state. The new numeric columns
  must ride the same dual-gate helper.
- state.py reducer learning (in-file): only parallel-superstep fields carry `operator.add`;
  `module_spine` is single-writer → plain field.
- Validation memo (run 20260716-212442): kME is WGCNA-native, bounded [-1,1]; **kIM** addresses the
  low-n kME caveat — "cheap addition to the schema."

### External References

- None used — pure internal schema change with strong local patterns (`entity_groups`,
  `entity_type_hints`) and no external contract surface.

## Key Technical Decisions

- **Module-centric `ModuleSpine` keyed by group** (L1): a member can belong to multiple modules with
  a *different* kME/kIM per module; a flat `name→kME` map would collapse that. Mirrors the existing
  `entity_groups: name→[groups]` multi-membership reality.
- **`members` is `dict[name → MemberWeight]`** (not a list, and not a bare `dict[name,float]`): the map
  keying matches R7 + the `entity_groups`/`entity_type_hints` precedent so axes B/D/E look up by name
  without re-indexing; `MemberWeight` bundles kME + optional kIM so both travel together per member.
- **`members` holds only weighted members**: a member with no kME cell is simply **absent from
  `members`** (it still exists in `run_analytes`/`entity_groups`). `MemberWeight.kme` therefore stays
  **required** — "missing-allowed" means *excluded from the spine*, never stored as `None`.
- **Plain single-writer state field, no reducer**: set once at intake, mirroring `entity_groups`.
- **Optional everywhere else**: no kME column → no `module_spine` (classic behavior, R6). Optional
  `direction`; optional `kim`. Absent for classic runs.
- **kME vs kIM validation differ** (feasibility/adversarial review): **kME** is a correlation → numeric,
  finite, ∈ [-1, 1] (with ±1e-6 epsilon tolerance so a serialized 1.0000002 clamps to 1.0 rather than
  rejecting a legitimate export). **kIM** is *raw* intramodular connectivity (kWithin), unbounded and
  non-negative → numeric, finite, `>= 0`. Applying a [-1, 1] bound to kIM would reject real Brown uploads.
- **Reject, don't clip, out-of-range kME/kIM**: a genuinely out-of-range value (kME beyond ±1 past the
  epsilon; kIM negative or non-finite) signals a mis-mapped column — exactly the silent-misdirection
  failure this axis exists to prevent — so the shared helper rejects the whole panel.
- **Conflict policy for a collapsed (name, group) key**: two rows folding to the same
  (case-insensitive name, canonicalized group) with **non-equal** kME (or kIM) → **reject** (matches the
  reject-don't-clip ethos; a silent first/last-write-wins could attach the wrong sign). Equal values are
  idempotent.
- **Canonicalize the group key consistently** for member accretion, direction matching, and
  `ModuleSpine` keying — using the *same* normalization the existing selection filter uses
  (`group.strip().lower()` for the join key; preserve a display label). The ME-trait direction table is
  a *separate* export, so cross-table casing/whitespace drift would otherwise silently warn-and-drop the
  direction (losing the L17/L18 synthesis input). A kME/kIM value on a **group-less** member has no
  `ModuleSpine` bucket → dropped with a surfaced warning (not silently).
- **Reserved v2 correlation slot** (L1): kept as an explicitly-absent field for seam stability. This is a
  **deliberate exception to YAGNI** — noted because the v2 matrix shape is undesigned; the field stays
  `None` until the v2 task defines it.

## Open Questions

### Ledger clarification (surfaced at plan review — touches the `module-weight-schema` seam)

- **L4 range clause applies to kME only, not kIM.** L4 reads "numeric kME/kIM columns … validate
  numeric in [-1,1]." That bound is correct for kME (a correlation) but wrong for **raw kIM**
  (kWithin), which is unbounded and non-negative. The plan validates kME ∈ [-1,1] and kIM as
  numeric/finite/`>= 0`. This is a *clarification*, not a reversal, of L4 — the "numeric column,
  missing-allowed, reject-not-clip" spirit is preserved. Axis B (kIM veto, L5) depends on kIM
  surviving ingest, so getting this right is load-bearing. **Flag for coordinator sign-off.**

### Resolved During Planning

- Keying (name vs. (name,group))? → group-keyed `ModuleSpine` (L1) resolves it; per-(name,group)
  weights fall out naturally. `members` inside a `ModuleSpine` is `dict[name → MemberWeight]`.
- Where does the module→outcome direction arrive? → a small separate per-group input
  `{group, eigengene_trait_correlation, trait_label}` list carried on the WS payload beside
  `structured_analytes` (the ME-trait table is exported separately from the member table), validated
  in the same helper and **bounded in count**. Optional for the run; required for the metric (R3/R8).

### Deferred to Implementation

- Exact `NormalizedPanel` field names for the new maps and the direction list (helper-internal).
- Whether coverage telemetry rides the existing `node_timings`/telemetry dict or a dedicated
  `module_spine_coverage` state key — decide when wiring intake (both are single-writer plain fields).
- Frontend auto-suggest header regexes for kME/kIM — finalize against real Brown export headers.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation
> specification. The implementing agent should treat it as context, not code to reproduce.*

Data flow (upload path only):

```
CSV/TSV row: {analyte, group, type, kME, kIM}   +   ME-trait rows: {group, ME_trait_corr, trait}
        │  analyteParse.ts (parse + column-map + numeric coerce)
        ▼
WS payload: structured_analytes[{name,group?,type?,kme?,kim?}] + module_directions[{group,eigengene_trait_correlation,trait_label}]
        │  main.py gate #1  ──►  validate_and_normalize(...)  ◄── intake.py gate #2 (Studio/harness)
        ▼                          (numeric parse, [-1,1] range, missing-allowed, per-(name,group) accretion)
NormalizedPanel(.. member weights per group, validated directions ..)
        │  intake.py builds ModuleSpine per group
        ▼
DiscoveryState.module_spine: dict[group → ModuleSpine{members:[MemberWeight], direction?, correlation=None}]
        │  (absent for classic / no-kME runs)
        ▼  consumed downstream by axes B / D / E (out of scope here)
```

Schema shape (directional — not final Pydantic):

```
MemberWeight   = { name: str (canonical run-set name), kme: float∈[-1,1] (required), kim: float>=0|None }
ModuleDirection= { eigengene_trait_correlation: float∈[-1,1], trait_label: str (<= field_max, control-char clean) }
ModuleSpine    = { group: str (display label), members: dict[name → MemberWeight] (only weighted members),
                   direction: ModuleDirection|None, correlation: None (reserved v2) }
DiscoveryState.module_spine: dict[str, ModuleSpine]   # group-canonical key; plain single-writer; absent when no kME
```

Validation note: kME uses a ±1e-6 epsilon on the [-1,1] bound; kIM is validated numeric/finite/`>= 0`
(raw kWithin is unbounded, NOT a correlation). A collapsed (name, group) key with non-equal kME/kIM is
rejected. Group keys are canonicalized (same `strip().lower()` as the selection filter) so the separately
exported ME-trait direction table joins reliably.

## Implementation Units

- [ ] **Unit 1: State models + `module_spine` field**

**Goal:** Define `MemberWeight`, `ModuleDirection`, `ModuleSpine` (frozen Pydantic) and add the
optional single-writer `module_spine: dict[str, ModuleSpine]` to `DiscoveryState`.

**Requirements:** R1, R2, R3, R4, R6, R7

**Dependencies:** None

**Files:**
- Modify: `backend/src/kestrel_backend/graph/state.py`
- Test: `backend/tests/graph/test_state_module_spine.py`

**Approach:**
- Frozen models mirroring the existing `EntityResolution`/`NoveltyScore` style.
  - `MemberWeight.kme`: `float`, **required**, `ge=-1, le=1`. `MemberWeight.kim`: `float | None`,
    **`ge=0`** (raw kWithin is unbounded/non-negative — do NOT bound to [-1,1]). `MemberWeight.name`:
    the canonical run-set name.
  - `ModuleDirection.eigengene_trait_correlation`: `float`, `ge=-1, le=1`; `trait_label`: `str`.
  - `ModuleSpine.members`: `dict[str, MemberWeight]` (name → weight, weighted members only);
    `direction: ModuleDirection | None`; `correlation: None` reserved-absent (deliberate YAGNI exception).
- `module_spine` is a plain field under `total=False` (no `operator.add`) — add the same
  single-writer rationale comment used for `entity_groups`/`structured_analytes`.

**Patterns to follow:** frozen `ConfigDict` models and the plain-field comment block for
`structured_analytes`/`entity_groups` in `state.py`.

**Test scenarios:**
- Happy path: construct a `ModuleSpine` with a `members` map of two `MemberWeight` (one with kIM, one
  without) and a `ModuleDirection`; round-trips and is frozen (mutation raises).
- Edge case: `kme` at boundaries -1.0 and 1.0 accepted; `kim` of 0, 3, 40+ accepted (unbounded).
- Error path: `kme` = 1.5 (or -2) raises `ValidationError`; `kim` = -1 raises `ValidationError`;
  constructing `MemberWeight` with no `kme` raises `ValidationError` (kME is mandatory).
- Edge case: missing `kim` yields `None`; missing `direction` yields `None`.
- Edge case: a `DiscoveryState` with no `module_spine` key is valid (absent-optional).

**Verification:** models importable; a state dict with and without `module_spine` both type-check;
boundary and out-of-range behavior as enumerated.

---

- [ ] **Unit 2: Ingest — parse/validate/normalize kME, kIM, and module directions**

**Goal:** Extend `validate_and_normalize` / `NormalizedPanel` to parse the new numeric per-row
columns and the optional per-module direction list, with fail-fast range/type validation and
per-(name, group) accretion.

**Requirements:** R1, R2, R3, R5, R7

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/src/kestrel_backend/analyte_ingest.py`
- Test: `backend/tests/test_analyte_ingest_weights.py` (or extend the existing analyte-ingest test file)

**Approach:**
- Add helper-internal maps to `NormalizedPanel` for per-(canonical-name, canonical-group) kME/kIM and a
  validated directions structure, plus the built `ModuleSpine` dict (or the raw material for intake to build).
- Parse kME/kIM per row → float when present (RETURNED error string, matching existing R19 style — never
  raise):
  - **kME**: reject non-numeric-when-present; reject outside `[-1 - 1e-6, 1 + 1e-6]`, then clamp a value
    within ±1e-6 of a bound to the bound (tolerate serialized 1.0000002).
  - **kIM**: reject non-numeric-when-present; reject negative or non-finite. Do **NOT** apply the [-1,1]
    bound — raw kWithin is unbounded/non-negative.
  - Blank/absent cell → member simply **not added** to that module's `members` map (missing-allowed).
- Attach weights under the **canonical run-set name** (`key_to_canonical[key]`, byte-identical to
  `entity_groups` keys) and the **canonicalized group** (`group.strip().lower()`, same as the selection
  filter; keep a display label). A member seen in two groups gets a `MemberWeight` under each group.
- **Conflict:** if the same (canonical-name, canonical-group) is seen twice with **non-equal** kME (or
  kIM) → reject (mis-map/duplicate footgun). Equal repeats are idempotent.
- **Group-less weight:** a kME/kIM value on a row with no group → dropped with a surfaced warning (no
  `ModuleSpine` bucket exists); the run proceeds.
- Validate each direction: numeric `eigengene_trait_correlation` ∈ [-1, 1]; `trait_label` non-empty,
  `<= analyte_field_max_len` (512), control-char clean (reuse `_has_control_chars`, `_clean_field`);
  match its group against the panel's **canonicalized** groups; group not present → warn-and-drop with the
  offending vs. available labels surfaced (don't reject the run).
- **Bound `module_directions` count** before per-item work — cap at the number of distinct groups in the
  panel (or a small constant) — so the Studio/harness gate #2 (which bypasses the WS byte cap) is bounded
  identically.
- No kME column anywhere → produce no module-spine material (classic behavior).

**Execution note:** Extend the existing R19 test coverage first — this helper is the single source of
truth crossed by both gates; add the weight cases alongside the current row-cap/dedup cases.

**Patterns to follow:** `validate_and_normalize` fail-fast RETURN-not-raise style; `_reject`,
`_clean_field`, `_has_control_chars`, the `key_to_canonical` cross-group accretion loop.

**Test scenarios:**
- Happy path: panel with kME (+some kIM incl. values > 1 like 40) → per-(name, group) weights populated;
  run set unchanged; kIM > 1 is accepted (not rejected).
- Happy path: a member in two groups with different kME → distinct `MemberWeight` under each group.
- Happy path: a case/whitespace-variant row name attaches its kME under the **canonical** run-set name
  (byte-identical to `entity_groups` keys), not the raw row name.
- Edge case: kME column absent → no module-spine material, normal panel still returned.
- Edge case: some rows blank kME, others populated → only populated members appear in `members` (partial).
- Edge case: kME boundary -1.0 / 1.0 accepted; kME 1.0000002 clamps to 1.0 (epsilon); kME 1.5 rejected.
- Error path: non-numeric kME cell → RETURNED error, run set cleared (rejection), no partial launch.
- Error path: negative kIM → RETURNED error; non-finite kME/kIM → RETURNED error.
- Error path: same (name, group) with two **different** kME values → RETURNED conflict error.
- Edge case: kME on a group-less row → dropped with a surfaced warning; run proceeds.
- Edge case: group casing/whitespace drift between panel and a direction entry → they still join
  (canonicalized), direction attached.
- Error path/edge: direction with out-of-range correlation → error; over-length `trait_label` → error;
  blank/control-char `trait_label` → error; direction for an unknown group → warn-and-drop (run proceeds).
- Edge case: `module_directions` longer than the distinct-group count (or the constant cap) → RETURNED
  cap error before per-item parse (bounds the harness gate).
- Integration: idempotent — running the helper twice on the same input yields identical weights
  (covers the dual-gate re-assertion).

**Verification:** all weight/direction cases behave as enumerated; kIM > 1 ingests successfully;
existing R19 tests still pass; no code path raises.

---

- [ ] **Unit 3: Intake wiring — build and emit `module_spine`**

**Goal:** In the intake structured branch, build `module_spine` from the normalized panel and emit
it in the IntakeOutput dict; emit coverage telemetry; keep it absent for no-kME/classic runs.

**Requirements:** R1–R4, R6, R7, R8

**Dependencies:** Units 1, 2, **and Unit 4** (the direction half reads `state.get("module_directions")`,
which Unit 4 threads through runner/state and passes into the helper's new `module_directions` param; the
kME/kIM half rides the already-threaded `structured_analytes`). Land Unit 4's state/runner threading before
intake consumes directions.

**Files:**
- Modify: `backend/src/kestrel_backend/graph/nodes/intake.py`
- Test: `backend/tests/graph/nodes/test_intake_module_spine.py`

**Approach:**
- After the existing gate-#2 `validate_and_normalize` call (now passed `module_directions` from state),
  assemble `ModuleSpine` per **canonical group** from the normalized weights + validated directions;
  attach to the returned dict as `module_spine` **only when non-empty** (absent otherwise — R6).
- Emit a small coverage summary (members-with-kME, members-with-kIM, direction-supplied per group, and a
  **`metric_computable` flag** = at least one module has both members-with-kME and a direction) via the
  existing telemetry surface (`node_timings`-adjacent single-writer dict, or a dedicated
  `module_spine_coverage` field — decide at wiring, both are plain fields). When no direction was
  supplied, the summary explicitly flags the sign-inversion metric as uncomputable (R3/R8) rather than
  silently reporting healthy coverage.
- Do not alter the `upload_rejected` short-circuit or the classic (non-upload) branch.

**Patterns to follow:** the structured-branch IntakeOutput dict assembly (~lines 685–719) that emits
`entity_type_hints`/`entity_groups`; the gate-#2 re-validation pattern.

**Test scenarios:**
- Happy path: structured panel with kME → intake output contains `module_spine` with correct
  per-group members and (when supplied) direction.
- Edge case: panel without kME → intake output has **no** `module_spine` key (classic parity).
- Edge case: classic (non-structured) query → unchanged; no `module_spine`.
- Integration: rejected panel (existing R19 rejection) still short-circuits via `upload_rejected`
  and does not emit a partial `module_spine`.
- Happy path: coverage summary reports the right members-with-kME / with-kIM / direction-supplied
  counts for a mixed panel.

**Verification:** downstream state carries `module_spine` exactly when kME was supplied; classic and
no-kME runs are byte-identical to today; coverage summary present and correct.

---

- [ ] **Unit 4: WS payload plumbing (main.py) + runner signatures**

**Goal:** Accept `module_directions` (and confirm the per-row kME/kIM already flow inside
`structured_analytes`) on the WS message and thread them through `handle_pipeline_mode` / runner into
state, behind gate #1.

**Requirements:** R3, R5, R6

**Dependencies:** Units 1, 2

**Files:**
- Modify: `backend/src/kestrel_backend/main.py`
- Modify: `backend/src/kestrel_backend/graph/runner.py`
- Test: `backend/tests/test_main_upload_weights.py` (or extend existing WS-handler tests)

**Approach:**
- kME/kIM ride each row of `structured_analytes` (already a `list[dict]`) — no new WS field needed for
  them; confirm the type-check tolerates extra keys.
- Add an optional `module_directions` field to the WS message and the runner/`handle_pipeline_mode`
  signatures (mirror `selected_groups` plumbing); pass into initial state and into
  `validate_and_normalize` (its signature gains a `module_directions` param that **both** gates pass).
  Type-check as list; reject malformed via `ErrorMessage` at gate #1 (fail-fast) consistent with existing
  guards. The authoritative count cap on `module_directions` lives in the shared helper (Unit 2), so both
  entry points are bounded identically.
- Gate #1 still calls `validate_and_normalize`; on rejection, surface the reason and skip the run.

**Patterns to follow:** `selected_groups` / `structured_analytes` plumbing in `main.py` (~915–1030)
and `runner.py` (the three functions carrying `structured_analytes`).

**Test scenarios:**
- Happy path: WS message with rows carrying kME/kIM + a `module_directions` list → state initialized
  with the panel and directions; run proceeds.
- Edge case: message with kME rows but no `module_directions` → proceeds (direction optional).
- Error path: `module_directions` not a list → `ErrorMessage`, no run (mirrors `selected_groups`).
- Integration: Studio/harness entry (bypasses main.py) still gets weights via intake gate #2 (proves
  the dual-gate holds for the new fields).

**Verification:** both entry paths (WS and runner-direct) land identical `module_spine` for the same
input; malformed direction payloads are rejected at the door.

---

- [ ] **Unit 5: Frontend column-mapping — kME/kIM columns + direction input**

**Goal:** Let the user map numeric kME/kIM columns in the existing per-row mapping UI, parse them into
`structured_analytes`, and optionally supply per-module directions; send both over the WS.

**Requirements:** R3, R5

**Dependencies:** Units 1–4 (backend contract)

**Files:**
- Modify: `client/src/lib/analyteParse.ts`
- Modify: `client/src/components/ColumnMappingPanel.tsx`
- Modify: `client/src/components/AnalyteUpload.tsx`
- Modify: `client/src/hooks/useWebSocket.ts`
- Test: `client/src/lib/analyteParse.test.ts`

**Approach:**
- Add `kme` and `kim` to `MappingTarget` / `ColumnMapping` and `kme?`/`kim?` to `StructuredAnalyte`;
  extend `buildAnalytes` to parse those cells to numbers (blank → undefined). Extend `suggestMapping` with
  unambiguous kME/kIM header regexes (R6-style — leave ambiguous headers like `value` unset).
- **Fix the formula-injection collision:** `isFormulaInjection` flags a leading `-`, and the mapping
  preview badges every cell through it — so ~half the (negative) kME cells would show a false "looks like a
  spreadsheet formula" warning, colliding with the whole point of signed kME. Exclude numeric-mapped
  (kme/kim) columns from that badge, or refine the check to not flag values that parse as a finite number.
- **Don't silently degrade a mis-mapped column** (this is the client-side twin of "reject don't clip"):
  when a kME column IS mapped but a non-trivial fraction of its cells are non-numeric/out-of-range, badge
  the offending cells in the preview (same amber visual language as the existing injection badge) and show
  a count ("N kME values out of range or non-numeric — the server will reject this upload") **before**
  the user can Continue. A mapped-but-nearly-all-invalid kME column must not be indistinguishable from a
  clean no-kME run. (Range check mirrors backend: kME ∈ [-1,1]; kIM finite ≥ 0.)
- **Direction input IA** (make it concrete, not "a small mapping"): after the analyte/group mapping is
  confirmed (groups are known via `distinctGroups`), render a per-module direction table — one row per
  distinct group, each with a bounded numeric `eigengene_trait_correlation` field and a **group value bound
  to `distinctGroups(analytes)`** (a dropdown/derived label, so an unknown-group typo is structurally
  impossible), plus **one shared `trait_label` field entered once** for the run (WGCNA runs study a single
  outcome; avoid five spellings of the same trait). All rows blank = "no direction supplied" (R3). Send as
  `module_directions` in the WS payload.
- **Layout:** keep the 3 existing per-row targets (analyte/group/type) grouped; add kME/kIM as a second,
  visually distinct "signed weights (optional)" mapping row; place the direction table below the
  preview/row-count summary. Keep client caps/defense-in-depth; backend R19 stays authoritative.

**Patterns to follow:** existing `MappingTarget`/`ColumnMapping`/`suggestMapping`/`buildAnalytes`,
`isFormulaInjection` + the preview-cell badge in `ColumnMappingPanel.tsx`, and the `structured_analytes`
send in `useWebSocket.ts`.

**Test scenarios:**
- Happy path: headers `kME`/`kIM` auto-suggested; rows parse to numeric weights on each analyte.
- Happy path: a **negative** kME cell is NOT badged as formula-injection (regression guard for the collision).
- Edge case: blank kME cell → `undefined` (member without weight); kME column unmapped → analytes built
  exactly as today (no weights, no regression).
- Edge case/error: a mapped kME column with non-numeric/out-of-range cells → offending cells badged +
  a count shown; Continue is gated/warned (not silently coerced to a no-kME run).
- Happy path: ambiguous header (e.g. `value`) is NOT auto-mapped to kME (R6 unambiguous-only).
- Happy path: a supplied direction (group from the dropdown + shared trait_label + correlation) is included
  in the outgoing `module_directions`; absent/all-blank → no direction sent.
- Edge case: the direction group selector only offers `distinctGroups(analytes)` (unknown-group typo
  impossible).

**Verification:** `npm run check` passes; vitest covers the weight-parse, negative-kME-not-badged, and
mapped-but-invalid-column cases; a panel with kME produces rows carrying numeric `kme`/`kim`; a panel
without kME is unchanged.

---

- [ ] **Unit 6: Measurement substrate + docs**

**Goal:** Ensure the sign-inversion-rate metric is computable and interpretable: pin the Brown module
upload used for before/after, and document the `module_spine` coverage summary as the provenance the
metric reads.

**Requirements:** R8

**Dependencies:** Units 1–5

**Files:**
- Modify: `backend/assessment_data/` (pin the Brown module upload + a small README note; artifact
  hygiene SOP — pinned file + data commit/SHA)
- Modify: `CLAUDE.md` or a short note under `docs/` describing the `module_spine` field + coverage
  summary for downstream axes.

**Approach:**
- The eval itself (synthesis-claimed direction vs. `module_spine` ground truth) is axis E; this unit
  only guarantees the substrate exists: signed ground truth in state + a reproducible pinned input +
  the coverage summary. No new eval harness here.
- **The pinned Brown upload MUST include the ME-trait/direction table.** The sign-inversion metric is
  member-vs-*outcome* = sign(kME) × sign(direction); without direction the metric is uncomputable. If the
  pinned input lacks it, `metric_computable` is false and Unit 6's verification is not satisfied — pin
  both the member table (kME/kIM) and the ME-trait table so the metric can actually be computed.

**Test expectation:** none — documentation + pinned data artifact; behavior is covered by Units 1–5.

**Verification:** the pinned Brown upload (member table + ME-trait table) reproduces the same
`module_spine` deterministically; the coverage summary reports `metric_computable = true` for it; the
schema + coverage summary are documented for axes B/D/E.

## System-Wide Impact

- **Interaction graph:** new optional state field read (later) by triage/entity-semantics/synthesis;
  written only by intake. No node currently *requires* it (R6).
- **Error propagation:** weight/direction problems surface on the existing errors/warnings channel via
  the shared helper; rejections route through the existing `upload_rejected` → END path.
- **State lifecycle risks:** single-writer plain field (no reducer) — matches `entity_groups`; a
  reducer would duplicate-concat (documented state.py hazard).
- **API surface parity:** both entry points (WS gate #1, intake gate #2) must handle the new fields
  identically — enforced by the single shared helper (learning:
  untrusted-panel-upload-entry-point-validation).
- **Integration coverage:** dual-gate idempotence (Unit 2) + runner-direct vs WS parity (Unit 4).
- **Unchanged invariants:** classic mode and existing no-kME uploads produce byte-identical state; the
  `{name, group?, type?}` panel contract is a superset-extended, not broken (extra optional keys).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| A mis-mapped numeric column silently injects wrong signs (the exact failure this axis fixes) | Reject (not clip) out-of-range kME/kIM at the shared gate; unambiguous-only header auto-map on the client. |
| Reducer added by habit → duplicate-concat of `module_spine` | Plain single-writer field with the documented no-reducer rationale (Unit 1). |
| Overclaiming what signed-kME buys | Problem Frame + origin doc carry the n=13–15 / D2-null / Arivale caveats; kIM carried for the low-n veto. |
| Direction input UX complexity | Direction is fully optional; panels without it run unchanged. |
| Downstream axes assume `module_spine` always present | R6 makes it absent for classic/no-kME; downstream must treat it as optional (documented in Unit 6). |

## Documentation / Operational Notes

- Document the `module_spine` schema + coverage summary as the cross-axis contract (Unit 6) so axes
  B/D/E consume a stable shape.
- No migration / rollout concern — additive optional state field; no DB or wire-breaking change.

## Sources & References

- **Origin document:** docs/brainstorms/2026-07-16-signed-weight-data-spine-requirements.md
- Related code: `backend/src/kestrel_backend/graph/state.py`,
  `backend/src/kestrel_backend/analyte_ingest.py`,
  `backend/src/kestrel_backend/graph/nodes/intake.py`,
  `backend/src/kestrel_backend/main.py`, `client/src/lib/analyteParse.ts`
- Related learning: docs/solutions/best-practices/untrusted-panel-upload-entry-point-validation-2026-07-09.md
- Prior art (PR #92): docs/plans/2026-07-09-001-feat-analyte-file-upload-column-mapping-plan.md
- Method validation: `~/.claude/skills/orchestrate/runs/20260716-212442/validation-memo.md`
