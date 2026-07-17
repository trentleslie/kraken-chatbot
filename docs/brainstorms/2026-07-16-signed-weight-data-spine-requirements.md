# Signed-weight data spine (module_spine) — Requirements

**Date:** 2026-07-16
**App:** kraken-chatbot (discovery pipeline)
**Axis:** A — Signed-weight data spine (of the discovery-pipeline hypothesis-generation improvements project)
**Owns seam:** `module-weight-schema`
**Status:** Brainstorm — pending CP2 plan approval

---

## 1. Motivation

The discovery pipeline's failure mode is **direction (sign), not hit rate**. A measured
**30.8% sign-inversion rate** means the pipeline currently explains co-variation it has never
seen: it is handed a bare member list (name/group/type) and has no signal for *which way* each
member moves with its module, nor which way the module points toward the outcome. ~18–20%
Tier-3 progression is acceptable; the real problem is that the one hit that mattered pointed at
the wrong outcome and nothing in state could say which way it would go.

**Finding #1 — feed the module's signed kME (and connectivity), not just a member list —** is
the root direction fix. This axis threads per-member **signed kME**, per-member **kIM** (raw
intramodular connectivity), and an optional per-module **eigengene→outcome direction** from the
upload path into pipeline state so that downstream axes (B/triage, D/entity-semantics,
E/synthesis) can consume signed within-module structure.

### What this buys — and what it does NOT (do not overclaim)

- **Buys:** a signed, magnitude-bearing within-module structure in state, plus an optional
  signed module→outcome orientation. This is the substrate every downstream direction check
  reads (L5, L15, L17/L18).
- **Does NOT buy** (must survive into the spec): the D1 rank concordances are n=13–15
  (suggestive; rules out *strong* not *moderate* concordance). The solid result is the D2 null
  (956 draws). Arivale is a wellness cohort, so non-coherence with Brown ≠ Brown being wrong.
  The signed spine makes the direction signal *available*; it does not by itself prove the
  concordance is strong. The **low-n kME caveat** (kME can crown a weakly-connected member on
  eigengene correlation alone) is why we also carry **kIM** (validation memo, Finding #1).

## 2. Scope

**In scope (this axis owns `intake.py` + `state.py` + the upload path only):**

- Frontend upload/column-mapping (`client/src/lib/analyteParse.ts`, `ColumnMappingPanel.tsx`,
  `AnalyteUpload.tsx`): add numeric `kME` and `kIM` column targets + an optional
  module→outcome direction input.
- WebSocket ingest + shared R19 gate (`main.py`, `analyte_ingest.py`): parse, validate, and
  normalize the new numeric columns and the optional per-module direction.
- State schema (`state.py`): define the `ModuleSpine` model and the `module_spine` state field.
- Intake wiring (`intake.py`): populate `module_spine` from the normalized panel.

**Explicitly OUT of scope (owned by other axes — do NOT change):**

- Triage logic (axis B) — consumes `module_spine` but is not touched here.
- Integration / bridge specificity (axis C).
- Entity-semantics / sign-coherence guard (axis D).
- Synthesis direction label (axis E).
- **Full N×N correlation matrix** — deferred to v2 (L2). The schema RESERVES a slot; no
  ingestion/validation for it in v1.

## 3. Key decisions (from the decisions ledger — FIXED, do not relitigate)

| # | Decision | Choice |
|---|----------|--------|
| L1 | Schema shape | **Module-centric `ModuleSpine` object** — `module_spine: dict[group → ModuleSpine]`, each holding `members: {name → kME}`, optional ME→outcome direction, optional per-member kIM, reserved correlation slot |
| L2 | v1 scope | per-member signed kME + optional ME→outcome direction + kIM; **defer full correlation matrix to v2** |
| L3 | ME→outcome direction | **Yes**, optional: signed ME-trait correlation + trait label; panels without it still run |
| L4 | Ingest vehicle | numeric **kME / kIM columns** in the existing PR#92 per-row column-mapping; validate numeric ∈ [-1, 1]; **missing allowed** (partial panels degrade) |

### Downstream constraints this schema MUST satisfy (from sibling axes' ledger)

| # | Constraint | Implication for this schema |
|---|-----------|-----------------------------|
| L5 | Triage centrality = \|kME\| primary + **kIM veto** | `kIM` is load-bearing — carry it per member, keyed like kME. |
| L7 | Triage layers centrality over an edge-count fallback for **no-kME runs** | `module_spine` must be **cleanly optional/absent** for classic (non-upload) runs; downstream must never require it. |
| L15 | (D) sign-coherence guard reads per-(name,group) signed kME **+ magnitude** | Keep **signed float magnitude**, never sign-only. Per-(name,group) keying is preserved because `ModuleSpine` is keyed by group. |
| L16 | (D) splits sign-incoherent groups into sub-programs using these weights | Weights must be readable per group with member identity intact. |
| L17/L18 | (E) synthesis consumes signed kME + ME→outcome direction for a deterministic direction label | Both `members[name].kME` and the optional `direction` must be present and unambiguously signed. |

## 4. Data model — the `module-weight-schema` seam

Keyed by **module (group)** because the module is the unit downstream axes consume, and because
an analyte can belong to multiple modules with a *different* kME/kIM per module (mirrors the
existing `entity_groups: name → [groups]` multi-membership reality — a flat `name → kME` map
would collapse that and lose per-module sign).

```python
class MemberWeight(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str                      # verbatim post-trim analyte name (byte-identical to raw_entities)
    kme: float                     # signed module eigengene-based connectivity ∈ [-1, 1]
    kim: float | None = None       # raw intramodular connectivity (optional; low-n kME veto, L5)

class ModuleDirection(BaseModel):
    model_config = ConfigDict(frozen=True)
    eigengene_trait_correlation: float   # signed ME↔trait correlation ∈ [-1, 1]
    trait_label: str                     # e.g. "frailty index", "progression"

class ModuleSpine(BaseModel):
    model_config = ConfigDict(frozen=True)
    group: str                                   # module / group id
    members: list[MemberWeight]                  # per-member signed weights
    direction: ModuleDirection | None = None     # optional ME→outcome orientation (L3)
    # RESERVED for v2 (L2): within-module correlation matrix. Absent in v1.
    correlation: None = None

# state.py — DiscoveryState (total=False), plain single-writer field (NO operator.add reducer,
# mirroring entity_groups/entity_type_hints — set once at intake before the direct_kg|cold_start fork):
#   module_spine: dict[str, ModuleSpine]   # group -> ModuleSpine; ABSENT for classic/no-kME runs (L7)
```

**Rationale notes:**
- `members` as a list of `MemberWeight` (not a bare `dict[name, float]`) so kME *and* kIM travel
  together per member and the model stays extensible (L2/L5).
- `module_spine` is a plain single-writer field like `entity_groups` — **no `operator.add`
  reducer** (it is set once at intake, before the parallel superstep; a reducer would
  duplicate-concat, per the state.py learning already documented for `structured_analytes`).
- The field is **entirely optional/absent** for classic runs and no-kME uploads (L7). Nothing in
  this axis makes any downstream node *require* it.

## 5. Ingestion, validation, and degradation

**Vehicle (L4):** extend the existing single-file per-row column-mapping.

- **Frontend** (`analyteParse.ts`): add `kme` and `kim` to `MappingTarget` / `ColumnMapping`;
  auto-suggest from headers (`^kme$/^k\.?me$`, `^kim$/^k\.?in$/^connectivity$`) restricted to
  unambiguous matches (R6 precedent); parse numeric, coerce blank→undefined. Per-member weights
  ride each `StructuredAnalyte` row. The optional module→outcome direction is a small separate
  input (per-group `{group, eigengene_trait_correlation, trait_label}` list) — a module CSV
  export usually ships the ME-trait table separately from the member table.
- **Backend R19 gate** (`analyte_ingest.py::validate_and_normalize` + `NormalizedPanel`):
  - Parse `kme`/`kim` per row → float; **reject** non-numeric-when-present (fail-fast, RETURNED
    error, consistent with existing R19 style).
  - **Range check** kME/kIM ∈ [-1, 1]; out-of-range is a rejection (do not silently clip — a
    kME > 1 signals a mis-mapped column).
  - **Missing allowed** (L4): a row with no kME contributes a member without a weight; a whole
    panel with no kME column yields **no `module_spine`** (classic behavior, L7).
  - Per-(name, group) keying: a member seen in two groups accretes a `MemberWeight` under each
    group's `ModuleSpine` (mirrors the existing cross-group `entity_groups` accretion).
  - Validate the optional direction list: `eigengene_trait_correlation` ∈ [-1, 1] numeric,
    `trait_label` non-empty + control-char clean (existing `_has_control_chars`), group must
    match a panel group else warn-and-drop.
- **Intake gate #2** (`intake.py`): re-run the same normalization on non-WS entry paths
  (Studio/harness), identical to the existing dual-gate pattern; emit `module_spine` in the
  IntakeOutput dict; absent when no kME.

**Degradation principle (project SOP — surface, never except-swallow):** any weight problem
degrades the *member/module* it concerns and is surfaced on the `errors` channel; it never
crashes the run and never silently drops a whole panel.

## 6. Measurement hook (required — tied to Finding #1)

Every change needs a measurement hook tied to its finding. Finding #1's metric is the
**sign-inversion rate before/after**.

- This axis's obligation is to make the metric *computable*: the signed ground truth (per-member
  kME sign + module→outcome direction) must be present in state and reproducible from the pinned
  upload. The eval itself (comparing synthesis-claimed direction vs. `module_spine` ground truth)
  is consumed by axis E, but the substrate is owned here.
- **Provenance:** record, per run, how many members carried kME, how many carried kIM, and
  whether a module→outcome direction was supplied (a small `module_spine` coverage summary), so
  a run's sign-inversion number is interpretable against how much signed signal it actually had.
  Surface via the existing per-node timing/telemetry pattern (e.g. a `module_spine_coverage`
  entry), not a new subsystem.
- **Pinned inputs (artifact hygiene SOP):** the Brown module upload used to measure before/after
  is pinned (file + data commit/SHA) alongside the eval output.

## 7. Non-goals / v2 backlog

- Full within-module N×N correlation matrix ingestion (L2) — reserved schema slot only.
- kME p-value / connectivity-floor thresholds beyond carrying kIM (the veto logic is axis B).
- Any change to triage / integration / synthesis behavior (sibling axes).

## 8. Open questions for the plan stage

None blocking — the four seam decisions are settled (L1–L4) and the downstream constraints
(L5, L7, L15, L16, L17/L18) are captured as schema obligations. The plan will sequence:
(1) state model, (2) backend R19 parse/validate/normalize, (3) intake wiring, (4) frontend
column-mapping + direction input, (5) coverage telemetry + measurement substrate, (6) tests.
