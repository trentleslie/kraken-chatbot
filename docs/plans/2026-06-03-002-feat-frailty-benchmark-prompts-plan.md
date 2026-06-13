---
title: "feat: Frailty multiomic benchmark prompts for the existing pipeline"
type: feat
status: active
date: 2026-06-03
origin: docs/brainstorms/frailty-multiomic-benchmark-requirements.md
---

# feat: Frailty multiomic benchmark prompts for the existing pipeline

## Overview

Add ~6 realistic benchmark prompts — one per curated frailty WGCNA module — to the existing
assessment harness. Each prompt presents the module's top-connectivity analytes in the **format the
existing intake regex already parses**, so the current pipeline + structural checks + (separately
run) LLM-judge exercise the pipeline on real multiomic signatures. No new scoring machinery.

## Problem Frame

`backend/assessment_data/queries.json` holds 10 small synthetic queries; none exercises the pipeline
on real multi-analyte multiomic module inputs (the actual target workload, per the production prompt
history). The frailty WGCNA dataset gives real per-module analyte sets to turn into prompts.
(see origin: `docs/brainstorms/frailty-multiomic-benchmark-requirements.md`)

## Requirements Trace

- **R1.** ~6 prompts (Brown, Blue, Turquoise, Black, Green, Midnightblue), each the module's top-25
  analytes by `IntramodularConnectivity`, `Chemistry` analytes excluded.
- **R2.** Each prompt uses the existing intake labeled-section format (`Proteins:` GeneSymbols /
  `Metabolites:` names), newline-delimited, frailty-framed preamble.
- **R3.** Metabolite names are normalized so they parse correctly and resolve better (strip `*`/`**`
  flags; **eliminate internal commas** that would otherwise fragment the parser's comma-split).
- **R4.** Each case carries a **small curated `expected_curies`** subset (not names) for the recall
  check (proteins→human NCBIGene, ~5 canonical metabolites→CHEBI). *(Corrects the origin doc, which
  assumed recall used `expected_entities` names — verified: `report.py` feeds the recall check
  `metadata.get("expected_curies")`, exact-match CURIEs.)*
- **R5.** Cases plug into the existing harness with **no harness code changes** — new `queries.json`
  entries + a new `path_type` value.
- **R6.** A committed **parse-verification test** proves each prompt's intake extraction yields the
  intended analytes (no comma fragmentation, Chemistry excluded).

## Scope Boundaries

- No objective enrichment-recovery / GSEA / ORA / R-environment work; notebooks stay reference-only.
- No pipeline or intake-regex changes — prompts conform to what exists.
- No full CURIE derivation for all analytes — only a small high-confidence subset (R4).
- Not all 17 modules — 6 curated; expandable via the same generator.

### Deferred to Separate Tasks

- Expansion to all 17 modules (rerun the generator).
- Wiring the LLM-judge (`scorer.py`) into `report.py` (today it is run à la carte) — out of scope;
  this plan only adds data + a parse test.
- Objective enrichment-recovery metric (separate future track).

## Context & Research

### Relevant Code and Patterns

- **Input format (authoritative):** `backend/src/kestrel_backend/graph/nodes/intake.py`
  `extract_entities()` — Priority-1 section regex recognizes `Metabolites|Proteins|Genes|Entities`
  headers, then `re.split(r",\s*|\n+", ...)` on the list → **internal commas fragment a name**
  (drives R3). `extract_aliases()` strips parentheticals to a primary name for resolution fallback.
- **Harness contract (verified):** `backend/src/kestrel_backend/assessment/report.py::generate_report`
  runs `run_all_checks(state, baseline_stats=tolerance_bands.get(qhash),
  expected_curies=metadata.get("expected_curies"))`. `metadata` = every `queries.json` field except
  `query` (`runner.py`). Recall check **skips/passes when `expected_curies` is empty**; finding-count
  stability **skips when no baseline** (new queries have none) — both confirmed in `checks.py`.
- **CURIE labeling rule + format:** `tests/fixtures/entity_resolution_hard_variants.json` — gene
  symbols → verified **human `NCBIGene:`**; metabolites → canonical **`CHEBI:`** (`L-`/acid form);
  notes `_ambiguous` exact-match brittleness. Mirror this rule for R4.
- **Existing case format:** `backend/assessment_data/queries.json`
  (`query/path_type/expected_entities/notes`).
- **Data:** `docs/data/Frailty_Multiomic_WGCNA-modules.tsv` (`GeneSymbol`, `ChemName`, `Dataset`,
  `ModuleID`, `IntramodularConnectivity`). Verified: all 6 modules have named analytes; metabolite
  names carry `*` flags + comma-bearing lipid shorthand; Black/Green/Midnightblue are metabolite-only.

### Institutional Learnings

- `docs/solutions/developer-experience/pytest-venv-path-spaces-module-invocation-2026-06-01.md` —
  run tests as `cd backend && uv run python -m pytest …` (module form).
- Backend full-suite has a pre-existing test-isolation bug — validate new tests by running the file
  individually.

## Key Technical Decisions

- **Generator script over hand-authoring** — a small committed script reads the TSV and emits the
  `queries.json` entries; reproducible and trivially expandable to all 17 modules. Rationale:
  25-analyte hand lists are error-prone, and v2 expansion should be one command.
- **Name normalization (R3):** strip `*`/`**`; replace commas *inside* an analyte token (e.g. lipid
  shorthand `(d18:2/23:0, d18:1/23:1)`) with `;` so the comma-split keeps each analyte whole; keep
  the descriptive name (don't collapse all sphingomyelins to one token).
- **Recall = small curated CURIE subset (R4)** — proteins→NCBIGene, ~5 canonical metabolites→CHEBI;
  the rest of the module's analytes carry no expected CURIE (recall measured on a confident subset,
  not punishing on unresolvable lipid names).
- **`path_type: "multiomic-module"`** — new label; no baseline exists, so finding-stability skips
  (intended).

## Open Questions

### Resolved During Planning

- *Does recall use names or CURIEs?* → CURIEs (`expected_curies`), exact-match; origin R4 corrected.
- *New `path_type` / no baseline?* → finding-stability check skips gracefully; safe.
- *Is the LLM-judge auto-run?* → No; `report.py` does structural checks only. Judge is separate;
  out of scope here.

### Deferred to Implementation

- Exact metabolite→CHEBI and gene→NCBIGene CURIE values for the curated subset (look up per analyte
  at implementation; mirror the fixture's verification rule).
- Final top-K (25 vs trimming metabolite-heavy modules) — start at 25; trim only if prompt size
  causes pipeline issues at execution.
- Exact preamble wording.

## Implementation Units

- [x] **Unit 1: Prompt generator + 6 frailty-module prompts in `queries.json`**

**Goal:** Produce the 6 benchmark prompts in the existing intake format and add them to the harness.

**Requirements:** R1, R2, R3, R5

**Dependencies:** None

**Files:**
- Create: `backend/assessment_data/generate_frailty_prompts.py` (reads the TSV, emits entries)
- Modify: `backend/assessment_data/queries.json` (append 6 entries)
- Test: covered by Unit 3's parse-verification test

**Approach:**
- For each of the 6 modules: filter to `Dataset in {Protein, Metabolite}`, sort by
  `IntramodularConnectivity` desc, take top-25. Build a frailty-framed preamble + `Proteins:`
  (GeneSymbols) and/or `Metabolites:` (normalized names) sections, **one analyte per line**.
- Apply the R3 normalization (strip `*`/`**`; replace intra-name commas with `;`).
- Emit `queries.json` entries: `query`, `path_type: "multiomic-module"`, `expected_curies` (filled in
  Unit 2), `notes` (module name + analyte counts; expected biology themes for human reference only).

**Patterns to follow:** existing `queries.json` entry shape; section format from `intake.py`.

**Test scenarios:** `Test expectation: none here — behavior is verified in Unit 3` (this unit is
data + a deterministic generator; correctness is the parse test).

**Verification:** 6 new entries present in `queries.json`; generator re-runs deterministically.

- [x] **Unit 2: Curated `expected_curies` per case**

**Goal:** Give each case a small, high-confidence recall target.

**Requirements:** R4

**Dependencies:** Unit 1

**Files:**
- Modify: `backend/assessment_data/queries.json` (add `expected_curies` to each new entry)

**Approach:**
- Per case, select a handful of unambiguously-resolvable analytes: **proteins → human `NCBIGene:`**
  (mirror the `entity_resolution_hard_variants.json` rule — verified human gene id, not ortholog);
  **~5 canonical metabolites → `CHEBI:`** (L-/acid canonical form). Skip ambiguous/lipid names.
- Keep the subset small and confident; exact-match recall is brittle (fixture documents this).

**Patterns to follow:** `tests/fixtures/entity_resolution_hard_variants.json` (CURIE rule + format).

**Test scenarios:**
- Edge case: every `expected_curies` value matches a `PREFIX:ID` shape (validated in Unit 3).

**Verification:** each new case has a non-empty, well-formed `expected_curies` list of confident IDs.

- [x] **Unit 3: Parse-verification test**

**Goal:** Prove the prompts work with the *existing* intake before they're trusted as benchmark
inputs.

**Requirements:** R6 (and guards R2/R3)

**Dependencies:** Unit 1, Unit 2

**Files:**
- Create: `backend/tests/test_frailty_benchmark_prompts.py`

**Approach:**
- Load the new `queries.json` entries (filter `path_type == "multiomic-module"`), run each `query`
  through `intake.extract_entities()`, and assert the extracted set matches the intended analytes.

**Execution note:** Start by writing this test against the intended analyte lists, then run the
generator until extraction matches — it is the correctness gate for Units 1–2.

**Patterns to follow:** existing intake tests; `cd backend && uv run python -m pytest` (module form).

**Test scenarios:**
- Happy path: each prompt extracts exactly the intended protein GeneSymbols + metabolite names
  (count and membership) for its module.
- Edge case (the R3 risk): a metabolite name with lipid shorthand (e.g. a Green-module sphingomyelin)
  is extracted as **one** entity, not fragmented at an internal comma.
- Edge case: `Chemistry` analytes do **not** appear in any extracted set.
- Edge case: prose preamble words are not extracted as entities (the regex's prose filter holds).
- Validation: every `expected_curies` entry matches `^[A-Za-z0-9.]+:[A-Za-z0-9:.-]+$` and uses
  `NCBIGene:`/`CHEBI:` prefixes.

**Verification:** the test passes when run as an individual file (per the isolation-bug learning).

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Internal commas in metabolite names fragment intake parsing | R3 normalization + the Unit 3 fragmentation test is the gate |
| Exact-match recall brittle / curated CURIE wrong | Keep the subset small + confident; mirror the fixture's verified-id rule; recall on a subset, not all analytes |
| 25-analyte prompts stress the pipeline (slow / the integration-contract failure on sparse findings) | Runtime is an execution concern; treat pipeline failure as a measured outcome (it does not block this data+test deliverable) |
| Origin doc said recall uses names | Corrected here (R4); recall uses `expected_curies` |

## Documentation / Operational Notes

- **Running the suite (usage, not built here):** `runner.py` executes the pipeline + captures state;
  `report.py::generate_report` produces structural checks; the LLM-judge (`scorer.py`) is run
  separately. Live runs make real Claude/Kestrel calls (cost/time) and may hit the integration-input
  contract failure on sparse modules — expected, recorded as outcome.
- Expanding to all 17 modules = re-run `generate_frailty_prompts.py` with the full module list.

## Sources & References

- **Origin document:** [docs/brainstorms/frailty-multiomic-benchmark-requirements.md](docs/brainstorms/frailty-multiomic-benchmark-requirements.md)
- Input format: `backend/src/kestrel_backend/graph/nodes/intake.py`
- Harness: `backend/src/kestrel_backend/assessment/{report,checks,runner,scorer}.py`, `backend/assessment_data/queries.json`
- CURIE rule: `backend/tests/fixtures/entity_resolution_hard_variants.json`
- Data: `docs/data/Frailty_Multiomic_WGCNA-modules.tsv`
