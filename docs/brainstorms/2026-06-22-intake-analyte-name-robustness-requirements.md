---
title: "Intake analyte-name robustness (unlock full-module submission)"
type: requirements
status: active
date: 2026-06-22
revised: 2026-06-22
---

# Intake analyte-name robustness — unlock full-module submission

## Problem

The discovery pipeline's intake node cannot cleanly accept real metabolomics/chemistry analyte names, so the full Brown WGCNA module can only be partially submitted (~130 of the named rows). This is **not** an entity-resolution limitation — in the last 130-analyte run, resolution received 130 names and resolved all 130 (112 biomapper + 14 fuzzy + 4 exact). The loss is upstream, in `extract_entities` (`backend/src/kestrel_backend/graph/nodes/intake.py:86-203`) plus a defensive harness filter:

1. **Internal commas fragment names.** Inside a labeled section, intake splits on bare commas (`re.split(r",\s*|\n+", section_content)`, `intake.py:114`), so `12,13-DiHOME` becomes `12` + `13-DiHOME`.
2. **Parenthetical stripping** (`intake.py:201`) drops trailing parentheticals — correct for aliases (`oxalate (ethanedioate)` → `oxalate`, which resolves) but it collapses the small set of names that differ only by an isomer disambiguator (`1-oleoylglycerol (18:1)` vs `(16:0)`).
3. **Harness defensive filter** (`backend/assessment_data/brown_c1_pilot_e2e.py:113-117`) excludes any metabolite/chemistry name containing `(` or `,`, dropping 69 of 153 named rows — most of which would actually resolve fine if submitted.

## What the data actually shows (drives the staged scope)

Of the 69 droppable names: **60 are alias-parentheticals that already strip to a resolvable primary name** (they need no code change — only the harness filter relaxed), **4 are comma-only** (need the comma fix), and **5 are "both" comma+paren** (the comma fix lets most through intact; their `[1]`/`[2]` suffixes keep them distinct). Only a **small subset (lipid-notation isomers like `(18:1)` vs `(16:0)`) genuinely collapse** at intake. So the high-value unlock is cheap; the isomer-preservation machinery would serve only a handful of rows whose real collision rate has never been observed.

Also: the literal module is **217 rows, but 14 Chemistry rows have no ChemName at all** (unnameable). The reachable target is **~203 named rows** (50 proteins + 153 named metabolite/chemistry), not 217.

## Goal

Unlock submission of the **full named Brown module (~203 rows)** with no *fragmentation* loss, via the smallest correct change, then **measure the actual same-CURIE collision rate from a real full-module run** to decide—empirically—whether isomer-preservation + collision reporting is worth building.

## Decisions Made (this brainstorm)

- **Isomer / dedup policy: submit distinct, dedup at resolution** — but **staged**. Phase 1 ships the cheap unlock and *measures* how many rows collapse (at intake or to a shared CURIE); it does not yet build machinery to preserve every isomer. Phase 2 builds isomer-preservation + provenance **only if** the measured collision rate is material.
- **Success bar: accept all named rows + measure collisions.** Full-module run completes; the collision count is measured (lightweight — distinct CURIEs vs submitted rows, computable from the run's `resolved_entities`). Rich per-row collision *reporting* is Phase 2, gated on the measurement.
- **Out of scope:** isomer-aware resolution (mapping isomers to distinct CURIEs; likely impossible — the KG probably lacks isomer-level nodes); a general intake NLP rewrite.

## Requirements — Phase 1 (cheap unlock + measure)

### Intake parsing
- **R1. No fragmentation.** Intake must not split a single analyte name on an internal comma. `12,13-DiHOME` is one entity.
- **R2. Unambiguous structured delimiter.** For labeled-section input, intake treats one analyte per line (newline-delimited) and does not comma-split within a line, so commas inside names are safe. The free-text/chat comma-list path remains supported (best-effort; not regressed).

### Harness / input contract
- **R3. Relax the harness filter + newline-join.** `pick_pilot` stops excluding `(`/`,` names; `build_query` emits one analyte per line within each labeled section (paired with R2). This is part of the real fix, not a workaround — newline delimiting is the correct contract for multi-analyte chemical names. Add a `Chemistry`/`Metabolites` section handling so chemistry rows with names are admitted (verify intake's `section_pattern` covers the labels used).

### Acceptance + measurement
- **R4. Full named-module run completes.** With R1–R3, the harness submits all ~203 named rows and a full-module discovery run completes end-to-end (generous ceiling; pipeline scale is already validated sublinear at 130).
- **R5. Measure the collision rate.** From the run, report a single lightweight number: submitted rows vs distinct resolved CURIEs (i.e. how many rows collapsed, at intake dedup and at resolution). This is the evidence that decides Phase 2. It can be computed from existing `resolved_entities` (e.g. in the harness `coverage()` or offline from the run JSON) — **no new per-row provenance structure required for Phase 1.**

## Deferred to Phase 2 (conditional — only if R5 shows material collisions)

- **Isomer-preservation in intake.** Keep isomer disambiguators so distinct rows don't collapse at parse time. NOTE: the naive "submit full name + alias fallback" approach does **not** work — `extract_aliases` (`intake.py:284-292`) rejects strings with colons/slashes, so `(18:1)` is silently dropped. The real options are (a) extend the alias filter, (b) a smart strip that keeps isomer disambiguators but drops pure abbreviations, or (c) preserve the full original name to resolution. Planning to choose with data.
- **Dedup-at-resolution with provenance** (a many-raw-names→one-CURIE map; net-new — no CURIE-dedup-with-provenance exists today: resolution returns a flat list, triage double-counts same-CURIE).
- **Per-row collision reporting** in the perf/coverage surface (needs the provenance map; decide whether it rides `synthesis_context_stats` or a new `resolution_stats` field written by entity_resolution).

## Success Criteria (Phase 1)

- All ~203 named Brown rows are accepted by intake with **0 fragmentation** (no name split on an internal comma) and no name dropped by the harness filter.
- A full named-module run completes; its output yields the **collision number** (submitted rows vs distinct CURIEs).
- Existing clean-name behavior (24/130-analyte runs — the prior validated harness configs) is unchanged; the chat free-text path is not regressed.
- A recorded **decision**: based on the measured collision count, is Phase 2 warranted?

## Scope Boundaries / Non-Goals

- Not building isomer-preservation, provenance, or collision *reporting* in Phase 1 (deferred, conditional).
- Not isomer-aware resolution; no KG/biomapper changes.
- Not the 14 unnameable Chemistry rows (no ChemName → cannot be submitted).
- Not changing #84–#87 (synthesis caps, register, context-management); this feeds them more entities.

## Open Questions (for Phase 1 planning)

- Exact intake change for R1/R2: prefer newline-splitting when a section is newline-delimited; only comma-split for inline free-text. Confirm both the PRIORITY-1 `section_pattern` and PRIORITY-2 free-text paths behave, and that the lowercase dedup (`intake.py:210-217`) is the cause of the prior 134→130 residual (4 rows) — and whether R1/R2 recover them.
- Does intake's `section_pattern` (matches `Metabolites?|Proteins?|Genes?|Entities?`) need a `Chemistry:` label, or fold chemistry under `Metabolites:`?
- Cheapest place to compute R5's collision number (harness `coverage()` vs offline from `resolved_entities` in the run JSON).

## Context & Evidence

The 69 droppable Brown metabolite/chemistry names (of 153 named; 84 admitted today):

| Category | Count | Examples | Phase-1 handling |
|---|---|---|---|
| comma-only | 4 | `12,13-DiHOME`, `9,10-DiHOME` | R1/R2 (comma fix) |
| paren-only | 60 | `5-methylthioadenosine (MTA)`, `oxalate (ethanedioate)` | R3 (relax filter) — already strip to a resolvable primary; small isomer subset collapses, measured by R5 |
| both | 5 | `diacylglycerol (14:0/18:1, 16:0/16:1) [1]` | R1/R2 (comma fix lets them through; `[1]`/`[2]` keep them distinct) |

Resolution health (130-analyte run): 130 received → 130 resolved, 0 lost — the bottleneck is intake/harness, not resolution. (Note: a separate 134→130 step in that run lost 4 at intake parsing — likely the prose-word filter or lowercase dedup; to be confirmed in planning.)

## Sources & References

- Code: `backend/src/kestrel_backend/graph/nodes/intake.py` (`extract_entities` 86-203; section split 114; paren strip 201; `extract_aliases` 284-292; dedup 210-217), `backend/src/kestrel_backend/graph/nodes/entity_resolution.py` (flat `resolved_entities`, no CURIE-dedup), `backend/assessment_data/brown_c1_pilot_e2e.py` (`pick_pilot` 113-117, `build_query` 121-127, `coverage`).
- Data: `docs/data/Frailty_Multiomic_WGCNA-modules.tsv` (Brown: 50 Protein + 153 Metabolite + 14 Chemistry; 14 Chemistry have no ChemName).
- Related: [[perf-report-context-insight-pr87]] (130-analyte scaling validation), [[brown-c1-48-run-synthesis-overflow-wall]].
