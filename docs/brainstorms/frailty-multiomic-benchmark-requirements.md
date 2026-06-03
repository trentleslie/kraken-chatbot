# Frailty Multiomic Benchmark Prompts — Requirements

**Status:** Ready for planning
**Date:** 2026-06-03
**Scope:** Lightweight

## Goal

Add **a handful of realistic benchmark prompts** to the **existing** KRAKEN assessment harness —
each presenting a real set of analytes from a **frailty WGCNA module**, formatted to parse cleanly
through the pipeline's **existing regex-based intake**, and scored by the **existing** checks +
LLM-judge. This replaces the toy hand-written queries with real multiomic signatures that mirror
actual usage (the production prompt history is full of exactly these "here are the analytes
associated with phenotype X — interpret them" queries).

**Explicitly not** a new scoring framework: no objective enrichment-recovery metric, no GSEA/ORA
ground-truth extraction, no R environment, no new scorer. (Those were considered and cut — see
Non-goals.)

## Problem Frame

The current benchmark (`backend/assessment_data/queries.json`, 10 items of
`query/path_type/expected_entities/notes`) uses small synthetic queries (e.g. "What is the
relationship between NAD+ metabolism and cellular aging?"). It doesn't exercise the pipeline on
**real, multi-analyte multiomic module inputs** — the actual target workload. The frailty WGCNA
dataset gives us real analyte sets per module; we turn ~6 of them into benchmark prompts.

## Requirements Trace

- **R1.** Add **~6 benchmark prompts**, one per curated module (Brown, Blue, Turquoise, Black,
  Green, Midnightblue), each the module's **top-25 analytes by `IntramodularConnectivity`**
  (`Chemistry` analytes excluded — no names, not a parseable section).
- **R2.** Each prompt uses the **existing intake labeled-section format** (verified in
  `graph/nodes/intake.py::extract_entities`, Priority-1 pattern): a short frailty-framed preamble
  followed by `Proteins:` (GeneSymbols) and/or `Metabolites:` (names) sections. Analytes are
  **newline-delimited, one per line** (see R3).
- **R3.** Handle messy metabolite names: Metabolon names contain **internal commas, `*`/`**`
  confidence flags, and parenthetical lipid shorthand** (e.g.
  `sphingomyelin (d18:2/23:0, d18:1/23:1)*`). Internal commas would break the parser's comma-split,
  so prompts are **newline-delimited**; decide in planning whether to also **lightly normalize**
  names (strip `*`, keep primary name) or use them **as-is** (more realistic; expect some to not
  resolve). Resolution failures on verbose lipid names are an **acceptable, measured outcome**, not
  a bug.
- **R4.** Each prompt ships with realistic `expected_entities` for the existing entity-resolution
  **recall** check — anchored on the analytes that *should* resolve (proteins by GeneSymbol/UniProt;
  well-known metabolites). Do **not** require recall on verbose Metabolon lipid names unlikely to be
  in the KG.
- **R5.** Cases plug into the **existing harness unchanged** — `runner.py`, `checks.py`
  (completion, entity-resolution recall, hypothesis completeness), `scorer.py` (LLM-judge). Add a
  `path_type` value for these (e.g. `multiomic-module`) and use `notes` to record the module + (for
  human review only) the expected biology themes.

## Scope Boundaries (Non-goals)

- **No** objective enrichment-recovery / pathway-overlap metric; **no** GSEA/ORA ground-truth
  extraction; **no** R/Bioconductor work; **no** new scorer. (The `docs/data/` notebooks remain
  reference material only.)
- **No** changes to the pipeline or its intake regex — prompts must fit the format that exists.
- **No** expected-CURIE derivation from the Arivale annotation files (that namespace-reconciliation
  rabbit hole is out — recall uses names/`expected_entities` as the existing harness already does).
- **Not** all 17 modules — 6 curated; expandable later.

## Context & Resources

- **Input format (authoritative):** `backend/src/kestrel_backend/graph/nodes/intake.py`
  `extract_entities()` Priority-1 section pattern recognizes `Metabolites|Proteins|Genes|Entities`
  headers (optionally `Significant ` / `(context)`), then splits the list on `,` **or** newlines and
  drops prose-like items. (`Chemistry`/`Lipids` are **not** recognized headers.) `extract_aliases()`
  strips parentheticals to a primary name for resolution fallback.
- **Data:** `docs/data/Frailty_Multiomic_WGCNA-modules.tsv` — `GeneSymbol` (proteins), `ChemName`
  (metabolites, all populated for the 6 modules), `Dataset`, `ModuleID`, `IntramodularConnectivity`.
  Module mix verified: Brown/Blue/Turquoise are protein+metabolite; Black/Green/Midnightblue are
  metabolite-only.
- **Existing harness:** `backend/assessment_data/queries.json`,
  `backend/src/kestrel_backend/assessment/` (runner/checks/scorer/variance).
- **Reference only:** `docs/data/MultiomicWGCNA-{GSEA,ORA}-RampDB.ipynb`, Arivale annotation `.xlsx`,
  `docs/discovery-pipeline-requirements.md` (north star — module+connectivity input).

## Key Decisions (resolved)

- Deliverable = **benchmark prompts for the existing pipeline**, scored by the existing harness.
- 6 curated modules, top-25 by connectivity, Chemistry excluded.
- Existing labeled-section input format; **newline-delimited** analytes (avoids the internal-comma
  break).
- GSEA/ORA + Arivale annotations are **reference material**, not scoring inputs.

## Open Questions (for planning)

- **Metabolite-name handling** — use Metabolon names as-is (realistic, some won't resolve) vs. light
  normalization (strip `*`, drop parenthetical shorthand)? Recommendation: newline-delimited +
  normalize only the `*`/`**` flags, keep the rest real.
- **`expected_entities` per case** — which analytes to assert for recall (proteins + well-known
  metabolites) so the recall check is meaningful but not punishing on unresolvable lipid names.
- **Top-25 vs smaller** — is 25 analytes a good prompt size for the pipeline, or trim to ~10–15 for
  the metabolite-heavy modules? (Runtime/cost per case is real.)
- **`path_type` label** + whether to record expected biology themes in `notes` for human review.

## Success Criteria

- ~6 frailty-module prompts added to `backend/assessment_data/queries.json` in the existing schema,
  each parsing through `extract_entities()` into the intended analytes (verified).
- The existing harness runs them end-to-end and reports entity-resolution recall + hypothesis
  quality, with no harness changes beyond data + an optional `path_type` value.
- Verbose-name resolution failures are visible in the output as a measured signal, not crashes.

## Sources & References

- Input format: `backend/src/kestrel_backend/graph/nodes/intake.py`.
- Data: `docs/data/Frailty_Multiomic_WGCNA-modules.tsv` (+ notebooks/annotations as reference).
- Harness: `backend/assessment_data/queries.json`, `backend/src/kestrel_backend/assessment/`.
- Real-usage evidence: production `kraken_turns` prompt history (multiomic-signature queries).
