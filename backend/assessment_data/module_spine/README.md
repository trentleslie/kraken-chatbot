# Signed-weight data spine — pinned measurement substrate (Axis A)

This directory pins the inputs used to reproduce the `module_spine` before/after the
sign-inversion-rate metric (the metric eval itself is Axis E, not this axis). Axis A's job is
only to guarantee the *substrate* exists and is reproducible: signed ground truth in state, a
pinned input, and the coverage summary.

## Files

| File | Purpose |
|------|---------|
| `brown_signed_weights.csv` | Representative per-analyte panel with signed `kME` + raw `kIM` columns, across two modules (Brown, Blue). Maps to `structured_analytes` after column-mapping (Unit 5). |
| `brown_module_directions.csv` | The (separately-exported) per-module eigengene→outcome direction table. Maps to `module_directions`. |

## Provenance / reproduction pin (artifact-hygiene SOP)

- **Shape** mirrors a Frailty WGCNA export: `kME` is the signed module-eigengene correlation
  (bounded [-1, 1]); `kIM` is raw intramodular connectivity (`kWithin`, unbounded, `>= 0`, note the
  40.1 value — it must ingest, not be rejected); the direction table carries the signed ME-trait
  correlation + trait label per module.
- These are **representative sample values** chosen to exercise every spine code path
  (multi-module, mixed sign, a member with no `kIM`, and both modules carrying a direction so the
  sign-inversion metric is computable) — they are NOT raw cohort measurements.
- **Data pin:** committed into the repo at this path; the reproducing code is
  `backend/tests/test_module_spine_pinned_fixture.py`, which reads these CSVs through the same
  shared R19 gate (`analyte_ingest.validate_and_normalize`) + intake spine builder
  (`graph.nodes.intake._build_module_spine`) the live upload path uses, and asserts a stable
  `module_spine` + `metric_computable = True`.
- When a real Brown export becomes available, drop it in beside these and update the fixture test's
  expected structure; the schema (below) is stable.

## Reproduce

```bash
cd backend && uv run python -m pytest tests/test_module_spine_pinned_fixture.py -v
```

See `docs/module-spine-contract.md` for the `module_spine` schema + coverage-summary contract that
axes B (triage), D (entity-semantics), and E (synthesis) consume.
