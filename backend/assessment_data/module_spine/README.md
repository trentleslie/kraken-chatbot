# Signed-weight `module_spine` — pinned measurement substrate (Axis A)

This directory pins the upload used to exercise the **signed-weight data spine** end to end and to
make the **sign-inversion-rate** metric computable and reproducible. It is the measurement substrate
promised by Unit 6 of
`docs/plans/2026-07-16-001-feat-signed-weight-data-spine-plan.md` — it does **not** contain the eval
harness itself (that is Axis E). It guarantees only that signed ground truth exists in state, from a
pinned input, with an honest coverage summary.

## Files

| File | Role |
|------|------|
| `brown_module_members.csv` | Per-member table: `analyte, module, type, kME, kIM`. kME is the signed module-eigengene correlation (∈ [-1, 1]); kIM is raw intramodular connectivity (kWithin, unbounded, ≥ 0). A blank kME (see the `Grey` row) means the member carries no weight — it stays in the run set but is absent from `module_spine.members`. |
| `brown_module_directions.csv` | ME-trait table: `group, eigengene_trait_correlation, trait_label`. The per-module eigengene→outcome direction. **Required** for the sign-inversion metric — without it `metric_computable` is false. |

## Provenance / reproducibility (artifact-hygiene SOP)

- **Representative fixture, not a raw cohort export.** These values are a small, hand-pinned panel in
  the shape of a Brown/Blue WGCNA export (signed kME + kWithin + a single-trait ME-trait table). It
  is committed so the before/after sign-inversion measurement has a fixed, versioned input. When the
  real Brown module export is available, replace these two files (keeping the column layout) and note
  the source data commit SHA here.
- **Pinned input = these two files at this repo commit.** The metric reads `module_spine` +
  `module_spine_coverage`; both are a deterministic function of these files (no seed, no network).
- Determinism is asserted by `backend/tests/test_module_spine_pinned_fixture.py`.

## How the metric reads this

The sign-inversion metric is **member-vs-outcome**:

```
inverted(member) = sign(kME) × sign(eigengene_trait_correlation)  < 0
```

so it needs BOTH a member's signed kME **and** its module's direction. `module_spine_coverage`
reports `members_with_kme`, `members_with_kim`, `groups_with_direction`, and a **`metric_computable`**
flag (True iff at least one module has both weighted members and a direction). For this pinned upload
`metric_computable` is **true**.

## Reproduce locally

```python
import csv
from kestrel_backend.analyte_ingest import validate_and_normalize
from kestrel_backend.graph.nodes.intake import _build_module_spine
from kestrel_backend.config import get_settings

def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

members = [
    {"name": r["analyte"], "group": r["module"], "type": r["type"],
     "kme": r["kME"] or None, "kim": r["kIM"] or None}
    for r in _rows("assessment_data/module_spine/brown_module_members.csv")
]
directions = _rows("assessment_data/module_spine/brown_module_directions.csv")

panel = validate_and_normalize(members, [], get_settings(), module_directions=directions)
spine, coverage = _build_module_spine(panel)
assert coverage["metric_computable"] is True
```
