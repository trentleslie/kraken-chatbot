# `module_spine` contract (Axis A — the `module-weight-schema` seam)

Axis A threads signed within-module structure from the analyte-upload path into pipeline state as
`DiscoveryState.module_spine`. This document is the stable shape that axes **B (triage)**,
**D (entity-semantics)**, and **E (synthesis)** consume. Axis A is emit-only: it changes no
triage / integration / synthesis behavior.

## When it is present

`module_spine` is set **once at intake** (plain single-writer field, no `operator.add` reducer) and
is present **only** when the upload carried a mapped `kME` column that produced at least one
weighted member. It is **absent** for classic runs, single-entity runs, and no-kME uploads —
downstream code MUST treat it as optional (`state.get("module_spine")`), never required (R6).

## Schema

```
DiscoveryState.module_spine: dict[str, ModuleSpine]   # key = canonical group (group.strip().lower())

ModuleSpine:
  group:      str                          # display label (first-seen verbatim group)
  members:    dict[str, MemberWeight]      # canonical name -> weight; ONLY weighted members
  direction:  ModuleDirection | None       # optional eigengene -> outcome direction
  correlation: None                        # reserved v2 within-module N×N matrix slot (stays None)

MemberWeight:
  name: str                                # canonical run-set name, byte-identical to entity_groups keys
  kme:  float  in [-1, 1]  (REQUIRED)      # signed module-eigengene correlation
  kim:  float | None,  >= 0                # raw intramodular connectivity (kWithin); UNBOUNDED

ModuleDirection:
  eigengene_trait_correlation: float in [-1, 1]
  trait_label: str
```

### Load-bearing invariants

- **`members` holds only weighted members.** A member with no `kME` cell is absent here but still
  lives in `run_analytes` / `entity_groups`. "Missing kME" therefore never appears as `kme = None`.
- **`MemberWeight.name` is byte-identical to `entity_groups` keys** (the first-seen canonical run-set
  name), so a downstream name→CURIE join cannot miss silently.
- **kME vs kIM validation differ.** kME is a correlation, bounded [-1, 1] (with a ±1e-6 epsilon
  clamp). kIM is raw `kWithin` — unbounded and non-negative (`>= 0`). Do **not** assume kIM ∈ [-1, 1].
  kIM exists as the downstream **low-n kME veto** input (a high |kME| on a weakly-connected member is
  suspect); axis B owns the veto threshold, not this schema.
- **Direction is per-module and optional.** It is load-bearing for the sign-inversion metric
  (`member-vs-outcome = sign(kME) × sign(direction)`); a module without it still runs, but the metric
  is uncomputable for that module.

## Coverage summary — `module_spine_coverage`

Emitted alongside `module_spine` (also single-writer, plain field). Makes the sign-inversion metric
interpretable and flags when it is uncomputable rather than silently reporting healthy coverage:

```
module_spine_coverage:
  groups: { <group_key>: { members_with_kme: int, members_with_kim: int, direction_supplied: bool } }
  total_members_with_kme: int
  total_members_with_kim: int
  directions_supplied:    int
  metric_computable:      bool   # True iff some module has kME members AND a direction
  warnings:               list[str]   # group-less weights dropped, unmatched directions, ...
```

## Overclaim guardrails (carry these into any axis that reads the spine)

The signed spine makes the direction signal *available*; it does not by itself prove strong
concordance. D1 rank concordances are n=13–15 (suggestive — rules out *strong*, not *moderate*); the
solid result is the D2 null (956 draws); Arivale is a wellness cohort, so non-coherence with Brown ≠
Brown being wrong. Frame direction confidence as evidence-strength, never a probability.

## Reproduce the substrate

`backend/assessment_data/module_spine/` pins a representative Brown/Blue panel + direction table;
`backend/tests/test_module_spine_pinned_fixture.py` reproduces a deterministic `module_spine` through
the live ingest path.
