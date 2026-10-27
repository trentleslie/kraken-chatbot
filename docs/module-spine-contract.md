# `module_spine` — signed-weight cross-axis contract (Axis A)

Axis A owns the `module-weight-schema` seam. This is the stable shape that axes **B** (triage),
**D** (entity-semantics), and **E** (synthesis) consume. It threads per-member signed WGCNA weights
and an optional per-module eigengene→outcome direction from the analyte-upload path into
`DiscoveryState`. Source of truth: `backend/src/kestrel_backend/graph/state.py`.

## State fields (both single-writer, set once at intake — no `operator.add` reducer)

- `module_spine: dict[str, ModuleSpine]` — keyed by the **canonical group key**
  (`group.strip().lower()`, the same normalization the selection filter uses). **Absent** for
  classic runs and for uploads with no kME column (R6) — downstream axes MUST treat it as optional.
- `module_spine_coverage: dict` — the per-run coverage summary (R8), present iff `module_spine` is.

### Models

```
MemberWeight   = { name: str,             # canonical run-set name, byte-identical to entity_groups keys
                   kme:  float,           # REQUIRED, ∈ [-1, 1]  (module-eigengene correlation)
                   kim:  float | None }   # OPTIONAL, ≥ 0        (raw intramodular connectivity kWithin; UNBOUNDED)
ModuleDirection= { eigengene_trait_correlation: float,  # ∈ [-1, 1]
                   trait_label: str }
ModuleSpine    = { group: str,                          # display label
                   members: dict[str, MemberWeight],    # weighted members only (name → weight)
                   direction: ModuleDirection | None,
                   correlation: None }                  # reserved v2 within-module N×N matrix slot
```

Key invariants downstream code can rely on:

- `members` holds **only weighted members**. A member with no kME cell is absent from `members` but
  still present in `run_analytes` / `entity_groups`. `MemberWeight.kme` is therefore never `None`.
- A member in two modules gets a **distinct `MemberWeight` per group** (different kME/kIM each).
- **kME is bounded [-1, 1]; kIM is NOT** — kIM is raw kWithin (unbounded, non-negative). Do not
  assume kIM ∈ [-1, 1].
- `member.name` is byte-identical to the `entity_groups` key, so the name→CURIE join is exact.

## Coverage summary (`module_spine_coverage`)

```
{ modules: int,
  members_with_kme: int,
  members_with_kim: int,
  groups_with_direction: int,
  metric_computable: bool,   # True iff ≥1 module has BOTH weighted members AND a direction
  per_group: { <group_key>: { members_with_kme, members_with_kim, direction_supplied } },
  warnings: [str] }          # non-fatal: group-less weights dropped, unmatched directions
```

The **sign-inversion metric** is member-vs-outcome: `sign(kME) × sign(direction) < 0`. It needs both
a member's kME and its module's direction, so `metric_computable` is `False` when no direction was
supplied — the summary flags that rather than reporting healthy coverage.

## Obligations on consuming axes

- **Optionality (all axes):** never require `module_spine`; branch on its presence.
- **Axis E (synthesis) — `trait_label` is untrusted free text.** It is length- and control-char
  validated at ingest, but synthesis MUST still delimit it as data in any prompt (mirror
  `synthesis.py`'s `<user_query>` handling). Ingest validation is not prompt-injection defense.
- **Axis B (triage) — kIM veto:** kIM is carried specifically as the low-n kME veto input; it may be
  absent per member, so guard `kim is None`.

## Provenance / measurement

The pinned before/after upload (member table + ME-trait table) lives in
`backend/assessment_data/module_spine/`; `backend/tests/test_module_spine_pinned_fixture.py` asserts
it reproduces `module_spine` deterministically with `metric_computable = true`.

See `docs/plans/2026-07-16-001-feat-signed-weight-data-spine-plan.md` for the full rationale and the
n=13–15 / D2-null / Arivale caveats (the spine makes the direction signal *available*; it does not by
itself prove strong concordance).
