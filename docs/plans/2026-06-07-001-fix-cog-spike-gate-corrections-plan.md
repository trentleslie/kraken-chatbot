---
title: "fix: Code-on-Graph spike — silent-drop fix, gate corrections, N=100 re-run"
type: fix
status: active
date: 2026-06-07
origin: docs/code-on-graph-phase0-results.md
---

# fix: Code-on-Graph Spike — Silent-Drop Fix, Gate Corrections, N=100 Re-run

## Overview

The Phase-0 recall gate produced a NO-GO that the results doc (`docs/code-on-graph-phase0-results.md`) diagnosed as a **measurement artifact**, not a real failure of iteration. Before a fresh, defensible N=100 re-run, three things must land:

1. **Fix the silent-drop bug** in the gold-set builder, which conflates transient Kestrel failures with genuine non-reachability and lets a build quietly shrink (it collapsed a rebuild to 40 survivors).
2. **Pre-register two gate corrections** the results doc recommended: promote the already-pre-registered **any-one bridge unit** to primary (strict kept as sensitivity), and **narrow the grounding override** to finding-level hallucination (report query-argument leakage as a caveat instead of a hard kill).
3. **Commit** the already-completed gold-set growth / source-pinning / auto-persist work, then re-run the gate at N=100 and write fresh results.

This is a methodology-hardening pass on a **standalone spike** (`backend/tests/code_on_graph_spike/`). Nothing in the production discovery pipeline changes.

## Problem Frame

The original gate (`gate_recall.py`) kills the idea under two rules the results doc argues are mis-specified for multi-bridge DrugMechDB gold:

- **Strict bridge unit** (`kestrel_rest.py:path_contains_all` — a hit requires *all* gold interior nodes in one returned path). For 3-bridge `random` items, Kestrel rarely holds the exact curated path, so both arms fail ~all 70 (2/70 vs 3/70). That concordant-miss wash dilutes the lift and starves McNemar of discordant pairs → strict NO-GO.
- **Grounding override** (`gate_recall.py:45` — `hallucinated > 0 → NO-GO`, where `hallucinated` sums *query-argument* leakage). The run produced 18 such violations, all on non-winning items; no hit depended on an ungrounded query. R9's actual intent is to block *fabricated findings* (structurally ~0 here), not incidental query-steering leakage. As-is, any re-run is near-certain to NO-GO on this override regardless of recall — uninformative.

Separately, the gold-set builder (`gold_set.py:build_random_slice`) caught post-retry transient exceptions via `asyncio.gather(return_exceptions=True)` and treated them identically to a genuine "not reachable" filter, silently dropping records. A naive full rebuild collapsed from the expected ~70+ survivors to 40. This makes the fixture non-reproducible and must be fixed before any future rebuild.

See origin: `docs/code-on-graph-phase0-results.md` (§5 grounding analysis, §6 diagnosis, §7 recommendation).

## Requirements Trace

- **R1.** The gold-set builder never silently returns fewer than the requested N due to transient errors — it retries, and if it still cannot meet N it fails loudly (or reports the shortfall), distinguishing genuine non-reachability (a legitimate filter) from transport failure.
- **R2.** Each arm records *both* bridge-unit metrics per item (strict and any-one) in a single pass; no post-hoc re-scoring step (the lost run hit a re-scoring bug computing the secondary after the fact).
- **R3.** The gate computes its verdict on a **configurable primary bridge unit** defaulting to any-one, and reports the other metric as a sensitivity line.
- **R4.** The grounding override hard-fails only on **finding-level hallucination** (a hit whose recovering path contains a CURIE never returned by an executed call). Query-argument leakage is preserved and reported as a caveat, not a kill.
- **R5.** All threshold values that are *not* part of the documented corrections (lift bars `0.15`/`0.50`, `alpha`, `n_target`, seed, turn cap) remain unchanged; the config re-freeze documents exactly what changed and why.
- **R6.** The completed gold-set growth / pinning / auto-persist work is committed; the gate is re-run at N=100 with results persisted to a durable artifact; a fresh results doc supersedes the N=90 numbers.

## Scope Boundaries

- No change to the production discovery pipeline (`backend/src/kestrel_backend/graph/**`). The spike remains standalone (`integration.py` is touched only after a PROCEED, per the original plan).
- No change to the **lift thresholds, alpha, N, seed, or turn cap** — only metric *priority* and override *scope* change.
- No new fractional-recovery metric — the primary becomes the already-pre-registered any-one unit, not a newly invented fractional threshold (see Key Technical Decisions).
- No EITL precision arm, crosswalk ETL, or reproducibility manifest — those remain Phase-1 work in the original spike plan (`docs/plans/2026-06-03-002-feat-code-on-graph-spike-plan.md`).

### Deferred to Separate Tasks

- **Phase-1 methodology** (DrugMechDB-scale gold set, EITL expert precision, fairness guard, hashed manifest): only on a PROCEED, per the original staged plan.
- **Hardening `_post` timeout / concurrency** for depth-5 reachability beyond the silent-drop contract: optional follow-up if the re-run hits another latency storm (the loop already retries 2× via `_post`).

## Context & Research

### Relevant Code and Patterns

- `backend/tests/code_on_graph_spike/gold_set.py` — `build_random_slice` (lines 43-64), `_evaluate` (32-40): the silent-drop site.
- `backend/tests/code_on_graph_spike/kestrel_rest.py` — `path_contains_all` / `any_path_recovers` (135-143): the strict bridge unit; `is_grounded` (146-158): the grounding membership check; `_post` (56-68): existing 2× transient retry then raise.
- `backend/tests/code_on_graph_spike/baseline.py:35` and `iterate_loop.py:189` — both arms set `hit = any_path_recovers(...)`.
- `backend/tests/code_on_graph_spike/iterate_loop.py:166-168` — per-turn `grounding_violations` accumulation (query-argument leakage).
- `backend/tests/code_on_graph_spike/recall_scorer.py` — `build_table` reads `r["hit"]`; would read the configured primary.
- `backend/tests/code_on_graph_spike/gate_recall.py` — `evaluate_gate` (45-54): the verdict lattice, including the grounding override.
- `backend/tests/code_on_graph_spike/config.py` — frozen pre-registration; already updated this session (`n_target=100`, pinned DrugMechDB SHA).
- `backend/tests/code_on_graph_spike/run_phase0.py` — orchestrator; already auto-persists results to `runs/` this session.

### Institutional Learnings

- `docs/code-on-graph-phase0-results.md` — the full diagnosis and both-metric tables this plan operationalizes.
- Global SOP (this session): expensive-to-reproduce runs must auto-persist by default — already applied to `run_phase0.py`; the re-run unit relies on it.
- The lost N=90 run's re-scoring bug (iterate intermediates stored in `runs[]`, not top-level) motivates R2's "record both metrics inline" decision.

## Key Technical Decisions

- **Record both bridge-unit metrics inline; gate on a configurable primary.** Each arm emits `hit_strict` and `hit_any` per item in one pass. Rationale: eliminates the post-hoc re-scoring bug class that corrupted the lost run, and makes the sensitivity report free.
- **Primary = any-one (promote the pre-registered secondary), not a new fractional metric.** Any-one was already pre-registered as the secondary, so re-prioritizing it (with the documented measurement-artifact justification) is a defensible correction, not post-hoc metric invention. A fractional-threshold metric would be brand-new and weaken the pre-registration claim. Strict is retained as the sensitivity line.
- **Narrow the grounding override to finding-level hallucination.** Hard-fail only if a *hit's recovering path* contains a CURIE that was never returned by an executed call (structurally ~0 by construction, but checked defensively). Keep counting query-argument leakage and report it as a caveat. Rationale: matches R9's actual intent (no fabricated findings) and stops an incidental-leakage rule from masking a legitimate recall signal.
- **Silent-drop contract: never quietly shrink.** Transient post-retry failures are retried at the slice level and, if N still cannot be met, surface as a loud failure or an explicit shortfall report — distinct from genuine non-reachability, which remains a silent filter. Rationale: a gold-set build that returns 40 when asked for 80 must not look like success.
- **Re-freeze is documented, threshold-preserving.** Only metric priority and override scope change; the config docstring and a results-doc addendum record the diff and rationale. Lift bars, alpha, N, seed, turn cap unchanged.

## Open Questions

### Resolved During Planning

- *Re-run as-is, or pre-register corrections first?* → Pre-register both corrections (user decision). As-is is near-certain NO-GO on the grounding override (uninformative).
- *Which primary bridge unit?* → Promote the pre-registered any-one to primary; keep strict as sensitivity (avoids inventing a new metric).
- *Fix the bug before or independently of the re-run?* → Independent of the re-run (the N=100 fixture already exists and was verified append-only); fixed for future builds and as recommended hygiene.

### Deferred to Implementation

- Exact field names / signatures for the dual-metric records and the finding-level grounding flag — settle against the real code.
- Whether the silent-drop fix raises vs. returns a structured shortfall — decide once the retry wrapper is in place and tests express the contract.
- Whether the re-run needs concurrency/timeout tuning — decide only if it hits another latency storm; the loop already retries.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

Verdict inputs after the corrections (decision matrix — what changes vs. the current gate):

| Gate input | Current (frozen) | After corrections |
|---|---|---|
| Primary bridge unit | strict (all interior, one path) | **any-one** (any interior); strict → sensitivity |
| Recall lift / McNemar | computed on strict `hit` | computed on **primary (any-one)** `hit` |
| Grounding override | `sum(query-arg violations) > 0 → NO-GO` | **`finding_level_hallucinations > 0 → NO-GO`**; query-arg leakage → reported caveat |
| Lift bars / alpha / N / seed | 0.15 / 0.50 / 0.05 / 90→100 / fixed | **unchanged** |

Verdict lattice, unchanged in shape (`gate_recall.py:45-54`), but fed the corrected inputs:

    finding_level_hallucinations > 0   -> NO-GO (override; contract intact, now correctly scoped)
    N < powered-N                      -> INCONCLUSIVE
    discordant flapped across K        -> INCONCLUSIVE
    primary lift < bar OR McNemar n.s. -> NO-GO
    else                               -> PROCEED-TO-PHASE-1

**Expectation-setting (critical):** under the lost N=90 any-one numbers (lift +0.144, recover-frac ≈ 0.22), `lift_ok` is **False** against both the 0.15 absolute and 0.50 relative bars — even though McNemar is highly significant. So the corrected gate can still return **NO-GO on the lift threshold** ("significant but sub-threshold"). The corrections make the re-run *informative*; they do not preordain PROCEED. The 10 added random items (N=100) may shift the magnitude either way.

## Implementation Units

- [ ] **Unit 1: Commit the completed gold-set growth, pinning, and auto-persist work**

**Goal:** Bank the verified, already-done changes before adding more.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Modify (commit only): `backend/tests/code_on_graph_spike/config.py`, `backend/tests/code_on_graph_spike/drugmechdb.py`, `backend/tests/code_on_graph_spike/gold_set.py`, `backend/tests/code_on_graph_spike/run_phase0.py`, `backend/tests/fixtures/code_on_graph_spike/gold_set.json`
- Add: `docs/code-on-graph-phase0-results.md`

**Approach:**
- Group as one coherent commit: gold set 90→100 (append-only, first 90 byte-identical), DrugMechDB pinned to `aef2242` + SHA-scoped cache, `n_target=100`, `run_phase0` auto-persist, and the N=90 results doc. Land on `feat/code-on-graph-spike` (PR workflow per repo CLAUDE.md — no direct deploy).
- The untracked `docs/data/` and the biomapper brainstorm are unrelated; do not include them.

**Patterns to follow:** repo commit conventions (`feat`/`fix`/`test(code-on-graph): …` as in recent history).

**Test scenarios:** Test expectation: none — commit of already-verified work (the append-only byte-identity and 13 passing tests were confirmed this session).

**Verification:** Working tree clean for the listed files; branch history shows one descriptive commit; `git show --stat` lists exactly the intended files.

---

- [ ] **Unit 2: Fix the silent-drop in `build_random_slice`**

**Goal:** A gold-set build can never silently shrink below the requested N due to transient Kestrel failures.

**Requirements:** R1

**Dependencies:** None (independent of the gate units)

**Files:**
- Modify: `backend/tests/code_on_graph_spike/gold_set.py`
- Test: `backend/tests/code_on_graph_spike/test_gold_set.py`

**Approach:**
- Separate the two outcomes currently collapsed in `build_random_slice`: a genuine non-reachable/`None` result (legitimate silent filter) vs. a post-retry transport exception (must not be treated as "not reachable").
- On a transient exception for a record, retry that record a bounded number of times; if it still fails, do not drop it silently — either raise a loud build error or accumulate a structured shortfall and fail when the target N cannot be met with genuine survivors. Decide raise-vs-report once the retry wrapper exists (deferred).
- Preserve reproducibility: survivors still appended in shuffled order; genuine filters still reduce the pool exactly as before.

**Execution note:** Add a characterization test for the current happy-path behavior first, then introduce the error/shortfall handling test-first.

**Patterns to follow:** `kestrel_rest.py:_post` transient-exception set (`ReadTimeout`/`ConnectTimeout`/`RemoteProtocolError`) and its retry/backoff shape; `test_gold_set.py` fake-rest fixtures.

**Test scenarios:**
- Happy path: a fake rest where all records resolve+reach → returns exactly N survivors in shuffled order (unchanged behavior).
- Edge case: some records genuinely non-reachable (`_evaluate` → None) → silently filtered; build still reaches N from later survivors.
- Error path: a record's reachability call raises transient errors on every attempt → it is retried, then the build fails loudly / reports a shortfall rather than returning N-1 survivors.
- Error path: enough transient failures that N genuine survivors cannot be collected → build raises/reports, never returns a short list as if successful.
- Integration: mixed genuine-miss + transient-fail in one chunk → genuine misses filter, transient fails retried; final count and order correct.

**Verification:** New tests pass; a forced transient-failure run does not return a short survivor list silently; the existing append-only build path still yields identical survivors for a healthy rest.

---

- [ ] **Unit 3: Dual-metric scoring — record strict + any-one, gate on configurable primary**

**Goal:** Both bridge-unit metrics are recorded per item in one pass; the gate's verdict is computed on a configurable primary (default any-one), with the other reported as sensitivity.

**Requirements:** R2, R3, R5

**Dependencies:** None (can precede or follow Unit 2)

**Files:**
- Modify: `backend/tests/code_on_graph_spike/kestrel_rest.py` (add an any-one recovery helper alongside `path_contains_all`)
- Modify: `backend/tests/code_on_graph_spike/baseline.py`, `backend/tests/code_on_graph_spike/iterate_loop.py` (record `hit_strict` and `hit_any`)
- Modify: `backend/tests/code_on_graph_spike/recall_scorer.py` (build the table from the configured primary)
- Modify: `backend/tests/code_on_graph_spike/gate_recall.py` (verdict on primary; emit sensitivity line)
- Modify: `backend/tests/code_on_graph_spike/config.py` (add `primary_bridge_unit` defaulting to any-one; re-freeze docstring)
- Test: `backend/tests/code_on_graph_spike/test_baseline.py`, `test_iterate_loop.py`, `test_recall_gate.py`, `test_config.py`

**Approach:**
- Add an "any-one interior recovered" predicate next to the strict `path_contains_all`; compute both per item where `hit` is set today (`baseline.py:35`, `iterate_loop.py:189`).
- Keep a single canonical `hit` field semantics by routing it through the configured primary, OR carry `hit_strict`/`hit_any` explicitly and have the scorer select — settle the exact field shape in implementation (deferred), but both metrics must be present on every record.
- Scorer and gate read the primary for the McNemar table and lift; the gate output adds a sensitivity block with the non-primary metric's recall/discordant counts.

**Patterns to follow:** existing `any_path_recovers` / `path_contains_all` structure in `kestrel_rest.py`; `recall_scorer.build_table` cell convention; `gate_recall.evaluate_gate` output dict shape.

**Test scenarios:**
- Happy path: item where a path recovers any-but-not-all interior → `hit_any=True`, `hit_strict=False`; primary=any-one counts it a hit, strict sensitivity does not.
- Happy path: item recovering all interior → both True.
- Edge case: empty paths → both False.
- Edge case: `primary_bridge_unit` set to strict → scorer/gate reproduce the original strict verdict exactly (back-compat).
- Integration: a small fixture of mixed items → gate verdict on any-one + a sensitivity line for strict, both internally consistent with the per-item flags.

**Verification:** With primary=strict the gate output matches the pre-correction numbers on a fixture; with primary=any-one the verdict uses any-one and reports strict as sensitivity; both metrics present on every arm record.

---

- [ ] **Unit 4: Narrow the grounding override to finding-level hallucination**

**Goal:** The gate hard-fails only when a hit depended on an ungrounded query; query-argument leakage is reported as a caveat, not a kill.

**Requirements:** R4, R5

**Dependencies:** Unit 3 (shares `gate_recall.py` and `config.py` edits — sequence after to avoid churn)

**Files:**
- Modify: `backend/tests/code_on_graph_spike/iterate_loop.py` (record a finding-level hallucination flag/count per item: does the recovering path contain any CURIE never in `returned_curies`?)
- Modify: `backend/tests/code_on_graph_spike/gate_recall.py` (override keys off finding-level count; query-arg leakage moved to a reported caveat field)
- Modify: `backend/tests/code_on_graph_spike/config.py` (document the override-scope change in the re-freeze docstring)
- Test: `backend/tests/code_on_graph_spike/test_iterate_loop.py`, `test_recall_gate.py`

**Approach:**
- In the iterate arm, when a hit is recorded, verify every CURIE on the recovering gold path was in `returned_curies` (it should be, by construction — flag any exception as finding-level hallucination).
- Keep the existing per-turn `grounding_violations` (query-argument leakage) intact; surface it in the gate output as an advisory caveat alongside cost.
- Gate override: `finding_level_hallucinations > 0 → NO-GO` replaces `sum(grounding_violations) > 0 → NO-GO`.

**Patterns to follow:** `is_grounded` / `returned_curies` tracking in `iterate_loop.py`; the advisory-vs-kill split already modeled by `cost_advisory` in `gate_recall.py`.

**Test scenarios:**
- Happy path: hit whose recovering path is fully grounded, with nonzero query-arg leakage → no override; leakage reported as caveat; verdict proceeds on recall.
- Error path: a (constructed) hit whose recovering path contains a CURIE never returned → `finding_level_hallucinations > 0` → NO-GO override fires.
- Edge case: zero leakage and zero finding-level → override silent; caveat field present and zeroed.
- Integration: records mixing leakage on non-winning items + clean wins → gate output shows the wins counted, override not fired, leakage caveat populated (mirrors the N=90 situation).

**Verification:** On a fixture replicating the N=90 leakage pattern (18 query-arg violations, all on non-winning items, zero finding-level), the corrected gate does NOT fire the override and reports leakage as a caveat; a synthetic finding-level hallucination still triggers NO-GO.

---

- [ ] **Unit 5: Re-freeze pre-registration and refresh the results doc**

**Goal:** The pre-registration record reflects exactly the two corrections (and only those), with rationale; the results doc frames the N=100 re-run.

**Requirements:** R5, R6

**Dependencies:** Units 3, 4

**Files:**
- Modify: `backend/tests/code_on_graph_spike/config.py` (freeze docstring: enumerate what changed — primary metric, override scope — and what did not — lift bars, alpha, N, seed, turn cap)
- Modify: `docs/code-on-graph-phase0-results.md` (addendum: corrections pre-registered, why, and that the N=90 numbers are superseded by the forthcoming N=100 run)

**Approach:**
- Treat this as a fresh, documented re-freeze: the git commit timestamp on `config.py` is the freeze proof (as the file's own header states). The addendum makes the pre-registration diff auditable.
- Do not alter any threshold value here — documentation + freeze only.

**Test scenarios:** Test expectation: none — documentation and a config docstring/freeze change (the behavioral config field added in Unit 3 is tested there).

**Verification:** `config.py` docstring enumerates the corrections and the unchanged thresholds; results-doc addendum is present and links the corrections to their rationale; no threshold value changed in the diff.

---

- [ ] **Unit 6: Re-run the N=100 gate and write fresh results**

**Goal:** A durable, reproducible N=100 verdict under the corrected gate, with a results doc that supersedes the N=90 numbers.

**Requirements:** R6

**Dependencies:** Units 3, 4, 5

**Files:**
- Run (no source change): `backend/tests/code_on_graph_spike/run_phase0.py` → artifact under `backend/tests/code_on_graph_spike/runs/`
- Add/Modify: `docs/code-on-graph-phase0-results.md` (replace N=90 numbers with the N=100 verdict, both metrics, leakage caveat)

**Approach:**
- This unit is **execution-deferred** (belongs to `ce:work`, not this plan): the live run costs ~hours of Kestrel + SDK and may hit the latency storm seen this session. It auto-persists (SOP). Capture the saved artifact path in the results doc.
- Interpret on the primary (any-one): report verdict, McNemar, per-stratum lift, strict sensitivity, and the query-arg leakage caveat. Be explicit if the outcome is "significant but sub-threshold → NO-GO" (a live possibility per the expectation-setting note).

**Test scenarios:** Test expectation: none — execution + interpretation, not a code unit. (Correctness of the gate it exercises is covered by Units 3–4 tests.)

**Verification:** A result JSON exists under `runs/`; the results doc cites it, reports both metrics + the leakage caveat, and states the verdict with its driver (lift bar vs. significance vs. override).

## System-Wide Impact

- **Interaction graph:** Changes are confined to the spike package. `gate_recall.py` and `config.py` are touched by both Units 3 and 4 — sequence 3 → 4 to avoid churn. No production graph node or state contract is affected.
- **Error propagation:** Unit 2 changes a *silent filter* into a *loud failure* for transport errors — intended. Verify no caller of `build_random_slice` depends on the swallow-and-continue behavior (only the gold-set build path uses it).
- **State lifecycle risks:** None persistent — the gold-set fixture is already built and committed (Unit 1). The re-run only writes new `runs/` artifacts.
- **API surface parity:** The dual-metric `hit_strict`/`hit_any` shape must be consistent across both arms and the scorer; an inconsistency would silently bias the table. Covered by Unit 3 integration tests.
- **Unchanged invariants:** Lift thresholds, alpha, N, seed, turn cap, the verdict-lattice *shape*, and the production pipeline are explicitly unchanged. Only metric priority and override scope move.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Corrected gate still NO-GO on the lift bar (any-one lift +0.144 < 0.15 at N=90) | Set expectation up front (High-Level Technical Design); the re-run is for an *informative* verdict, not a guaranteed PROCEED; N=100 may shift magnitude. |
| Re-run hits the same Kestrel latency storm (silent-drop class) | Auto-persist (SOP) enables resume; Unit 2's loud-failure contract surfaces problems; consider running when Kestrel is healthy. Note the iterate loop already retries via `_post`. |
| Finding-level grounding check mis-implemented (false-pass real fabrication) | Unit 4 includes a synthetic finding-level hallucination test that must still trigger NO-GO. |
| Dual-metric change accidentally alters the strict (back-compat) numbers | Unit 3 includes a `primary_bridge_unit=strict` test that must reproduce the pre-correction verdict on a fixture. |
| Pre-registration integrity questioned (changing the gate after seeing results) | Only re-prioritize an already-pre-registered metric + narrow an override to its stated intent; no threshold tuning; document the diff (Unit 5) with the freeze-timestamp proof. |

## Documentation / Operational Notes

- Per repo CLAUDE.md, all work lands via commits on `feat/code-on-graph-spike` then a PR to `main` (or `dev` first); no direct deploy. Greptile review runs on the PR.
- The full backend test suite has a known ~16-18 pre-existing failures in aggregate; validate spike changes by running the affected `code_on_graph_spike` test files individually (use `python -m pytest`, not bare `uv run pytest`, per the venv-path-spaces note in CLAUDE.md).

## Sources & References

- **Origin document:** [docs/code-on-graph-phase0-results.md](docs/code-on-graph-phase0-results.md)
- Original spike plan: `docs/plans/2026-06-03-002-feat-code-on-graph-spike-plan.md`
- Feasibility: `docs/code-on-graph-feasibility.md`
- Gate / scoring code: `backend/tests/code_on_graph_spike/{gate_recall,recall_scorer,kestrel_rest,iterate_loop,baseline,gold_set,config,run_phase0}.py`
