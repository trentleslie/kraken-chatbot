---
title: "Synthesis node context overflow silently falls back at module scale"
date: 2026-06-22
category: performance-issues
module: discovery-pipeline/synthesis
problem_type: performance_issue
component: assistant
symptoms:
  - "48-analyte module run reports status=complete, errors=0 but produces no module narrative"
  - "Synthesis node returns in ~3.5s instead of the expected ~150s real LLM call"
  - "Deliverable is an ~882KB raw per-entity dump (fallback_report) with no cross-entity narrative"
  - "Assembled synthesis context reaches ~882K chars (~230K tokens), exceeding the ~200K-token sonnet-4 input window"
  - "Failure is swallowed by a silent except, so nothing surfaces the degradation in the run status"
root_cause: config_error
resolution_type: code_fix
severity: high
related_components:
  - tooling
  - testing_framework
tags:
  - synthesis
  - context-window
  - token-budget
  - silent-fallback
  - langgraph-pipeline
  - module-scale
  - aggregation
---

# Synthesis node context overflow silently falls back at module scale

## Problem

The KRAKEN discovery pipeline's `synthesis` node (`backend/src/kestrel_backend/graph/nodes/synthesis.py`)
assembles all accumulated pipeline state into one LLM context block, then calls the Claude Agent SDK to
write the final report. `assemble_synthesis_context()` emitted **per-entity** disease/pathway/findings
sections with **no size bound**. At module scale — a 48-analyte Brown WGCNA run with 44 well-characterized
entities — the context reached **~882K chars ≈ 230K tokens**, exceeding sonnet-4's ~200K-token input window.
The SDK call failed, a broad `except Exception:` silently substituted `fallback_report(state)` (an 882KB raw
per-entity dump with no module narrative), and the run reported **`status=complete, errors=0`**. The actual
deliverable was never produced and nothing surfaced the degradation.

## Symptoms

- Run reads as a clean success: `status=complete`, `errors=0`.
- Synthesis node ran ~3.5s instead of the expected ~150s (the only external tell, if anyone checks timings).
- Output is an ~882KB raw per-entity dump instead of a ~24KB synthesized module narrative.
- The assessment harness reported "37 findings" — a ~110× under-count of the true ~4,099, hiding the scale
  that caused the overflow.

## What Didn't Work

- **Trusting `status=complete, errors=0`.** The silent `except` masked total failure as success — the run
  looked perfect. This masking is the central lesson, not a side note. (session history)
- **Guessing the failure mode.** Initial uncertainty over SDK-crash vs. empty-output vs. transient flakiness.
  Resolved not by a paid re-run but by **measuring the persisted failure artifact offline**: the per-section
  char breakdown was findings 58% / disease 21% / pathway 17% (96% of the dump); 4,099 findings. (session history)
- **The initial "~42 findings, manageable" read.** The harness's `coverage()` said 37–42; the real count was
  ~4,099 bullets. `direct_findings` accumulates via an `operator.add` reducer across parallel branches, and the
  harness merged stream deltas with a reducer-blind `merged.update()` (overwrite), so it read only the last
  node's slice — the ~110× under-count. (session history)
- **Capping by character count alone.** First instinct. The real ceiling is **tokens**; chars are only a proxy
  and must be calibrated to the content. Measured **3.80 chars/token** for CURIE/predicate-dense content —
  *below* the 4.0 rule of thumb — so a char cap sized via 4.0 would under-protect.
- **A marker-string assertion that never matched (caught in review, post-merge).** The new U7 acceptance check
  searched `errors` for `"synthesis_degraded"`, a string the actual marker never emits (the real markers say
  `"fell back to deterministic"` / `"LLM call failed"`). The gate was permanently green — a test that could
  never fail.

## Solution

Fixed on `feat/module-aware-synthesis-context` (PR #85, merged to `dev`, Greptile 5/5). Six coordinated parts.

### 1. `SynthesisConfig` with token-derived caps

New sub-model in `pipeline_config.py` (mirrors `BridgeGroundingConfig`, accessed via `@lru_cache
get_pipeline_config()` — `synthesis.py` did not import the config at all before this). `max_context_chars=350_000`
is an explicit **char proxy** for the ~200K-token window (≈100K-token target at 3.5–3.8 chars/token, leaving
headroom for system prompt + output). Plus `max_findings_per_tier=50`, `max_aggregated_diseases=30`,
`max_aggregated_pathways=30`, decoupled `module_mode_min_entities=5` and `min_members_for_recurrence=2`,
`max_member_table_rows=50`.

### 2. Module-aware assembly dispatch

When distinct resolved entities ≥ `module_mode_min_entities`, emit cross-entity aggregation + a bounded member
table instead of per-entity dumps. Wired into **both** `assemble_synthesis_context` and `fallback_report`, so
even the degraded path is bounded.

Before (per-entity, unbounded — inline in `assemble_synthesis_context`):

```python
disease_associations = state.get("disease_associations", [])
sections.append(format_disease_associations(disease_associations))
pathway_memberships = state.get("pathway_memberships", [])
sections.append(format_pathway_memberships(pathway_memberships))
```

After (`_disease_pathway_sections`, dispatched by entity count):

```python
def _disease_pathway_sections(state: DiscoveryState) -> list[str]:
    cfg = get_pipeline_config().synthesis
    disease_associations = state.get("disease_associations", [])
    pathway_memberships = state.get("pathway_memberships", [])
    resolved = state.get("resolved_entities", [])
    novelty_scores = state.get("novelty_scores", [])
    distinct_entities = len({e.curie for e in resolved if e.curie})

    out: list[str] = []
    if distinct_entities >= cfg.module_mode_min_entities:
        candidates = [
            aggregate_shared_diseases(disease_associations, cfg.min_members_for_recurrence, cfg.max_aggregated_diseases),
            aggregate_shared_pathways(pathway_memberships, cfg.min_members_for_recurrence, cfg.max_aggregated_pathways),
            format_member_table(resolved, novelty_scores, disease_associations, cfg.max_member_table_rows),
        ]
    else:
        candidates = [
            format_disease_associations(disease_associations),
            format_pathway_memberships(pathway_memberships),
        ]
    return [s for s in candidates if s]
```

Aggregation dedups by `(entity_curie, key_curie)`, counts *distinct* members, keeps the strongest evidence per
pair, ranks by member count, caps to `max_items`. Single/pair/small queries keep the per-entity shape.

### 3. Capped findings (`format_findings_summary`)

`max_per_tier` param: rank within tier by confidence (high→moderate→low), keep the top N, append
`"… and N more (tier T)"`. `max_per_tier=None` preserves historical unbounded behavior for small callers.

### 4. Context-char backstop (tripwire, not truncator)

```python
if len(context) > cfg.max_context_chars:
    logger.warning(
        "synthesis context %d chars (~%dK est. tokens) exceeds max_context_chars=%d "
        "(~200K-token window) — a per-section cap is likely mis-set",
        len(context), round(len(context) / 3.5 / 1000), cfg.max_context_chars,
    )
```

Logs an *estimated token count* so the warning is interpretable. It does not silently chop the context —
reaching it means a per-section cap is mis-set.

### 5. Visible degradation in `run()`

Before (silent — masks total failure as success):

```python
try:
    text = await query_synthesis(prompt=context, ...)
    report = text
except Exception:
    report = fallback_report(state)   # silent; run still status=complete, errors=0
```

After (records a marker in the additive `errors` channel):

```python
fallback_marker: str | None = None
if HAS_SDK:
    try:
        text, usage_record = await query_with_usage(prompt=context, options=options, node_name="synthesis")
        if text.strip():
            report = text
        else:
            report = fallback_report(state)
            fallback_marker = ("synthesis: LLM returned empty output; fell back to deterministic "
                               "report (possible context overflow)")
    except Exception as e:
        logger.warning("synthesis LLM call failed, using fallback report: %s", e, exc_info=True)
        report = fallback_report(state)
        fallback_marker = (f"synthesis: LLM call failed ({type(e).__name__}); fell back to "
                           "deterministic report")
else:
    report = fallback_report(state)   # SDK absent in dev/test is an environment condition, not a degradation

result: dict[str, Any] = {"synthesis_report": report, "model_usages": [...]}
if fallback_marker:
    result["errors"] = [fallback_marker]   # operator.add reducer → surfaces the degradation
return result
```

(A concurrent feature, PR #84, independently shipped an equivalent marker; on merge, dev's version was adopted
and the duplicate dropped.)

### 6. Reducer-aware harness + machine-checkable acceptance

In `backend/assessment_data/brown_c1_pilot_e2e.py`, `merge_delta` derives the `operator.add` list keys from
`DiscoveryState` annotations and concatenates them across `stream_mode="updates"` deltas (fixes the ~110×
under-count):

```python
def _additive_list_keys() -> set[str]:
    keys: set[str] = set()
    for name, ann in get_type_hints(DiscoveryState, include_extras=True).items():
        if get_origin(ann) is Annotated:
            base, *meta = get_args(ann)
            if operator.add in meta and get_origin(base) is list:
                keys.add(name)
    return keys

def merge_delta(merged: dict, out: dict) -> None:
    for k, v in out.items():
        if k in _ADDITIVE_KEYS and isinstance(v, list):
            merged[k] = merged.get(k, []) + v
        else:
            merged[k] = v
```

Five acceptance checks gate the 48-analyte run: `no_synthesis_degraded`, `synthesis_ran_real_llm` (>30s),
`context_under_token_budget` (< ~100K est. tokens), `report_not_raw_dump` (<10 raw `### CURIE` sections),
`findings_counted_plausibly`. The degradation check was corrected (review P1) to match the **actual** marker:

```python
degraded = [e for e in errors if "fell back to deterministic" in str(e) or "LLM call failed" in str(e)]
```

## Why This Works

- **Module scale is the regime that overflows, and at that scale per-entity detail is the wrong shape anyway.**
  Switching to cross-entity recurrence aggregation + a bounded member table both fits the budget *and* produces
  the narrative the deliverable actually wants. The three caps target the exact overflow contributors
  (findings 58%, disease 21%, pathway 17% = 96%).
- **Caps are sized against tokens, calibrated to measured chars/token (3.5–3.8) for CURIE-dense content**, not
  the generic 4.0 — so the 350K-char budget sits ~2.5× under the 882K/230K that crashed.
- **Failure is now observable.** The fallback marker lands in the `operator.add` `errors` channel, so
  coverage/monitoring/acceptance see the degradation instead of a false success.
- **Both the LLM path and the fallback path are bounded**, so even a degraded run can't re-emit an 882KB dump.

Verification: U7 48-analyte re-run PASS — 156.7s real synthesis, 97.6K-char context ≈ 27.9K tokens, 24KB module
narrative (vs the 882KB dump), 2,735 findings counted. 23 unit tests; zero regressions
(`test_langgraph_prototype.py` 31 fail / 85 pass, identical to the known pre-existing baseline).

## Prevention

- **Bound any LLM context against the model's TOKEN window, not chars.** Chars are a proxy; calibrate
  chars/token to your *actual* content (CURIE/code/structured data runs denser than the 4.0 rule of thumb —
  here 3.5–3.8). Size the cap below the window with headroom for the system prompt and the output.
- **At "module"/batch scale, aggregate — don't dump.** When N entities grow large, per-entity sections both
  overflow the budget and bury the signal. Switch to cross-entity aggregation + a bounded table above an
  explicit entity-count threshold; keep the detailed per-entity shape only for small N.
- **Never let an `except` silently fall back.** A degraded path that returns plausible-looking output while
  reporting success is worse than a crash — it hides. Record degradation in observable state (here, the additive
  `errors` channel). Distinguish environment conditions (SDK absent in dev) from runtime degradation so you
  don't emit false-positive markers. (See the sibling "visible `degraded` flag" guards below.)
- **Diagnose context/scale overflow offline from the persisted artifact.** Size math on a saved failure artifact
  is free and conclusive; paid re-runs to "see what happens" are not. (Reinforces the persist-expensive-artifacts
  SOP.)
- **Make acceptance assertions machine-checkable AND verify the string actually matches the emitted marker.** A
  marker-string assertion that never matches is a permanently-green test. String-coupling across files is
  fragile — keep the assertion's marker text in sync with the producer, and treat "did this gate ever fail?" as a
  question to answer, not assume.
- **Use a reducer-aware merge when consuming LangGraph `stream_mode="updates"` deltas.** A naive `dict.update()`
  clobbers all but the last node's slice of every `operator.add` key. Derive the additive key set from the state
  annotations so a new reducer field can't silently reintroduce the bug.

## Related Issues

- [`best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md`](../best-practices/reliable-long-running-llm-batch-runs-2026-06-07.md)
  — the parent "fail loud / drop-with-disclosure ≠ silently shrink" principle. The silent `except` here is a
  textbook violation of its rule #1; this doc operationalizes the same rule for a new failure class
  (token-window overflow).
- [`best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md`](../best-practices/sdk-stdio-mcp-unavailable-migrate-to-http-data-in-prompt-2026-06-02.md)
  — sibling "visible `degraded` flag, never silent" guard at the SDK layer. Same anti-pattern (a masked failure
  presenting as success), one layer down (MCP transport vs. token budget).
- [`best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md`](../best-practices/discovery-pipeline-one-graph-methods-within-nodes-2026-05-28.md)
  — pipeline node topology the synthesis node lives in. (Note: lists a 10-node graph; the live pipeline is now 12
  nodes — stale node count, not stale guidance.)
- [`best-practices/langgraph-pipeline-production-formalization.md`](../best-practices/langgraph-pipeline-production-formalization.md)
  — the config-with-documented-rationale convention that `SynthesisConfig` follows.
- Plan: `docs/plans/2026-06-22-001-feat-module-aware-synthesis-context-plan.md`. Failure artifact:
  `backend/assessment_data/brown_diagnostic_runs/brown_c1_pilot_20260622T070237Z.json`; verified run:
  `…brown_c1_pilot_20260622T200900Z.json`. GitHub issue #61 (SDK stdio MCP migration) is adjacent, not a duplicate.
```

